"""④ 角色的 LLM 能力。每个能力都是**单遍**的一次调用（不循环），由角色模板（prompts/roles/）驱动：

  plan_queries      改写检索 query：一个节点一次调用，覆盖它分到的全部指标；同时给出「换角度」的备选说法（angles）
  annotate_results  评估检索结果：一个节点一次调用，覆盖它自己探查的全部证据卡；只做标注（写成证据），不增删候选
  self_evaluate     自评是否接活（被点名 / 广播揭榜）：语义上是否胜任，必须给理由
  compose_question  把需求写成自足的求助问题

约束（铁律 3 / 4 / 5）：调用经 `hub.llm` 闸门（开关 / 总预算 / 并发 / 超时），不可用时返回 None / 空，调用方回落规则；
LLM 只能引用通道真实返回的资产 id，编造的一律丢弃；LLM 不直接增删候选，候选合并 / 去重 / 融合打分仍由代码完成。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..prompts import load_prompt
from ..schemas.collab import EvidenceCard, Need
from ..schemas.recall import Evidence, EvidenceKind
from .base import RoleSpec, new_evidence_id

if TYPE_CHECKING:  # pragma: no cover
    from ..collab.hub import CollabHub

_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")
MAX_QUERY_LEN = 200
MAX_QUESTION_LEN = 300
MAX_CANDIDATES_PER_CARD = 8
MAX_ANGLES = 5


def _render(name: str, values: dict[str, str]) -> str:
    """单遍占位符替换：被插入的内容（query / 知识 / 角色模板）不会再被当作模板解析，杜绝占位符注入。"""
    return _PLACEHOLDER.sub(lambda m: values.get(m.group(1), m.group(0)), load_prompt(name))


def role_text(spec: RoleSpec) -> str:
    return load_prompt(spec.template_name) if spec.template_name else f"# 角色：{spec.name}\n\n{spec.description}"


def _dumps(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, indent=2)


def contract_json(hub: "CollabHub") -> str:
    c = hub.env.contract
    it = c.intent
    return _dumps({
        "query": c.query, "domain": it.domain if it else None, "intent_type": it.intent_type if it else None,
        "slots": {n: getattr(c, n).value for n in ("metrics", "caliber", "grain", "time_range", "dimensions", "filters")},
    })


def knowledge_json(hub: "CollabHub", metrics: list[str], limit: int = 8) -> str:
    from ..stages.s3_planning.levels import mentions       # 延迟导入：避免 roles ↔ stages 循环依赖
    items = hub.env.knowledge
    rel = [k for k in items if any(mentions(k, m) for m in metrics)] or items
    return _dumps([{"id": k.id, "kind": k.kind.value, "title": k.title, "content": k.content[:160], "asset_ids": k.asset_ids} for k in rel[:limit]]) if rel else "（无）"


# ── 改写检索 query ───────────────────────────────────────────────────────────
@dataclass
class QueryPlan:
    query: str = ""
    angles: list[str] = field(default_factory=list)
    limit: int | None = None            # LLM 决定召回多少条（已钳制到 [MIN, MAX]）；None = 默认 top-k


def clamp_limit(hub: "CollabHub", v: Any) -> int | None:
    """LLM 给的条数：必须是整数（布尔不算）；钳制到 [RECALL_MIN_LIMIT, RECALL_MAX_LIMIT]。"""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
        return None
    s = hub.settings
    return int(max(s.RECALL_MIN_LIMIT, min(s.RECALL_MAX_LIMIT, v)))


async def plan_queries(hub: "CollabHub", spec: RoleSpec, metrics: list[str], kinds: tuple[str, ...]) -> dict[str, QueryPlan]:
    """指标 → 检索 query + 备选说法。LLM 不可用 / 输出非法 → 空（调用方直接用指标名）。"""
    if not hub.llm.enabled or not metrics:
        return {}
    assignments = [{"metric": m, "channels": sorted(spec.channels), "needs": list(kinds)} for m in metrics]
    prompt = _render("roles/_plan", {
        "role": role_text(spec), "assignments": _dumps(assignments), "contract": contract_json(hub), "knowledge": knowledge_json(hub, metrics),
        "default_limit": str(hub.settings.PREINJECT_RETRIEVAL_TOPK), "min_limit": str(hub.settings.RECALL_MIN_LIMIT), "max_limit": str(hub.settings.RECALL_MAX_LIMIT),
    })
    data = await hub.llm.json("plan", prompt)
    raw = data.get("plans") if data else None
    out: dict[str, QueryPlan] = {}
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        m, q = item.get("metric"), item.get("query")
        if not isinstance(m, str) or m not in metrics or m in out:
            continue
        q = q.strip() if isinstance(q, str) else ""
        if len(q) > MAX_QUERY_LEN:
            q = ""
        if q and m.casefold() not in q.casefold():
            q = f"{m} {q}"                                                   # 必须保留指标名
        angles: list[str] = []
        for a in item.get("angles") if isinstance(item.get("angles"), list) else []:
            a = a.strip() if isinstance(a, str) else ""
            if a and len(a) <= MAX_QUERY_LEN and a.casefold() != m.casefold() and a not in angles:
                angles.append(a)
        limit = clamp_limit(hub, item.get("limit"))
        if q or angles or limit is not None:
            out[m] = QueryPlan(query=q, angles=angles[:MAX_ANGLES], limit=limit)
    return out


# ── 评估检索结果 ─────────────────────────────────────────────────────────────
def _clamp01(v: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return default


@dataclass
class AnnotateResult:
    all_irrelevant: list[EvidenceCard] = field(default_factory=list)       # 被判定全部不相关的卡 → 换角度重试
    more: dict[str, tuple[int, str | None]] = field(default_factory=dict)  # 觉得不够的卡 key → (继续召回的条数, 可选的新 query)


def _more_request(hub: "CollabHub", current: int, raw: Any) -> tuple[int, str | None]:
    """继续召回多少：LLM 建议的条数必须大于当前条数；否则翻倍；都不超过 RECALL_MAX_LIMIT。"""
    cap = hub.settings.RECALL_MAX_LIMIT
    asked = clamp_limit(hub, raw.get("limit")) if isinstance(raw, dict) else None
    limit = asked if asked is not None and asked > current else min(cap, current * 2)
    q = raw.get("query") if isinstance(raw, dict) else None
    q = q.strip() if isinstance(q, str) and 0 < len(q.strip()) <= MAX_QUERY_LEN else None
    return limit, q


async def annotate_results(hub: "CollabHub", agent_id: str, spec: RoleSpec, cards: list[EvidenceCard]) -> AnnotateResult:
    """评估自己探查出的证据卡里的候选资产（只评估还没评估过的）。每条判断写成一条证据（source=llm_filter）挂在卡上，**不增删候选**。
    同时判断每张卡的结果够不够：不够就给出继续召回的条数（`more`）；被判定全部不相关的卡交给调用方换角度重试。"""
    res = AnnotateResult()
    cards = [c for c in cards if c.asset_ids]
    if not hub.llm.enabled or not cards:
        return res
    payload = []
    for c in cards:
        done = hub.judged.setdefault(c.key, set())
        fresh = [a for a in hub.board.assets_of(c) if a.asset_id not in done][:MAX_CANDIDATES_PER_CARD * 4]
        if not fresh:
            continue
        payload.append({"card": c.key, "subject": c.subject, "limit": hub.limits.get(c.key) or hub.settings.PREINJECT_RETRIEVAL_TOPK,
                        "returned": len(c.asset_ids), "candidates": [
            {"asset_id": a.asset_id, "name": a.name, "type": a.asset_type, "grain": a.meta.grain, "fields": a.meta.fields[:10], "score": a.retrieval_score}
            for a in fresh]})
    if not payload:
        return res
    prompt = _render("roles/_annotate", {"role": role_text(spec), "contract": contract_json(hub), "cards": _dumps(payload),
                                         "max_limit": str(hub.settings.RECALL_MAX_LIMIT)})
    data = await hub.llm.json("annotate", prompt)
    if not data:
        return res
    by_key = {c.key: c for c in cards}
    shown = {p["card"]: {x["asset_id"] for x in p["candidates"]} for p in payload}
    verdicts: dict[str, dict[str, bool]] = {}
    evs: dict[str, list[Evidence]] = {}
    raw = data.get("judgements")
    for j in raw if isinstance(raw, list) else []:
        if not isinstance(j, dict) or not isinstance(j.get("relevant"), bool):
            continue
        key, aid = j.get("card"), j.get("asset_id")
        if key not in shown or aid not in shown[key] or aid in verdicts.get(key, {}):          # 只认通道真实返回、且展示给它的资产
            continue
        rel = j["relevant"]
        why = str(j.get("reason") or "").strip() or "no reason given"
        verdicts.setdefault(key, {})[aid] = rel
        evs.setdefault(key, []).append(Evidence(
            id=new_evidence_id(), asset_id=aid, kind=EvidenceKind.OTHER,
            statement=f"LLM relevance check ({by_key[key].subject}): {'relevant' if rel else 'irrelevant'} — {why}",
            value="relevant" if rel else "irrelevant", role=hub.role_of(agent_id), source="llm_filter", subject=by_key[key].subject,
            confidence=_clamp01(j.get("confidence"), 0.7 if rel else 0.2),
        ))
    for key, es in evs.items():
        await hub.board.attach_evidences(key, es)
    for key, ids in shown.items():
        hub.judged[key] |= ids                                                                  # 展示过的都算评估过，下一轮只评估新增的
    res.all_irrelevant = [by_key[k] for k, v in verdicts.items() if len(v) == len(shown[k]) and not any(v.values())]
    cov = data.get("coverage")
    for c in cov if isinstance(cov, list) else []:
        if not isinstance(c, dict) or c.get("card") not in shown or c.get("sufficient") is not False or by_key[c["card"]] in res.all_irrelevant:
            continue                                                                            # 全部不相关的走换角度重试，不是「多召回一点」
        key = c["card"]
        cur = hub.limits.get(key) or hub.settings.PREINJECT_RETRIEVAL_TOPK
        if cur < hub.settings.RECALL_MAX_LIMIT:
            res.more[key] = _more_request(hub, cur, c.get("more"))
    return res


# ── 自评是否接活 ─────────────────────────────────────────────────────────────
@dataclass
class SelfEval:
    accept: bool
    fit: float
    reason: str
    defer_to: str | None = None          # 婉拒时点名的、更合适的在场 agent


def peers_view(hub: "CollabHub", me: str, need: Need) -> list[dict[str, Any]]:
    """这次运行里在场的其他 agent（含请求方）：模型据此判断「有没有更合适的人」，而不是凭空猜。"""
    return [{
        "agent_id": a, "role": r.spec.name, "description": r.spec.description.split("：")[0], "serves": sorted(r.spec.serves),
        "targets": list(r.targets), "domains": sorted(r.spec.domains),
        "can_serve_this_need": need.kind in r.spec.serves and a != need.via and (not need.domains or set(need.domains) <= set(r.spec.domains)),
    } for a, r in hub.agents.items() if a != me]


def _valid_deferral(hub: "CollabHub", me: str, need: Need, requester: str, name: Any) -> str | None:
    """婉拒点名的 agent 必须：在场、不是自己、不是正在求助的请求方、且确实能服务这类需求（业务域覆盖）。否则婉拒无效。"""
    if not isinstance(name, str) or not name or name == me or name == requester:
        return None
    r = hub.agents.get(name)
    if r is None or need.kind not in r.spec.serves or (need.domains and not set(need.domains) <= set(r.spec.domains)):
        return None
    return name


async def self_evaluate(hub: "CollabHub", agent_id: str, spec: RoleSpec, need: Need, *, fit_rule: float, active: int, requester: str = "") -> SelfEval | None:
    """语义上是否胜任这个请求。不可用 / 输出非法 / 婉拒没有具体依据 → None（调用方回落规则的匹配度判断）。

    婉拒必须点名一个在场且能服务这类需求的 agent（`defer_to`）；没有这样的 agent 时「更适合别人」不成立，婉拒无效。"""
    if not hub.llm.enabled:
        return None
    status = {"role": spec.name, "agent_id": agent_id, "画像匹配度(规则)": round(fit_rule, 3), "正在响应的询问数": active, "同时响应上限": spec.max_concurrency}
    request = {"kind": need.kind, "subject": need.subject, "question": need.text, "domains": need.domains, "hint_assets": need.hint_assets,
               "via": need.via, "hop": need.hop, "requester": requester}
    peers = peers_view(hub, agent_id, need)
    prompt = _render("roles/_selfeval", {"role": role_text(spec), "status": _dumps(status), "request": _dumps(request),
                                         "peers": _dumps({"peers": peers}), "contract": contract_json(hub)})
    data = await hub.llm.json("selfeval", prompt)
    if not data or not isinstance(data.get("accept"), bool):
        return None
    reason = str(data.get("reason") or "").strip()
    fit = _clamp01(data.get("fit"), fit_rule)
    if data["accept"]:
        return SelfEval(accept=True, fit=fit, reason=reason or "LLM 自评可接")
    who = _valid_deferral(hub, agent_id, need, requester, data.get("defer_to"))
    if who is None or not reason:
        return None
    return SelfEval(accept=False, fit=fit, reason=f"{reason}（让位给 {who}）", defer_to=who)


# ── 撰写求助问题 ─────────────────────────────────────────────────────────────
async def compose_question(hub: "CollabHub", requester_role: str, need: Need) -> str | None:
    """把需求写成自足的问题。必须包含主题；不可用 / 输出非法 → None（沿用模板问句）。"""
    if not hub.llm.enabled:
        return None
    via = hub.board.cards.get(need.via)
    nd = {"kind": need.kind, "subject": need.subject, "domains": need.domains, "hint_assets": need.hint_assets,
          "triggered_by": {"card": via.key, "summary": via.summary} if via is not None else None, "template_question": need.text}
    prompt = _render("roles/_ask", {"requester": requester_role, "need": _dumps(nd), "contract": contract_json(hub), "subject": need.subject})
    data = await hub.llm.json("ask", prompt)
    q = data.get("question") if data else None
    q = q.strip() if isinstance(q, str) else ""
    if not (6 <= len(q) <= MAX_QUESTION_LEN) or need.subject.casefold() not in q.casefold():
        return None
    return q
