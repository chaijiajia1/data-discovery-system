"""② 协调 · Coordination agent：用户提问 + ① 预注入知识 → intent（+ 契约槽位草稿）。

一次 LLM 调用同时产出 `intent`（domain / sub_domain / intent_type / keywords …）与 `slots`（metrics / grain / time_range …）：
    prompt（词表由 taxonomy 渲染，与校验同源）→ LLM JSON → 归一化 → validate_intent → Intent

LLM 是可选增强（铁律 3）：未配置 / 调用失败 / 输出不可解析 → 规则兜底（intent_rules + extractor.heuristic_extract），
并在 `Intent.source` / `fallback_reason` 中如实记录。
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ...json_repair import try_partial_parse
from ...errors import UpstreamError
from ...llm_client import LLMClient
from ...prompts import load_prompt
from ...schemas.intent import AssetRef, Intent
from ...schemas.knowledge import CONSUMER_COORDINATION, PreInjectionResult
from ...schemas.task import SLOT_NAMES
from .extractor import heuristic_extract
from .intent_rules import extract_asset_refs, infer_domain, infer_sub_domains, rule_intent
from .intent_validate import drop_invalid_sub_domains, normalize_intent_lists, validate_intent
from .taxonomy import DOMAIN_ENUM, INTENT_SUB_TYPES, INTENT_TYPES, KB_INTENTS, render_signals, render_taxonomy

logger = logging.getLogger(__name__)

_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")
_LIST_SLOTS = ("metrics", "dimensions", "filters")
_NO_DIM_TERMS_DOMAINS = {"resource_cost", "overseas"}      # 维值库仅覆盖三大业务域，这两域词触发维值召回必 miss


@dataclass
class CoordinationOutput:
    intent: Intent
    slots: dict[str, Any] = field(default_factory=dict)


# ── prompt ────────────────────────────────────────────────────────────────
def format_knowledge(pre: PreInjectionResult | None) -> str:
    """进 LLM prompt 的知识 = ① 分发给 ② 的视图（规则兜底路径仍读 injected 全量，见 extractor / defaults / intent_rules）。"""
    items = pre.view(CONSUMER_COORDINATION) if pre else []
    return "\n".join(f"- [{i.kind.value}] {i.title}: {i.content}" for i in items) or "（无）"


def _flatten(v: Any) -> str:
    return " ".join(str(x) for x in v) if isinstance(v, (list, tuple, set)) else str(v)


def format_clarifications(clarifications: dict[str, Any] | None) -> str:
    return "\n".join(f"- {k}: {_flatten(v)}" for k, v in (clarifications or {}).items()) or "（无）"


def recognition_text(query: str, clarifications: dict[str, Any] | None) -> str:
    """用于识别的文本：原问题 + 用户补充。"""
    return " ".join([query, *(_flatten(v) for v in (clarifications or {}).values())])


def build_prompt(query: str, pre: PreInjectionResult | None, session_id: str = "", clarifications: dict[str, Any] | None = None) -> str:
    """单遍占位符替换：被插入的内容（query / 知识）不会再被当作模板解析，杜绝占位符注入。"""
    values = {
        "taxonomy": render_taxonomy(), "signals": render_signals(), "session_id": session_id,
        "knowledge": format_knowledge(pre), "clarifications": format_clarifications(clarifications), "query": query,
    }
    return _PLACEHOLDER.sub(lambda m: values.get(m.group(1), m.group(0)), load_prompt("coordination"))


# ── LLM 输出解析 / 归一化 ───────────────────────────────────────────────────
def parse_llm_json(raw: str) -> dict[str, Any]:
    """容忍 markdown 围栏与前后多余文字；输出被截断时保留完整部分（见 json_repair）；无法得到 JSON 对象时抛 ValueError。"""
    s = (raw or "").strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        i, j = s.find("{"), s.rfind("}")
        if i < 0:
            raise ValueError("LLM 输出不含 JSON 对象") from None
        try:
            data = json.loads(s[i : j + 1]) if j > i else None
        except json.JSONDecodeError:
            data = None
        if data is None:
            data = try_partial_parse(s[i:])
            if data is None:
                raise ValueError("LLM 输出不是可解析的 JSON 对象") from None
    if not isinstance(data, dict):
        raise ValueError("LLM 输出不是 JSON 对象")
    return data


def _strs(v: Any) -> list[str]:
    out: list[str] = []
    for x in v if isinstance(v, list) else []:
        s = str(x).strip() if isinstance(x, (str, int, float)) else ""
        if s and s not in out:
            out.append(s)
    return out


def _opt_str(v: Any) -> str | None:
    return v.strip() if isinstance(v, str) and v.strip() else None


def build_intent(raw: dict[str, Any], *, query: str, session_id: str = "", source: str = "llm", text: str | None = None,
                 fallback_reason: str = "") -> Intent:
    """LLM 原始 intent → 合法 Intent：归一化 → 校验（warnings）→ 确定性修正（非法枚举剔除 / 空 sub_domain 规则补全）。"""
    text = text if text is not None else query
    d = dict(raw)
    d["query"], d["session_id"] = query, session_id            # 透传字段以输入为准，不信任 LLM 回写
    d["domain"] = d["domain"].strip() if isinstance(d.get("domain"), str) else None
    normalize_intent_lists(d)
    warnings = validate_intent(d, text)                         # 先校验（保留问题给下游看），再修正
    rule_ids: list[str] = []

    domain = d["domain"] if d["domain"] in DOMAIN_ENUM else None
    d["domain"] = domain
    if not isinstance(d.get("sub_domain"), list):
        d["sub_domain"] = []
    d["sub_domain"] = _strs(d["sub_domain"])
    subs = drop_invalid_sub_domains(d)["sub_domain"] if domain else []
    if domain and not subs:
        subs, rule_ids = infer_sub_domains(text, domain)        # domain 已识别时 sub_domain 不得为空
    if not domain:
        guess = infer_domain(text).domain
        if guess:
            warnings.append(f"domain 为空，但 query 含 `{guess}` 域强信号词，请确认是否漏识别")

    itype = d.get("intent_type")
    if itype not in INTENT_TYPES:
        if itype:
            warnings.append(f"intent_type `{itype}` 不在合法枚举 {list(INTENT_TYPES)}；按 other 处理")
        itype = "other"
    stype = d.get("intent_sub_type") if itype == "analysis" and d.get("intent_sub_type") in INTENT_SUB_TYPES else None

    if domain == "commercialization":
        kb = [k for k in _strs(d.get("kb_intents")) if k in KB_INTENTS] or ["write_sql"]
    else:
        kb = []

    keywords = _strs(d.get("keywords"))
    dim_terms = [] if domain in _NO_DIM_TERMS_DOMAINS else _strs(d.get("dim_terms"))
    keywords += [t for t in dim_terms if t not in keywords]    # dim_terms 是 keywords 的子集

    refs = [AssetRef(kind=str(r["kind"]), id=str(r["id"])) for r in (d.get("asset_refs") or [])
            if isinstance(r, dict) and r.get("kind") not in (None, "") and r.get("id") not in (None, "")]
    refs = refs or extract_asset_refs(text)

    return Intent(
        session_id=session_id, query=query, query_normalized=_opt_str(d.get("query_normalized")),
        intent_type=itype, intent_sub_type=stype, asset_refs=refs, domain=domain, sub_domain=subs, kb_intents=kb,
        keywords=keywords, dim_terms=dim_terms, time_hint=_opt_str(d.get("time_hint")), notes=_opt_str(d.get("notes")),
        source=source, rule_ids=rule_ids, fallback_reason=fallback_reason, warnings=warnings,
    )


def llm_slots(obj: Any) -> dict[str, Any] | None:
    """LLM 的 slots 对象 → 契约槽位草稿（只留 SLOT_NAMES；空值丢弃）。不是对象返回 None。"""
    if not isinstance(obj, dict):
        return None
    out: dict[str, Any] = {}
    for k in SLOT_NAMES:
        v = obj.get(k)
        if k in _LIST_SLOTS:
            v = [v] if isinstance(v, str) and v.strip() else v
            v = _strs(v) if isinstance(v, list) else []
        else:
            v = _opt_str(v)
        if v not in (None, [], ""):
            out[k] = v
    return out


def _intent_obj(data: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(data.get("intent"), dict):
        return data["intent"]
    return data if {"domain", "intent_type"} & set(data) else None       # 容忍 LLM 省掉 intent 外层


# ── agent ─────────────────────────────────────────────────────────────────
class CoordinationAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def run(self, query: str, pre: PreInjectionResult | None, *, session_id: str = "",
                  clarifications: dict[str, Any] | None = None) -> CoordinationOutput:
        text = recognition_text(query, clarifications)
        try:
            raw = await self.llm.complete(prompt=build_prompt(query, pre, session_id, clarifications), json_mode=True)
            data = parse_llm_json(raw)
            intent_raw = _intent_obj(data)
            if intent_raw is None:
                raise ValueError("LLM 输出缺少 intent 对象")
            intent = build_intent(intent_raw, query=query, session_id=session_id, source="llm", text=text)
            slots = llm_slots(data.get("slots"))
            return CoordinationOutput(intent, slots if slots is not None else heuristic_extract(query, pre))
        except (UpstreamError, ValueError) as e:
            reason = str(e)
        logger.info("coordination LLM fallback to rules: %s", reason)
        return CoordinationOutput(
            rule_intent(query, pre, session_id=session_id, text=text, fallback_reason=reason), heuristic_extract(query, pre),
        )
