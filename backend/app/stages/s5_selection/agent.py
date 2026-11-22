"""⑤ 选择 Agent：LLM 复核 + 写结构化理由；规则引擎是硬护栏与兜底（铁律 3）。

流程（每个口径一次调用，口径之间并行）：
    规则引擎先对全部候选做评估（rules.evaluate_group）→ 渲染 prompt（候选 + 规则预判 hints）→ LLM 输出三档 JSON
    → 校验：asset_id 必须来自候选、primary 至多 1 个、三档完整且唯一、query_fields 的字段必须存在、命中必须逐字复述候选原文、
      盲区只能引用后端 signals ——不合规的部分丢弃并告警
    → 护栏：LLM 选出带「硬不合规」标记的资产做主资产（无权限 / 粒度粗于要求 / 无目标指标字段 / 主键粒度不符 / scope 不足 / 维值无法过滤）
      → 拒收，整个口径回退规则结论
    LLM 未配置 / 调用失败 / 输出不可解析 → 该口径回退规则，并记录 fallback_reason。
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from ...errors import UpstreamError
from ...llm_client import LLMClient
from ...prompts import load_prompt
from ...schemas.knowledge import KnowledgeItem
from ...schemas.recall import CandidateAsset
from ...schemas.selection import AssetDecision, FieldRef, Hit, Miss, Reason
from ..s2_coordination.agent import parse_llm_json
from .rules import (
    Eval, GroupResult, SelectionInput, assemble, build_facts, build_query_fields, dim_term_support, evaluate_group, metric_groups,
    render_field, select_group, upstream_notes, _pick, _score,
)

logger = logging.getLogger(__name__)
_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")
MAX_FIELDS = 25
HOW = {"字面相同", "同义", "部分", "推断"}


# ── prompt ────────────────────────────────────────────────────────────────
def render_candidate(e: Eval, upstream: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    c, m = e.c, e.f.c.meta
    d: dict[str, Any] = {"id": c.asset_id, "type": c.asset_type, "name": c.name, "score": round(c.retrieval_score, 3), "channels": c.channels,
                         "domains": m.domains, "description": m.description, "tags": m.tags, "main_table": m.main_table,
                         "stats": {"heat_30d": m.heat_30d, "ref_count": m.ref_count}, "perm": {"granted": e.f.denied is None},
                         "is_onedash_asset": m.is_onedash, "grain": m.grain, "partition_field": m.partition_field,
                         "signals": e.f.signals, "hints": rule_hint(e), "upstream_evidence": upstream or []}
    if e.f.chart:
        d |= {"is_strict": m.is_strict, "metric_ids": m.metric_ids, "supported_dims": m.supported_dims}
    else:
        d["fields"] = [render_field(c, f) for f in m.fields[:MAX_FIELDS]]
    if m.cube_filters or e.f.cube:
        d["cube_rules"] = {"is_cube": True, "summary_filters": m.cube_filters}
    if m.sql_rules:
        d["sql_rules"] = m.sql_rules
    if m.knowledge_docs:
        d["knowledge_docs"] = m.knowledge_docs
    return d


def rule_hint(e: Eval) -> dict[str, Any]:
    return {"rule_verdict": e.verdict, "hard": e.hard, "coverage": e.cov, "note": e.conclusion or None}


def _knowledge_text(items: list[KnowledgeItem]) -> str:
    return "\n".join(f"- id={i.id} [{i.kind.value}] {i.title}: {i.content[:200]}" for i in items) or "（无）"


def build_prompt(inp: SelectionInput, label: str, evals: dict[str, Eval]) -> str:
    ct, it = inp.contract, inp.intent
    intent = {"query": ct.query, "intent_type": it.intent_type if it else None, "intent_sub_type": it.intent_sub_type if it else None,
              "domain": it.domain if it else None, "metrics": ct.metrics.value, "dimensions": ct.dimensions.value, "dim_terms": it.dim_terms if it else [],
              "grain": ct.grain.value, "time_range": ct.time_range.value, "time_hint": it.time_hint if it else None}
    refs = {}
    for r in inp.dim_refs:
        refs.setdefault(r.term, []).append({"field": r.field, "asset_id": r.asset_id})
    values = {
        "caliber": label, "intent": json.dumps(intent, ensure_ascii=False, indent=1),
        "intent_warnings": "\n".join(f"- {w}" for w in (it.warnings if it else [])) or "（无）", "knowledge_context": _knowledge_text(inp.knowledge),
        "dim_value_refs": json.dumps(refs, ensure_ascii=False) if refs else "（无）",
        "candidates": json.dumps([render_candidate(e, upstream_notes(inp, e.c.asset_id, label)) for e in evals.values()], ensure_ascii=False, indent=1),
    }
    return _PLACEHOLDER.sub(lambda m: values.get(m.group(1), m.group(0)), load_prompt("selection"))      # 单遍替换：候选内容不会被当成模板


# ── 解析 / 校验 ───────────────────────────────────────────────────────────
class Rejected(ValueError):
    """LLM 输出整体不可用（需回退规则）。"""


def _parse_reason(raw: Any, e: Eval, warnings: list[str], strict_blind: bool = True) -> Reason:
    if isinstance(raw, str):
        return Reason(conclusion=raw.strip(), blind_spots=[])
    raw = raw if isinstance(raw, dict) else {}
    c = e.c
    allowed = {render_field(c, f) for f in c.meta.fields} | {*c.meta.metric_ids, *c.meta.supported_dims, c.name, c.meta.description or "", *c.meta.tags}
    hits: list[Hit] = []
    for h in raw.get("命中") or []:
        try:
            hit = Hit(term=str(h["业务词"]), type=h["类型"], matched=str(h["匹配到"]), location=str(h.get("位置", "")), how=h.get("匹配方式", "推断"))
        except Exception:  # noqa: BLE001
            continue
        if hit.matched not in allowed and hit.matched != "[]":
            warnings.append(f"{c.asset_id}: the 'matched' text of hit '{hit.term}' is not candidate text; dropped ({hit.matched})")
            continue
        hits.append(hit)
    misses = []
    for m in raw.get("未命中") or []:
        try:
            misses.append(Miss(term=str(m["业务词"]), type=m["类型"], note=str(m.get("说明", ""))))
        except Exception:  # noqa: BLE001
            continue
    blind = [b for b in (raw.get("盲区") or []) if isinstance(b, str)]
    kept = [b for b in blind if b in e.f.signals]
    if len(kept) != len(blind):
        warnings.append(f"{c.asset_id}: blind spots cited something outside the backend signals; dropped")
    return Reason(conclusion=str(raw.get("结论") or "").strip(), hits=hits, misses=misses,
                  rules=[str(x) for x in raw.get("采用规则") or []], downweights=[str(x) for x in raw.get("降权信号") or []], blind_spots=kept)


def _parse_fields(raw: Any, c: CandidateAsset, warnings: list[str]) -> list[FieldRef]:
    out: list[FieldRef] = []
    valid = set(c.meta.fields)
    for f in raw or []:
        if not isinstance(f, dict) or f.get("role") not in ("metric", "dimension"):
            continue
        name = str(f.get("field_name") or "")
        if valid and name not in valid:
            warnings.append(f"{c.asset_id}: query_fields contained field {name} which the candidate does not have; dropped")
            continue
        dv = f.get("dim_values")
        out.append(FieldRef(field_name=name, field_alias=str(f.get("field_alias") or c.meta.field_aliases.get(name, "")), role=f["role"],
                            dim_values=[str(x) for x in dv] if isinstance(dv, list) else None))
    return out


def validate_group(raw: dict[str, Any], inp: SelectionInput, label: str, evals: dict[str, Eval], rule: GroupResult) -> GroupResult:
    """把 LLM 的三档 JSON 变成 GroupResult；整体不可用抛 Rejected。"""
    warnings: list[str] = []
    ids = set(evals)
    tiers: dict[str, tuple[str, Any]] = {}
    for tier in ("primary", "alternates", "discarded"):
        items = raw.get(tier)
        if items is None:
            continue
        if not isinstance(items, list):
            raise Rejected(f"{tier} is not an array")
        for it in items:
            aid = it.get("asset_id") if isinstance(it, dict) else None
            if aid not in ids:
                warnings.append(f"{tier} contained asset_id {aid!r} which is not a candidate; dropped")
                continue
            if aid in tiers:
                warnings.append(f"{aid} appeared in several tiers; kept the earliest tier")
                continue
            tiers[aid] = (tier, it)
    prim = [a for a, (t, _) in tiers.items() if t == "primary"]
    if len(prim) > 1:
        warnings.append(f"{len(prim)} primaries were returned; kept the first, demoted the rest to alternates")
        for a in prim[1:]:
            tiers[a] = ("alternates", tiers[a][1])
    primary_id = prim[0] if prim else None
    if primary_id is not None:
        e = evals[primary_id]
        if e.hard:
            raise Rejected(f"The LLM chose {primary_id}, which the rules judged non-compliant ({e.hard}: {e.conclusion})")
        if e.verdict == "alternate" and e.f.chart:
            raise Rejected(f"The LLM chose chart {primary_id}, which does not meet the strict chart conditions")
    elif rule.primary is not None and any(e.verdict == "eligible" for e in evals.values()):
        warnings.append("The LLM selected no primary although usable candidates exist")
    g = GroupResult(caliber=label, primary=None, mode="llm", keywords=rule.keywords, evals=evals)
    for aid, e in evals.items():                         # 缺失的候选：用规则的判定补齐，保证三档完整且唯一
        tier, item = tiers.get(aid, (None, None))
        if tier is None:
            if aid == (rule.primary.asset_id if rule.primary else None):
                tier = "alternates"                      # 规则的主资产没被 LLM 选中：作备选，不自动恢复为主资产
                warnings.append(f"{aid} was missing from the LLM output; filled in as an alternate")
            else:
                d = rule.alternates.get(aid) or rule.discarded.get(aid)
                (g.alternates if aid in rule.alternates else g.discarded)[aid] = d.model_copy(deep=True) if d else d  # type: ignore[assignment]
                warnings.append(f"{aid} was missing from the LLM output; filled in with the rule verdict")
                continue
        reason = _parse_reason(item.get("reason") if item else None, e, warnings) if item else Reason()
        if not reason.conclusion:
            reason.conclusion = e.conclusion or "(the LLM gave no reason)"
        if tier == "primary":
            fields = _parse_fields(item.get("query_fields"), e.c, warnings) or build_query_fields(inp, e, label)
            if not reason.rules:
                reason.rules = list(e.rules)
            g.primary = AssetDecision(asset_id=aid, role="primary", reason=reason, covers=[label], query_fields=fields,
                                      rule_ids=list(dict.fromkeys(e.rule_ids)), score=_score(e))
        else:
            role = "alternate" if tier == "alternates" else "discarded"
            d = AssetDecision(asset_id=aid, role=role, reason=reason, covers=[label] if role == "alternate" else [],   # type: ignore[arg-type]
                              needs_permission=e.needs_permission, rule_ids=list(dict.fromkeys(e.rule_ids)), score=_score(e))
            if role == "alternate" and e.hard == "permission":
                d.needs_permission = True
            (g.alternates if role == "alternate" else g.discarded)[aid] = d
    # 规则判定为「丢弃」的硬不合规候选，LLM 不能把它抬到备选以上（权限降级本就在备选里）
    for aid, d in list(g.alternates.items()):
        if evals[aid].verdict == "discarded" and evals[aid].hard:
            g.discarded[aid] = d.model_copy(update={"role": "discarded", "covers": []})
            del g.alternates[aid]
            warnings.append(f"{aid} was judged {evals[aid].hard} by the rules; the LLM's alternate verdict was changed to discarded")
    refs = raw.get("knowledge_refs")
    known = {k.id for k in inp.knowledge}
    g.knowledge_refs = [k for k in (refs if isinstance(refs, list) else []) if k in known]
    if isinstance(refs, list) and len(g.knowledge_refs) != len(refs):
        warnings.append("knowledge_refs contained ids not in knowledge_context; dropped")
    g.warnings = warnings
    return g


# ── agent ────────────────────────────────────────────────────────────────
class SelectionAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def _group(self, inp: SelectionInput, facts, metric, support) -> GroupResult:
        rule = select_group(inp, facts, metric, support)            # 规则结论：兜底 + 护栏 + 补齐
        label = rule.caliber
        evals = evaluate_group(inp, facts, metric, support)         # 重新评估一份干净的（select_group 里会就地改动 verdict）
        _pick(evals, inp.contract)                                  # 同样触发 cube / scope 让位，保持与规则一致的预判
        try:
            raw = parse_llm_json(await self.llm.complete(prompt=build_prompt(inp, label, evals), json_mode=True))
            return validate_group(raw, inp, label, evals, rule)
        except (UpstreamError, ValueError) as e:
            reason = str(e)
        logger.info("selection LLM fallback to rules (%s): %s", label, reason)
        rule.fallback_reason = reason
        return rule

    async def run(self, inp: SelectionInput):
        facts, support = build_facts(inp), dim_term_support(inp)
        groups = list(await asyncio.gather(*(self._group(inp, facts, m, support) for m in metric_groups(inp.contract))))
        if not inp.contract.metrics.value:
            groups[0].caliber = inp.contract.query
        modes = {g.mode for g in groups}
        warns = [f"[{g.caliber}] {w}" for g in groups for w in g.warnings]
        warns += [f"[{g.caliber}] LLM fell back to rules: {g.fallback_reason}" for g in groups if g.fallback_reason]
        return assemble(inp, groups, mode="llm" if modes == {"llm"} else ("rule" if modes == {"rule"} else "mixed"), warnings=warns)
