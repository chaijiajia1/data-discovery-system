"""默认补全规则：缺失但对结果影响有限的条件按默认值补全，并记录规则 id（来源可追溯）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ...schemas.knowledge import KnowledgeKind, PreInjectionResult


@dataclass(frozen=True)
class DefaultRule:
    id: str
    slot: str
    value: Callable[[PreInjectionResult | None, dict[str, Any]], Any]
    note: str


def _caliber_from_knowledge(pre, draft):
    """口径：优先用预注入的指标定义；否则 'standard'（并在 note 中标注）。"""
    metrics = {m.upper() for m in (draft.get("metrics") or [])}
    for it in (pre.injected if pre else []):
        if it.kind == KnowledgeKind.METRIC_DEF and it.title.upper() in metrics:
            return it.content
    return None


DEFAULT_RULES: dict[str, DefaultRule] = {
    "time_range": DefaultRule("default.time_range.last_7_days", "time_range", lambda p, d: "last_7_days", "未指定时间范围，默认最近 7 天"),
    "grain": DefaultRule("default.grain.day", "grain", lambda p, d: "day", "未指定粒度，默认按天"),
    "dimensions": DefaultRule("default.dimensions.none", "dimensions", lambda p, d: [], "未指定维度，默认不拆分"),
    "filters": DefaultRule("default.filters.none", "filters", lambda p, d: [], "未指定过滤条件，默认全量"),
    "caliber": DefaultRule("default.caliber.knowledge_or_standard", "caliber", _caliber_from_knowledge, "口径取预注入指标定义，缺省为标准口径"),
}
