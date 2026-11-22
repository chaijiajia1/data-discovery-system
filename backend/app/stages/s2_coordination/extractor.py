"""自然语言 → 契约槽位草稿（规则版）。

LLM 版由 `agent.CoordinationAgent` 统一产出（intent + slots 一次调用）；本模块是它的规则兜底，
也被 intent_rules 复用来抽取指标 / 维度。
注意：只"抽取用户明确表达的条件"，缺失条件的补全由 defaults.py 负责。
"""
from __future__ import annotations

import re
from typing import Any

from ...schemas.knowledge import KnowledgeKind, PreInjectionResult

_METRIC_TOKEN = re.compile(r"(?<![A-Za-z0-9_])[A-Z][A-Z0-9_]+(?![A-Za-z0-9_])")
_TIME_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?:最近|近|过去)\s*(\d+)\s*天|last\s*(\d+)\s*days", re.I), "last_{n}_days"),
    (re.compile(r"昨天|昨日|yesterday", re.I), "yesterday"),
    (re.compile(r"上周|last week", re.I), "last_week"),
    (re.compile(r"上月|last month", re.I), "last_month"),
]
_GRAIN_PATTERNS = [(re.compile(r"每小时|分小时|hourly"), "hour"), (re.compile(r"每天|按天|分天|daily"), "day"),
                   (re.compile(r"每周|按周|weekly"), "week"), (re.compile(r"每月|按月|monthly"), "month")]
_DIM_PATTERNS = [re.compile(r"(?:各|按|分)([\u4e00-\u9fff]{1,6}?)(?:的|和|与|、|，|,|\s|$)"), re.compile(r"\bby\s+(\w+)", re.I)]
_DIM_STOP = {"天", "周", "月", "小时"}


def heuristic_extract(query: str, pre: PreInjectionResult | None) -> dict[str, Any]:
    known_metrics = {i.title.upper() for i in (pre.injected if pre else []) if i.kind == KnowledgeKind.METRIC_DEF}
    metrics = list(dict.fromkeys(
        m for m in _METRIC_TOKEN.findall(query) if m in known_metrics or len(m) <= 6
    ))
    metrics += [m for m in sorted(known_metrics) if m in query.upper() and m not in metrics]

    time_range = None
    for pat, tpl in _TIME_PATTERNS:
        m = pat.search(query)
        if m:
            n = next((g for g in m.groups() if g), None)
            time_range = tpl.format(n=n) if n else tpl
            break
    grain = next((g for p, g in _GRAIN_PATTERNS if p.search(query)), None)
    dims: list[str] = []
    for p in _DIM_PATTERNS:
        dims += [d for d in p.findall(query) if d not in _DIM_STOP]
    return {"metrics": metrics, "time_range": time_range, "grain": grain, "dimensions": list(dict.fromkeys(dims))}
