"""metric 兜底：提取不到 metric 时不追问用户，改用 intent.keywords 作为后续查找的指标词。

keywords 是比 metrics 更宽的召回词集（混有品牌词 / 维度词），因此先剔除已识别的维度与疑似维度取值（dim_terms），
结果以 `ValueSource.INFERRED` + 固定 rule_id 写入契约，保证来源可追溯。keywords 也为空时返回 []，
下游（③④⑤）会回落到整句 query。
"""
from __future__ import annotations

from typing import Any

from ...schemas.intent import Intent

METRICS_FROM_KEYWORDS_RULE_ID = "fallback.metrics.from_keywords"
METRICS_FROM_KEYWORDS_NOTE = "未提取到指标，改用 intent.keywords（已剔除维度词）作为后续查找词"


def metrics_from_keywords(intent: Intent | None, draft: dict[str, Any]) -> list[str]:
    if intent is None:
        return []
    dims = {str(d).strip().lower() for d in (draft.get("dimensions") or [])}
    dims |= {t.strip().lower() for t in intent.dim_terms}
    out: list[str] = []
    for kw in intent.keywords:
        k = kw.strip()
        if k and k.lower() not in dims and k not in out:
            out.append(k)
    return out
