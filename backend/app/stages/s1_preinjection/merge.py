"""双路合并：按知识块 id 去重；两路都命中同一块是强信号，给 u_k 加成。"""
from __future__ import annotations

from ...schemas.knowledge import KnowledgeItem

DUAL_ROUTE_BONUS = 0.1


def merge_routes(*groups: list[KnowledgeItem]) -> list[KnowledgeItem]:
    by_id: dict[str, KnowledgeItem] = {}
    for group in groups:
        for it in group:
            cur = by_id.get(it.id)
            if cur is None:
                by_id[it.id] = it.model_copy(deep=True)
                continue
            keep, other = (cur, it) if cur.relevance >= it.relevance else (it.model_copy(deep=True), cur)
            keep.routes = sorted(set(cur.routes) | set(it.routes))
            keep.asset_ids = sorted(set(keep.asset_ids) | set(other.asset_ids))
            keep.source_pair_id = keep.source_pair_id or other.source_pair_id
            if len(keep.routes) > 1:
                keep.relevance = round(min(1.0, keep.relevance + DUAL_ROUTE_BONUS), 4)
            by_id[it.id] = keep
    return list(by_id.values())
