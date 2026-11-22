"""路 2：匹配历史 query。

每条命中的历史 query 展开为两类知识块，相关度 = query 相似度 × 历史可靠度：
- 「当时用到的知识」快照（与路 1 同 id 时由 merge 视为双路命中）；
- 一条 PAIR 摘要块：该问题最终选了哪些资产。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ...schemas.knowledge import KnowledgeItem, KnowledgeKind, KnowledgeOrigin, PairMatch, VerifiedPair
from ...stores import VerifiedPairStore
from ...text_utils import estimate_tokens


def reliability(pair: VerifiedPair) -> float:
    """历史成功率（新 pair 为 1）；被预注入后多次失败的 pair 会被自然降权。"""
    return (pair.success_count + 1) / (pair.hit_count + 1)


def pair_to_items(pair: VerifiedPair, similarity: float) -> list[KnowledgeItem]:
    rel = round(min(1.0, similarity * reliability(pair)), 4)
    digest = "；".join(f"{k}={v}" for k, v in pair.contract_digest.items())
    summary = f"历史相似问题「{pair.query}」最终选用资产 {', '.join(pair.asset_ids)}" + (f"（{digest}）" if digest else "")
    items = [KnowledgeItem(
        id=f"pair:{pair.pair_id}", kind=KnowledgeKind.PAIR, title=pair.query, content=summary, origin=KnowledgeOrigin.PAIR,
        routes=["pair"], asset_ids=list(pair.asset_ids), terms=list(pair.terms), relevance=rel,
        token_cost=estimate_tokens(summary), source_pair_id=pair.pair_id,
    )]
    for k in pair.knowledge:
        items.append(KnowledgeItem(
            id=k.id, kind=k.kind, title=k.title, content=k.content, origin=KnowledgeOrigin.PAIR, routes=["pair"],
            asset_ids=list(pair.asset_ids), relevance=rel, token_cost=estimate_tokens(f"{k.title}\n{k.content}"),
            source_pair_id=pair.pair_id, source_uri=k.source_uri,
        ))
    return items


@dataclass
class PairRoute:
    items: list[KnowledgeItem] = field(default_factory=list)
    matches: list[PairMatch] = field(default_factory=list)


class PairKnowledgeSource:
    def __init__(self, store: VerifiedPairStore, topk: int, min_sim: float):
        self.store, self.topk, self.min_sim = store, topk, min_sim

    async def fetch(self, terms: list[str]) -> PairRoute:
        route = PairRoute()
        for pair, sim in self.store.search(terms, self.topk):
            if sim < self.min_sim:
                continue
            items = pair_to_items(pair, sim)
            route.items.extend(items)
            route.matches.append(PairMatch(
                pair_id=pair.pair_id, query=pair.query, similarity=round(sim, 4), asset_ids=list(pair.asset_ids),
                knowledge_titles=[k.title for k in pair.knowledge], token_cost=sum(i.token_cost for i in items),
                reliability=round(reliability(pair), 4),
            ))
        return route
