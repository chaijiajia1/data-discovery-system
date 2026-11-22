"""路 1：context-query 检索（POST /v1/context/retrieve）。"""
from __future__ import annotations

from ...schemas.knowledge import RetrievalPack
from ...services.protocols import KnowledgeHybridRetrieval


class RetrievalKnowledgeSource:
    def __init__(self, kh: KnowledgeHybridRetrieval, max_tokens: int, inconsistent_discount: float):
        self.kh, self.max_tokens, self.discount = kh, max_tokens, inconsistent_discount

    async def fetch(self, query: str, user_id: str | None, cookie: str | None) -> RetrievalPack:
        pack = await self.kh.retrieve(query, user_id=user_id, cookie=cookie, max_tokens=self.max_tokens)
        for it in pack.items:
            it.routes = ['retrieval']
            # 一致性存疑 → 降低独立贡献 u_k
            if it.consistency == "possible_inconsistent":
                it.relevance = round(it.relevance * self.discount, 4)
        return pack
