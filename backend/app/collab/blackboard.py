"""共享黑板：并行角色之间的追加写协作区。

- 证据卡（EvidenceCard）：一次探查的可复用产出，按需求 key 索引。复用优先——后到的角色先 `lookup`，有卡就不再询问或广播；
- 认领（claim）+ 等待：同一需求只有一个角色真正去探查，其余角色 `wait_card` 等它的证据卡（而不是各自重复探查）；
  探查失败 `release`，等待者立刻解除阻塞并改为自己探查；
- 统一候选池（CandidatePool）：BI 通道与知识通道（向量 + 图谱）归一化后的资产按 asset_id 合并在这里；
- 追加写日志（BoardEntry）：认领 / 发现 / 广播 / 婉拒，供回放与展示。
全部写操作经 asyncio.Lock，保证 seq 单调、认领原子。
"""
from __future__ import annotations

import asyncio
from typing import Any

from ..schemas.collab import EvidenceCard, Need, new_card_id
from ..schemas.recall import BoardEntry, BoardEntryKind, CandidateAsset, Evidence
from ..services.normalize import CandidatePool


class Blackboard:
    def __init__(self) -> None:
        self._entries: list[BoardEntry] = []
        self._claims: dict[str, str] = {}
        self._events: dict[str, asyncio.Event] = {}
        self._evidences: dict[str, Evidence] = {}
        self._seq = 0
        self._lock = asyncio.Lock()
        self.cards: dict[str, EvidenceCard] = {}          # need key → 最新证据卡
        self.pool = CandidatePool()

    # ── 追加写日志 ──
    async def _append(self, author: str, kind: BoardEntryKind, topic: str, payload: dict[str, Any], confidence: float) -> BoardEntry:
        e = BoardEntry(seq=len(self._entries) + 1, author=author, kind=kind, topic=topic, payload=payload, confidence=confidence)
        self._entries.append(e)
        return e

    async def post(self, author: str, kind: BoardEntryKind, topic: str, payload: dict[str, Any] | None = None, confidence: float = 0.5) -> BoardEntry:
        async with self._lock:
            return await self._append(author, kind, topic, payload or {}, confidence)

    def entries(self, *, kind: BoardEntryKind | None = None, topic: str | None = None) -> list[BoardEntry]:
        return [e for e in self._entries if (kind is None or e.kind == kind) and (topic is None or e.topic == topic)]

    def snapshot(self) -> list[BoardEntry]:
        return list(self._entries)

    # ── 认领 ──
    async def try_claim(self, who: str, key: str) -> bool:
        async with self._lock:
            if key in self._claims and self._claims[key] != who:
                return False
            fresh = key not in self._claims
            self._claims[key] = who
            ev = self._events.get(key)
            if ev is not None and key not in self.cards:
                ev.clear()                                  # 上一次探查失败被 release 过：重新进入「等待产出」状态
            if fresh:
                await self._append(who, BoardEntryKind.CLAIM, key, {}, 1.0)
            return True

    def claimed_by(self, key: str) -> str | None:
        return self._claims.get(key)

    def release(self, key: str) -> None:
        """认领者探查失败：放弃认领并唤醒等待者（他们拿不到证据卡，会改为自己探查）。"""
        self._claims.pop(key, None)
        self._event(key).set()

    # ── 证据卡 ──
    def _event(self, key: str) -> asyncio.Event:
        return self._events.setdefault(key, asyncio.Event())

    async def publish(self, card: EvidenceCard, *, assets: list[CandidateAsset] | None = None, evidences: list[Evidence] | None = None) -> EvidenceCard:
        async with self._lock:
            self._seq += 1
            card = card.model_copy(update={"seq": self._seq, "id": card.id or new_card_id()}, deep=True)
            self.cards[card.key] = card
            if assets:
                self.pool.add(assets)
            for e in evidences or []:
                self._evidences[e.id] = e
            await self._append(card.author, BoardEntryKind.FINDING, card.key, {"card_id": card.id, "assets": len(card.asset_ids)}, card.confidence)
        self._event(card.key).set()
        return card

    async def attach_evidences(self, key: str, evidences: list[Evidence]) -> None:
        """给已发布的证据卡追加证据（如 LLM 对结果的相关性评估）。只增不删：不改变卡里的资产。"""
        async with self._lock:
            card = self.cards.get(key)
            for e in evidences:
                self._evidences[e.id] = e
                if card is not None and e.id not in card.evidence_ids:
                    card.evidence_ids.append(e.id)

    def lookup(self, need: Need, *, min_conf: float = 0.3, require_content: bool = True) -> EvidenceCard | None:
        """复用优先：找可复用的证据卡。置信度够、业务域相容；`require_content` 时空卡不算（空卡只能用来避免重复探查）。"""
        c = self.cards.get(need.key)
        if c is None or c.confidence < min_conf and not (c.empty and not require_content):
            return None
        if require_content and c.empty:
            return None
        if c.domains and need.domains and not set(c.domains) & set(need.domains):
            return None
        return c

    def get_card(self, card_id: str | None) -> EvidenceCard | None:
        return next((c for c in self.cards.values() if c.id == card_id), None) if card_id else None

    def mark_reused(self, card: EvidenceCard) -> None:
        cur = self.cards.get(card.key)
        if cur is not None:
            cur.reuse_count += 1

    async def wait_card(self, key: str, timeout: float) -> EvidenceCard | None:
        """等待别人认领的探查产出证据卡。超时或认领者失败返回 None。"""
        if key in self.cards:
            return self.cards[key]
        try:
            await asyncio.wait_for(self._event(key).wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        return self.cards.get(key)

    def assets_of(self, card: EvidenceCard) -> list[CandidateAsset]:
        return [a for i in card.asset_ids if (a := self.pool.get(i)) is not None]

    def evidences_of(self, card: EvidenceCard) -> list[Evidence]:
        return [e for i in card.evidence_ids if (e := self._evidences.get(i)) is not None]
