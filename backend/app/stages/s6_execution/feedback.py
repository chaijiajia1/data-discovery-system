"""⑥ → ①：通过核验的 查询—资产/知识 pair 回流到 pair store。"""
from __future__ import annotations

import hashlib

from ...schemas.execution import VerificationReport
from ...schemas.knowledge import KnowledgeKind, KnowledgeSnapshot, PreInjectionResult, VerifiedPair
from ...schemas.selection import SelectionResult
from ...schemas.task import TaskContract
from ...stores import VerifiedPairStore
from ...text_utils import extract_terms


def build_pair(query: str, contract: TaskContract, selection: SelectionResult, pre: PreInjectionResult | None,
               verification: VerificationReport, used_asset_id: str) -> VerifiedPair:
    assets = [used_asset_id] + [d.asset_id for d in (*selection.primary, *selection.alternates) if d.asset_id != used_asset_id]
    pid = hashlib.sha1(f"{' '.join(sorted(extract_terms(query)))}|{used_asset_id}".encode()).hexdigest()[:12]
    # 用到的知识 = 本次注入的非 PAIR 摘要块（路 1 / 路 2 的知识都沿用），保存快照
    snaps = [
        KnowledgeSnapshot(id=i.id, kind=i.kind, title=i.title, content=i.content, source_uri=i.source_uri)
        for i in (pre.injected if pre else []) if i.kind != KnowledgeKind.PAIR
    ]
    digest = {k: str(v.value) for k, v in contract.slots().items() if k in ("caliber", "grain", "time_range", "metrics") and v.value}
    return VerifiedPair(
        pair_id=pid, query=query, terms=extract_terms(query), asset_ids=assets, knowledge=snaps, knowledge_ids=[k.id for k in snaps],
        contract_digest=digest, verification_checks=[c.name for c in verification.checks if c.passed],
    )


def write_back(store: VerifiedPairStore, pair: VerifiedPair | None, pre: PreInjectionResult | None, success: bool) -> bool:
    """成功：写入新 pair；无论成败：回写被预注入 pair 的命中/成功统计。
    只统计至少被一个消费方（视图）看到的 pair：注入了却没人用到的 pair 没有影响本次结果，不该记命中。"""
    consumed = pre.consumed_ids() if pre else set()
    used_pairs = [i.source_pair_id for i in (pre.injected if pre else []) if i.source_pair_id and i.id in consumed]
    if used_pairs:
        store.record_outcome(used_pairs, success)
    if success and pair is not None:
        store.add(pair)
        return True
    return False
