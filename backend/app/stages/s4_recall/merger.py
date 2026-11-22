"""证据合并去重；冲突显式保留。"""
from __future__ import annotations

from collections import defaultdict

from ...collab.blackboard import Blackboard
from ...services.normalize import CandidatePool, fuse_score, merge_candidates

from ...schemas.knowledge import KnowledgeItem, KnowledgeKind
from ...schemas.recall import CandidateAsset, Conflict, Evidence, EvidenceKind, RecallResult, RoleReport

# 这些证据类型的 value 取值不同即视为冲突（如两个角色对同一资产粒度给出不同结论）
_CONFLICT_KINDS = (EvidenceKind.GRAIN, EvidenceKind.CALIBER, EvidenceKind.PARTITION)


def _norm(v) -> str:
    return str(v).strip().lower()


def _knowledge_evidence(knowledge: list[KnowledgeItem]) -> list[Evidence]:
    """预注入的指标定义直接作为口径证据，避免为已知口径再起一个角色去探查。"""
    return [
        Evidence(id=f"ev_kn_{k.id}_{aid}", asset_id=aid, kind=EvidenceKind.CALIBER, statement=k.content, value=k.content,
                 role="preinjection", source="preinjected_knowledge", confidence=max(0.3, k.relevance))
        for k in knowledge if k.kind == KnowledgeKind.METRIC_DEF for aid in k.asset_ids
    ]


def merge(reports: list[RoleReport], knowledge: list[KnowledgeItem] | None = None, board: Blackboard | None = None, role_of=lambda a: a) -> RecallResult:
    raw = 0
    assets: dict[str, CandidateAsset] = {}
    for rep in reports:
        for c in rep.candidates:
            raw += 1
            c = c.model_copy(update={"found_by": c.found_by or [rep.role], "evidence_ids": []}, deep=True)
            assets[c.asset_id] = merge_candidates(assets[c.asset_id], c) if c.asset_id in assets else c
    if board is not None:       # 转交链路里产生的证据卡不属于任何节点自己的报告：把黑板上所有卡的资产并入候选池（found_by = 产出该卡的角色）
        for card in sorted(board.cards.values(), key=lambda c: c.seq):
            for a in board.assets_of(card):
                x = a.model_copy(update={"found_by": [role_of(card.author)], "evidence_ids": []}, deep=True)
                assets[a.asset_id] = merge_candidates(assets[a.asset_id], x) if a.asset_id in assets else x
    for c in assets.values():                                  # 多通道命中是强信号：融合分加成
        c.retrieval_score = fuse_score(c)

    # 证据去重：同资产 + 同类型 + 同陈述
    evs: dict[tuple[str, str, str], Evidence] = {}
    all_evs = [e for rep in reports for e in rep.evidences] + (list(board._evidences.values()) if board is not None else []) + _knowledge_evidence(knowledge or [])
    for e in all_evs:
        key = (e.asset_id, e.kind.value, _norm(e.statement))
        if key not in evs or e.confidence > evs[key].confidence:
            evs[key] = e
    evidences = list(evs.values())

    for e in evidences:
        if e.asset_id in assets:
            assets[e.asset_id].evidence_ids.append(e.id)

    # 冲突检测
    groups: dict[tuple[str, EvidenceKind], list[Evidence]] = defaultdict(list)
    for e in evidences:
        if e.kind in _CONFLICT_KINDS and e.value is not None:
            groups[(e.asset_id, e.kind)].append(e)
    conflicts = [
        Conflict(asset_id=aid, topic=kind, evidence_ids=[e.id for e in es], values=[e.value for e in es],
                 note="不同角色对同一资产给出不一致结论，未自动择一，交由 ⑤ 决策")
        for (aid, kind), es in groups.items() if len({_norm(e.value) for e in es}) > 1
    ]
    candidates = sorted(assets.values(), key=lambda c: c.retrieval_score, reverse=True)
    pool = CandidatePool()
    pool.add(candidates)
    cards = sorted(board.cards.values(), key=lambda c: c.seq) if board else []
    entities, seen_e = [], set()
    for card in cards:
        for e in card.entities:
            if (e.kind, e.name) not in seen_e:
                seen_e.add((e.kind, e.name))
                entities.append(e)
    return RecallResult(
        candidates=candidates, evidences=evidences, conflicts=conflicts, role_reports=reports,
        raw_candidate_count=raw, dedup_removed=raw - len(candidates), cards=cards, related_entities=entities, by_channel=pool.by_channel(),
    )
