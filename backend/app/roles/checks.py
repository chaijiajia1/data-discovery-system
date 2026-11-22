"""召回角色的内联核对：原来的三个选择类角色（新鲜度预检 / 维度粒度核对 / 佐证核验）的逻辑并进召回 agent。

每个召回角色按自己提供的评估维度（`RoleSpec.dims`）做对应核对——边召回边核对，不再有单独的选择阶段节点：
  freshness  → 新鲜度与分区预检：候选的最新分区；元信息缺失（无分区 / 无粒度）的转交给 BI 重新解析       （BI / 全通道）
  dim_coverage → 维度覆盖核对（字段匹配与 ⑤ 共用 asset_match）：所需维度没有任何候选覆盖时，转交给血缘找带该维度的表                          （血缘 / 全通道）
  caliber    → 佐证核验：top3 候选两两比对共同字段，对不上的转交给口径查找                                  （口径 / 全通道）
核对对象 = 本节点负责的指标在黑板上已有的证据卡里的候选 + L0/L1 已知资产（seed）+ 依赖节点的产出。
只产出证据、不增删候选；发现问题通过转交（`hub.delegate`，受深度 / 总预算 / 开关限制）让别的 agent 补查。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..asset_match import covered_dims
from ..schemas.collab import Need
from ..schemas.recall import CandidateAsset, Evidence, EvidenceKind
from .base import new_evidence_id
from .retrieve import make_need

if TYPE_CHECKING:  # pragma: no cover
    from .base import Role, RoleContext


def _need(rc: "RoleContext", kind: str, subject: str, hints: list[str], via: str) -> Need:
    n = make_need(rc.hub, kind, subject)
    return n.model_copy(update={"hint_assets": list(dict.fromkeys(hints)), "via": via, "hop": 1})


def gather_candidates(rc: "RoleContext", metrics: list[str]) -> list[CandidateAsset]:
    seen: dict[str, CandidateAsset] = {}
    board = rc.hub.board
    for m in metrics:
        for kind in ("assets", "lineage", "caliber"):
            card = board.cards.get(f"{kind}:{m}")
            for a in (board.assets_of(card) if card is not None else []):
                seen.setdefault(a.asset_id, a)
    for rep in rc.upstream.values():                    # seed（L0/L1 已知资产）与依赖节点的产出
        for a in rep.candidates:
            seen.setdefault(a.asset_id, a)
    return list(seen.values())


async def inline_checks(role: "Role", rc: "RoleContext", metrics: list[str]) -> list[Evidence]:
    spec, hub = role.spec, rc.hub
    dims = set(spec.dims)
    do_fresh = "freshness" in dims
    do_dim = "dim_coverage" in dims and "graph" in spec.channels and bool(rc.contract.dimensions.value)
    do_cross = "caliber" in dims and "knowledge" in spec.channels
    if not (do_fresh or do_dim or do_cross):
        return []
    cands = gather_candidates(rc, metrics)
    evs: list[Evidence] = []
    notes: list[str] = []
    issues = 0
    needs: list[Need] = []

    if do_fresh:
        thin = [c for c in cands if not c.meta.partition_latest or not c.meta.grain]
        evs += [Evidence(id=new_evidence_id(), asset_id=c.asset_id, kind=EvidenceKind.FRESHNESS,
                         statement=f"{c.name} 最新分区 {c.meta.partition_latest or '未知'}", value=c.meta.partition_latest,
                         role=spec.name, source="partition_meta", confidence=0.6 if c.meta.partition_latest else 0.2) for c in cands]
        needs += [_need(rc, "assets", c.name or c.asset_id, [c.asset_id], f"freshness:{c.asset_id}") for c in thin[:3]]
        issues += len(thin)
        notes.append(f"{len(thin)} missing partition meta" if thin else "partitions known")

    if do_dim:
        dims_wanted = [str(d) for d in (rc.contract.dimensions.value or [])]
        uncovered = list(dims_wanted)
        for c in cands:
            covered = covered_dims(c, dims_wanted, hub.env.knowledge)             # 与 ⑤ 共用 asset_match：字面 / 同义（知识桥接）/ 部分（角色一致）
            uncovered = [d for d in uncovered if d not in covered]
            evs.append(Evidence(id=new_evidence_id(), asset_id=c.asset_id, kind=EvidenceKind.FIELD, statement=f"{c.name} 覆盖维度 {covered or '无'}",
                                value=covered, role=spec.name, source="bi_asset", confidence=0.6))
        miss = [d for d in uncovered if cands]
        needs += [_need(rc, "lineage", d, [c.asset_id for c in cands[:3]], f"dimension:{d}") for d in miss]
        issues += len(miss)
        notes.append(f"no candidate covers {', '.join(miss)}" if miss else "dimensions covered")

    if do_cross:
        top = sorted(cands, key=lambda c: c.retrieval_score, reverse=True)[:3]
        odd: set[str] = set()
        if len(top) < 2:
            evs += [Evidence(id=new_evidence_id(), asset_id=c.asset_id, kind=EvidenceKind.OTHER, statement="仅有单一候选，无法交叉核验",
                             role=spec.name, source="cross_check", confidence=0.2) for c in top]
        for i, a in enumerate(top):
            for b in top[i + 1:]:
                shared = sorted({f.lower() for f in a.meta.fields} & {f.lower() for f in b.meta.fields})
                if not shared:
                    odd |= {a.asset_id, b.asset_id}
                for x, y in ((a, b), (b, a)):
                    evs.append(Evidence(id=new_evidence_id(), asset_id=x.asset_id, kind=EvidenceKind.OTHER, statement=f"与 {y.name} 的共同字段 {shared or '无'}",
                                        value=shared, role=spec.name, source="cross_check", confidence=0.7 if shared else 0.4))
        by_id = {c.asset_id: c for c in top}
        needs += [_need(rc, "caliber", by_id[i].name or i, [i], f"crosscheck:{i}") for i in sorted(odd)]
        issues += len(odd)
        notes.append(f"{len(odd)} share no fields" if odd else "consistent")

    sent = await hub.delegate(role.agent_id, needs)
    hub.report_check(role.agent_id, len(cands), issues, f"checked {len(cands)} · " + " · ".join(notes) + (" → handed off" if sent else ""))
    return evs
