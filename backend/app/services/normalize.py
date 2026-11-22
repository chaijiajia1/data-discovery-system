"""④ 混合检索归一化：BI 通道 + 知识通道（向量 + 图谱）→ 同一个候选池。

  bi                BI 通道：已认证的看板 / 数据集 / 指标层，是真实可用的基础资产（元信息最可信，合并时优先）
  knowledge:graph   知识通道·图谱路：沿血缘关系扩展出的表 / 数据集 / 图表，覆盖问法与命名脱节的漏召回
  knowledge:vector  知识通道·向量路：知识库 chunk 里关联的资产、指标、维度实体

归一化后每个候选带 `channels`（来源通道）与 [0,1] 的 `retrieval_score`；同一资产被多路命中时合并为一条：
通道取并集、元信息以 BI 为准并由知识通道补缺口、分数取最大；多路命中是强信号，融合分 +0.1（与 ① 的双路命中加成一致）。
"""
from __future__ import annotations

from ..schemas.knowledge import KnowledgeItem, KnowledgeKind
from ..schemas.collab import RelatedEntity
from ..schemas.recall import AssetMeta, CandidateAsset

CH_BI, CH_KGRAPH, CH_KVEC, CH_KNOWN = "bi", "knowledge:graph", "knowledge:vector", "known"
CH_ORDER = (CH_BI, CH_KGRAPH, CH_KVEC, CH_KNOWN)
KNOWLEDGE_DISCOUNT = 0.9        # 知识通道是补充：同样的相关度，略低于 BI 的真实基础
DUAL_CHANNEL_BONUS = 0.1


def _clamp(x: float) -> float:
    return round(max(0.0, min(1.0, float(x or 0.0))), 4)


def kind_of_id(asset_id: str) -> str:
    return {"dataset": "dataset", "chart": "chart", "dashboard": "dashboard", "metric": "metric"}.get(asset_id.split(":", 1)[0], "table")


def from_bi(assets: list[CandidateAsset]) -> list[CandidateAsset]:
    return [c.model_copy(update={"channels": [CH_BI], "retrieval_score": _clamp(c.retrieval_score)}, deep=True) for c in assets]


def from_graph(assets: list[CandidateAsset]) -> list[CandidateAsset]:
    return [c.model_copy(update={"channels": [CH_KGRAPH], "retrieval_score": _clamp(c.retrieval_score * KNOWLEDGE_DISCOUNT)}, deep=True) for c in assets]


def from_knowledge(items: list[KnowledgeItem], resolved: dict[str, CandidateAsset] | None = None) -> tuple[list[CandidateAsset], list[RelatedEntity]]:
    """知识 chunk 关联的资产 → 候选；`resolved` 是已取回完整元信息的资产（未解析的作占位 stub，等待补全）。"""
    resolved = resolved or {}
    cands: dict[str, CandidateAsset] = {}
    ents: list[RelatedEntity] = []
    for it in items:
        score = _clamp(it.relevance * KNOWLEDGE_DISCOUNT)
        for aid in it.asset_ids:
            if aid in resolved:
                c = resolved[aid].model_copy(update={"channels": [CH_KVEC], "retrieval_score": score}, deep=True)
            else:
                c = CandidateAsset(asset_id=aid, name=aid, asset_type=kind_of_id(aid), channels=[CH_KVEC], retrieval_score=score,  # type: ignore[arg-type]
                                   meta=AssetMeta(extra={"stub": True}))
            cur = cands.get(aid)
            cands[aid] = c if cur is None or c.retrieval_score > cur.retrieval_score else cur
        kind = {KnowledgeKind.METRIC_DEF: "metric", KnowledgeKind.DIMENSION: "dimension"}.get(it.kind)
        if kind:
            ents.append(RelatedEntity(kind=kind, name=it.title, asset_ids=list(it.asset_ids), source=it.id))  # type: ignore[arg-type]
    return list(cands.values()), ents


def _is_stub(c: CandidateAsset) -> bool:
    return bool(c.meta.extra.get("stub"))


def _ordered(channels) -> list[str]:
    s = set(channels)
    return [c for c in CH_ORDER if c in s] + sorted(s - set(CH_ORDER))


def merge_candidates(a: CandidateAsset, b: CandidateAsset) -> CandidateAsset:
    """同一资产的两条记录合并。参数顺序无关。"""
    real = [c for c in (a, b) if not _is_stub(c)]
    primary = next((c for c in real if CH_BI in c.channels), real[0] if real else a)
    other = b if primary is a else a
    meta = primary.meta.model_copy(deep=True)
    if not _is_stub(other):
        for f in AssetMeta.model_fields:
            if f in ("fields", "extra"):
                continue
            mine, theirs = getattr(meta, f), getattr(other.meta, f)
            if (mine is None or mine == [] or mine == {} or mine is False) and theirs not in (None, [], {}, False):
                setattr(meta, f, theirs)                       # 缺口由另一路补
        meta.fields = list(dict.fromkeys([*meta.fields, *other.meta.fields]))
        meta.field_aliases = {**other.meta.field_aliases, **meta.field_aliases}
        meta.extra = {**other.meta.extra, **meta.extra}
    if real:
        meta.extra.pop("stub", None)
    return primary.model_copy(update={
        "meta": meta, "channels": _ordered([*a.channels, *b.channels]), "found_by": sorted({*a.found_by, *b.found_by}),
        "evidence_ids": list(dict.fromkeys([*a.evidence_ids, *b.evidence_ids])),
        "retrieval_score": max(a.retrieval_score, b.retrieval_score), "name": primary.name or other.name,
    }, deep=True)


def fuse_score(c: CandidateAsset) -> float:
    return _clamp(c.retrieval_score + (DUAL_CHANNEL_BONUS if len(c.channels) >= 2 else 0.0))


class CandidatePool:
    """统一候选池：各通道归一化后的结果按 asset_id 合并。"""

    def __init__(self) -> None:
        self._by_id: dict[str, CandidateAsset] = {}

    def add(self, assets: list[CandidateAsset]) -> None:
        for c in assets:
            cur = self._by_id.get(c.asset_id)
            self._by_id[c.asset_id] = c.model_copy(deep=True) if cur is None else merge_candidates(cur, c)

    def get(self, asset_id: str) -> CandidateAsset | None:
        return self._by_id.get(asset_id)

    def ids(self) -> list[str]:
        return list(self._by_id)

    def items(self) -> list[CandidateAsset]:
        return list(self._by_id.values())

    def by_channel(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self._by_id.values():
            k = "+".join(c.channels) or "unknown"
            out[k] = out.get(k, 0) + 1
        return out
