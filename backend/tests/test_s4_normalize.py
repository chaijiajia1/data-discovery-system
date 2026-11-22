"""④ 混合检索：BI 通道与知识通道（向量 + 图谱）归一化进同一个候选池。"""
import pytest

from app.schemas.knowledge import KnowledgeItem, KnowledgeKind, KnowledgeOrigin
from app.schemas.recall import AssetMeta, CandidateAsset
from app.services.normalize import (
    CH_BI, CH_KGRAPH, CH_KVEC, CandidatePool, from_bi, from_graph, from_knowledge, fuse_score, merge_candidates,
)


def _a(i, *, score=0.5, grain=None, part=None, fields=(), typ="dataset", **extra):
    return CandidateAsset(asset_id=i, name=f"n_{i}", asset_type=typ, retrieval_score=score,
                          meta=AssetMeta(grain=grain, partition_field=part, fields=list(fields), extra=dict(extra)))


def _k(i, title, kind=KnowledgeKind.METRIC_DEF, assets=(), rel=0.8):
    return KnowledgeItem(id=i, kind=kind, title=title, content=f"{title} 的说明", origin=KnowledgeOrigin.RETRIEVAL, asset_ids=list(assets), relevance=rel)


def test_from_bi_tags_channel_and_clamps_score():
    out = from_bi([_a("dataset:1", score=1.7), _a("dataset:2", score=-0.2)])
    assert [c.channels for c in out] == [[CH_BI], [CH_BI]] and [c.retrieval_score for c in out] == [1.0, 0.0]


def test_from_graph_is_the_knowledge_graph_path_and_discounted_against_bi_base():
    out = from_graph([_a("table:t", score=0.8, typ="table")])
    assert out[0].channels == [CH_KGRAPH] and out[0].retrieval_score == pytest.approx(0.72)      # BI 是真实可用的基础，知识通道略折价


def test_from_knowledge_resolves_linked_assets_and_extracts_entities():
    items = [_k("k1", "DAU", assets=["dataset:1", "dws.t"]), _k("k2", "渠道", KnowledgeKind.DIMENSION, assets=["dws.t"]), _k("k3", "业务术语", KnowledgeKind.BUSINESS_TERM)]
    resolved = {"dataset:1": _a("dataset:1", grain="day", part="dt", fields=["dau"])}
    cands, ents = from_knowledge(items, resolved)
    by = {c.asset_id: c for c in cands}
    assert set(by) == {"dataset:1", "dws.t"} and all(c.channels == [CH_KVEC] for c in cands)
    assert by["dataset:1"].meta.grain == "day" and "stub" not in by["dataset:1"].meta.extra        # 已解析：带真实元信息
    assert by["dws.t"].meta.extra["stub"] is True and by["dws.t"].meta.grain is None                  # 未解析：占位，等待补全
    assert by["dataset:1"].retrieval_score == pytest.approx(0.8 * 0.9)
    assert {(e.kind, e.name) for e in ents} == {("metric", "DAU"), ("dimension", "渠道")}
    assert next(e for e in ents if e.name == "DAU").asset_ids == ["dataset:1", "dws.t"]


def test_merge_prefers_bi_meta_unions_channels_and_fills_gaps():
    bi = _a("x", score=0.6, grain="day", fields=["a"]); bi.channels = [CH_BI]; bi.found_by = ["r1"]
    kg = _a("x", score=0.9, part="dt", fields=["a", "b"], typ="table"); kg.channels = [CH_KGRAPH]; kg.found_by = ["r2"]
    m = merge_candidates(kg, bi)                                   # 参数顺序无关
    assert m.channels == [CH_BI, CH_KGRAPH] and m.found_by == ["r1", "r2"]
    assert m.meta.grain == "day" and m.meta.partition_field == "dt" and m.meta.fields == ["a", "b"]    # BI 元信息优先，缺口由知识通道补
    assert m.asset_type == "dataset" and m.retrieval_score == 0.9


def test_stub_never_overrides_real_meta():
    real = _a("x", grain="day"); real.channels = [CH_BI]
    stub = _a("x", grain="month", stub=True); stub.channels = [CH_KVEC]
    assert merge_candidates(stub, real).meta.grain == "day" and "stub" not in merge_candidates(stub, real).meta.extra
    assert merge_candidates(stub, stub.model_copy(deep=True)).meta.extra.get("stub") is True


def test_fuse_score_gives_dual_channel_bonus_like_stage1():
    one = _a("x", score=0.5); one.channels = [CH_BI]
    two = _a("y", score=0.5); two.channels = [CH_BI, CH_KGRAPH]
    capped = _a("z", score=0.95); capped.channels = [CH_BI, CH_KVEC]
    assert fuse_score(one) == 0.5 and fuse_score(two) == pytest.approx(0.6) and fuse_score(capped) == 1.0


def test_pool_merges_across_adds_and_counts_by_channel():
    pool = CandidatePool()
    pool.add(from_bi([_a("a", grain="day"), _a("b")]))
    pool.add(from_graph([_a("a", typ="table"), _a("c", typ="table")]))
    cands, _ = from_knowledge([_k("k", "DAU", assets=["c", "d"])], {})
    pool.add(cands)
    assert set(pool.ids()) == {"a", "b", "c", "d"}
    assert pool.get("a").channels == [CH_BI, CH_KGRAPH] and pool.get("c").channels == [CH_KGRAPH, CH_KVEC]
    assert pool.by_channel() == {"bi": 1, "bi+knowledge:graph": 1, "knowledge:graph+knowledge:vector": 1, "knowledge:vector": 1}
