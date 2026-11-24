"""① 按消费方分发知识视图：契约 / profile / 构建 / 阶段接入 / 下游消费 / 回流。"""
import pytest

from app.schemas.knowledge import (
    CONSUMER_CHECKS, CONSUMER_COORDINATION, CONSUMER_PLANNER, CONSUMER_SELECTION, KnowledgeItem, KnowledgeKind,
    KnowledgeOrigin, KnowledgeView, PreInjectionResult, recall_consumer,
)
from app.stages.s1_preinjection.views import DEFAULT_PROFILES, KnowledgeProfile, build_view, build_views

K = KnowledgeKind


def _k(i, kind=K.METRIC_DEF, rel=0.9, cost=10, assets=(), title=None, pair=None):
    return KnowledgeItem(id=i, kind=kind, title=title or i, content=f"content of {i}", origin=KnowledgeOrigin.RETRIEVAL,
                         relevance=rel, token_cost=cost, asset_ids=list(assets), source_pair_id=pair)


# ── 契约 ────────────────────────────────────────────────────────────────
def test_view_falls_back_to_injected_when_no_views():
    pre = PreInjectionResult(injected=[_k("a"), _k("b")])
    assert [i.id for i in pre.view("planner")] == ["a", "b"]


def test_view_resolves_ids_against_injected_in_injected_order():
    pre = PreInjectionResult(injected=[_k("a"), _k("b"), _k("c")],
                             views={"x": KnowledgeView(consumer="x", item_ids=["c", "a"])})
    assert [i.id for i in pre.view("x")] == ["a", "c"]           # 视图是 injected 的子集，保持注入顺序


def test_empty_view_is_not_fallback():
    pre = PreInjectionResult(injected=[_k("a")], views={"x": KnowledgeView(consumer="x", item_ids=[])})
    assert pre.view("x") == []                                   # 视图存在但为空：该消费方就是不需要知识，不能回落到全量


def test_view_ids_unknown_to_injected_are_ignored():
    pre = PreInjectionResult(injected=[_k("a")], views={"x": KnowledgeView(consumer="x", item_ids=["ghost", "a"])})
    assert [i.id for i in pre.view("x")] == ["a"]


def test_consumed_ids_union_of_views_or_all_injected():
    pre = PreInjectionResult(injected=[_k("a"), _k("b"), _k("c")],
                             views={"x": KnowledgeView(consumer="x", item_ids=["a"]), "y": KnowledgeView(consumer="y", item_ids=["a", "c"])})
    assert pre.consumed_ids() == {"a", "c"}
    assert PreInjectionResult(injected=[_k("a")]).consumed_ids() == {"a"}      # 没有视图：视为全部被消费


def test_recall_consumer_name():
    assert recall_consumer("lineage_discovery") == "recall.lineage_discovery"


# ── 构建 ────────────────────────────────────────────────────────────────
def test_build_view_filters_by_kind_and_records_exclusion():
    items = [_k("m"), _k("d", K.DIMENSION), _k("p", K.PAIR)]
    v = build_view(KnowledgeProfile("c", {K.METRIC_DEF: 1.0, K.DIMENSION: 1.0}), items)
    assert v.item_ids == ["m", "d"]
    assert [(e.item_id, e.reason) for e in v.excluded] == [("p", "kind_filtered")]


def test_build_view_topk_keeps_highest_weighted_relevance_in_injected_order():
    items = [_k("a", rel=0.5), _k("b", rel=0.9), _k("c", rel=0.7)]
    v = build_view(KnowledgeProfile("c", {K.METRIC_DEF: 1.0}, topk=2), items)
    assert v.item_ids == ["b", "c"]                                       # a 被挤掉
    assert [e.reason for e in v.excluded] == ["topk"] and v.excluded[0].item_id == "a"


def test_build_view_kind_weight_reorders_priority():
    items = [_k("m", K.METRIC_DEF, rel=0.8), _k("d", K.DIMENSION, rel=0.7)]
    v = build_view(KnowledgeProfile("c", {K.METRIC_DEF: 1.0, K.DIMENSION: 1.5}, topk=1), items)
    assert v.item_ids == ["d"]                                            # 0.7*1.5=1.05 > 0.8


def test_build_view_token_budget_is_hard_cap():
    items = [_k("a", rel=0.9, cost=60), _k("b", rel=0.8, cost=60), _k("c", rel=0.7, cost=30)]
    v = build_view(KnowledgeProfile("c", {K.METRIC_DEF: 1.0}, token_budget=100), items)
    assert v.item_ids == ["a", "c"] and v.used_tokens == 90 and v.token_budget == 100
    assert [(e.item_id, e.reason) for e in v.excluded] == [("b", "over_budget")]


def test_profile_rejects_non_positive_weight():
    with pytest.raises(ValueError):
        KnowledgeProfile("c", {K.METRIC_DEF: 0})


def test_default_profiles_are_distinct_and_cover_all_consumers():
    names = set(DEFAULT_PROFILES)
    assert {CONSUMER_COORDINATION, CONSUMER_PLANNER, CONSUMER_SELECTION, CONSUMER_CHECKS} <= names
    for r in ("bi_asset_discovery", "lineage_discovery", "caliber_lookup", "omni_discovery", "dim_value_resolver"):
        assert recall_consumer(r) in names
    kinds = {n: frozenset(p.kinds) for n, p in DEFAULT_PROFILES.items()}
    assert kinds[recall_consumer("caliber_lookup")] != kinds[recall_consumer("dim_value_resolver")]
    assert K.PAIR not in kinds[CONSUMER_SELECTION]                         # ⑤ 不吃历史摘要块（与改造前一致）


def test_build_views_is_subset_of_injected_and_never_invents_items():
    injected = [_k("m", assets=["t1"]), _k("d", K.DIMENSION), _k("p", K.PAIR, assets=["t1"])]
    views = build_views(injected)
    ids = {i.id for i in injected}
    assert set(views) == set(DEFAULT_PROFILES)
    assert all(set(v.item_ids) <= ids for v in views.values())
    caliber = views[recall_consumer("caliber_lookup")].item_ids
    dim = views[recall_consumer("dim_value_resolver")].item_ids
    assert "d" in dim and "d" in caliber and "p" not in caliber


def test_build_views_custom_profiles():
    views = build_views([_k("m")], [KnowledgeProfile("only", {K.METRIC_DEF: 1.0})])
    assert list(views) == ["only"]


# ── 阶段接入 ────────────────────────────────────────────────────────────
async def test_stage_emits_views(make_ctx):
    from app.stages.s1_preinjection import PreInjectionStage
    res = await PreInjectionStage().run(make_ctx("昨天DAU是多少"))
    assert res.injected and set(res.views) == set(DEFAULT_PROFILES)
    ids = {i.id for i in res.injected}
    assert all(set(v.item_ids) <= ids for v in res.views.values())


async def test_stage_views_can_be_disabled(make_ctx, monkeypatch):
    from app.settings import settings
    from app.stages.s1_preinjection import PreInjectionStage
    monkeypatch.setattr(settings, "PREINJECT_VIEWS_ENABLED", False)
    res = await PreInjectionStage().run(make_ctx("昨天DAU是多少"))
    assert res.views == {} and [i.id for i in res.view("planner")] == [i.id for i in res.injected]
