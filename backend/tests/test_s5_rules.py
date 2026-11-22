"""⑤ 资产选择 · 规则引擎（对齐 old-version asset_selection.md 的核心选用原则 1-8）。"""
import pytest

from app.schemas.intent import Intent
from app.schemas.knowledge import KnowledgeItem, KnowledgeKind, KnowledgeOrigin
from app.schemas.recall import AssetMeta, CandidateAsset, Conflict, Evidence, EvidenceKind
from app.schemas.task import Slot, TaskContract, ValueSource
from app.stages.s5_selection.rules import DimRef, SelectionInput, rule_select, signals_for


def A(id, name, *, typ="table", fields=("dt", "dau"), aliases=None, pks=(), desc=None, heat=None, refs=None, grain="day", part="dt",
      score=0.6, **meta):
    aliases = {"dau": "DAU", "dt": "日期", "channel": "渠道", "gmv": "GMV"} if aliases is None else aliases
    return CandidateAsset(
        asset_id=id, name=name, asset_type=typ, retrieval_score=score,
        meta=AssetMeta(grain=grain, partition_field=part, fields=list(fields), field_aliases={k: v for k, v in aliases.items() if k in fields},
                       primary_keys=list(pks), description=desc, heat_30d=heat, ref_count=refs, **meta),
    )


def C(metrics=("DAU",), dims=(), *, query=None, sub_type=None, dim_terms=(), warnings=(), grain=None, time_range=None, intent_type="data_fetch", caliber=None):
    q = query or "".join(dims) + "".join(metrics)
    return TaskContract(
        query=q, metrics=Slot(value=list(metrics), source=ValueSource.USER), dimensions=Slot(value=list(dims), source=ValueSource.USER),
        grain=Slot(value=grain), time_range=Slot(value=time_range),
        intent=Intent(query=q, intent_type=intent_type, intent_sub_type=sub_type, domain="community", dim_terms=list(dim_terms), warnings=list(warnings)),
    )


def S(contract, cands, **kw):
    return SelectionInput(contract=contract, candidates=list(cands), **kw)


def tiers(res):
    return ([d.asset_id for d in res.primary], [d.asset_id for d in res.alternates], [d.asset_id for d in res.discarded])


def K(id, title, terms=(), kind=KnowledgeKind.DIMENSION, content=""):
    return KnowledgeItem(id=id, kind=kind, title=title, content=content or title, origin=KnowledgeOrigin.RETRIEVAL, terms=list(terms), relevance=0.9, token_cost=5)


# ── 原则 1：primary 恰好 1 个；三档完整且唯一 ───────────────────────────────────
def test_exactly_one_primary_and_every_candidate_in_exactly_one_tier():
    a = A("a", "A", fields=("dt", "channel", "dau"), pks=("dt", "channel"))
    b = A("b", "B", fields=("dt", "dau"), pks=())
    c = A("c", "C", fields=("dt", "gmv"))
    res = rule_select(S(C(dims=("渠道",)), [a, b, c]))
    assert tiers(res) == (["a"], ["b"], ["c"])
    ids = [d.asset_id for d in (*res.primary, *res.alternates, *res.discarded)]
    assert sorted(ids) == ["a", "b", "c"] and len(set(ids)) == 3
    assert res.primary[0].role == "primary" and res.alternates[0].role == "alternate" and res.discarded[0].role == "discarded"
    assert "No target metric field" in res.discarded[0].reason.conclusion and res.discarded[0].reason.misses[0].term == "DAU"


def test_empty_candidates_give_empty_tiers_and_a_flag():
    res = rule_select(S(C(), []))
    assert tiers(res) == ([], [], []) and "no_asset_selected" in res.risk_flags


def test_no_candidate_touches_the_metric_means_no_primary():
    res = rule_select(S(C(), [A("x", "X", fields=("dt", "gmv")), A("y", "Y", fields=("dt", "ctr"))]))
    assert tiers(res) == ([], [], ["x", "y"]) and "no_asset_selected" in res.risk_flags
    assert any(f.startswith("metrics_not_covered") for f in res.risk_flags)


def test_hive_table_only_can_still_be_primary():
    assert tiers(rule_select(S(C(), [A("t", "T", typ="table")])))[0] == ["t"]


# ── 原则 3：字段覆盖度 > 资产类型 > 数据层级 > 热度 ──────────────────────────────────
def test_field_coverage_beats_heat_and_type():
    full = A("full", "F", typ="table", fields=("dt", "channel", "dau"), pks=("dt", "channel"), heat=1)
    part = A("part", "P", typ="dataset", fields=("dt", "dau"), pks=(), heat=99999)
    res = rule_select(S(C(dims=("渠道",)), [part, full]))
    assert res.primary[0].asset_id == "full" and "highest field coverage" in res.primary[0].reason.rules
    assert next(x for x in res.ranking if x.asset_id == "full").dims["coverage"] == 1.0
    assert next(x for x in res.ranking if x.asset_id == "part").dims["coverage"] == 0.5


def test_similar_coverage_dataset_beats_hive_table():
    ds, tb = A("ds", "DS", typ="dataset"), A("tb", "TB", typ="table")
    res = rule_select(S(C(), [tb, ds]))
    assert res.primary[0].asset_id == "ds" and "asset type dataset > hive_table" in res.primary[0].reason.rules


def test_same_type_application_layer_beats_detail_layer():
    app, dwd = A("app.x", "应用", main_table="app_user_active"), A("dwd.x", "明细", main_table="dwd_user_active")
    res = rule_select(S(C(), [dwd, app]))
    assert res.primary[0].asset_id == "app.x" and "application layer first" in res.primary[0].reason.rules


def test_dataset_layer_is_read_from_name_keywords():
    res = rule_select(S(C(), [A("d1", "用户启动明细", typ="dataset"), A("d2", "用户日活宽表", typ="dataset")]))
    assert res.primary[0].asset_id == "d2"


def test_detail_layer_chosen_only_when_application_layer_lacks_the_dimension_and_says_so():
    app = A("app.x", "应用", fields=("dt", "dau"), pks=("dt",), main_table="app_a")
    dwd = A("dwd.x", "明细", fields=("dt", "channel", "dau"), pks=("dt", "channel"), main_table="dwd_a")
    res = rule_select(S(C(dims=("渠道",)), [app, dwd]))
    assert res.primary[0].asset_id == "dwd.x" and "Application-layer app.x lacks 渠道; using detail-layer dwd.x" in res.primary[0].reason.conclusion


def test_heat_breaks_ties_only_when_significantly_higher():
    hot, cold = A("hot", "H", heat=100), A("cold", "C", heat=40)
    assert rule_select(S(C(), [cold, hot])).primary[0].asset_id == "hot"
    near_a, near_b = A("a", "A", heat=100, score=0.7), A("b", "B", heat=90, score=0.9)
    assert rule_select(S(C(), [near_a, near_b])).primary[0].asset_id == "b"          # 热度不显著 → 退回召回分


def test_trend_prefers_summary_granularity():
    detail, summary = A("d", "用户明细表"), A("s", "大盘汇总表")
    res = rule_select(S(C(sub_type="trend"), [detail, summary]))
    assert res.primary[0].asset_id == "s" and "trend analysis prefers summary grain" in res.primary[0].reason.rules
    assert rule_select(S(C(), [detail, summary])).primary[0].asset_id in ("d", "s")        # 非趋势不触发该规则


# ── 原则 2：chart 严格四条件 ─────────────────────────────────────────────────────
def chart(**kw):
    base = dict(typ="chart", fields=(), is_strict=True, metric_ids=["DAU"], supported_dims=[])
    return A("chart:1", "DAU 趋势图", **{**base, **kw})


def test_chart_with_all_four_conditions_beats_a_better_dataset():
    ds = A("ds", "DS", typ="dataset", fields=("dt", "channel", "dau"), pks=("dt", "channel"), heat=99999)
    res = rule_select(S(C(dims=("渠道",)), [ds, chart()]))
    assert res.primary[0].asset_id == "chart:1" and "chart strict conditions" in res.primary[0].reason.rules and res.alternates[0].asset_id == "ds"


@pytest.mark.parametrize("override,needle,perm", [
    ({"is_strict": False}, "is_strict", None),
    ({"metric_ids": ["GMV"]}, "metric_ids", None),
    ({"supported_dims": ["平台"]}, "supported_dims", None),
    ({}, "perm", (False, "无权限")),
])
def test_chart_failing_any_condition_goes_to_alternates(override, needle, perm):
    ds = A("ds", "DS", typ="dataset", fields=("dt", "channel", "dau"), pks=("dt", "channel"))
    perms = {"chart:1": perm} if perm else {}
    res = rule_select(S(C(dims=("渠道",)), [chart(**override), ds], perms=perms))
    assert res.primary[0].asset_id == "ds"
    d = res.decision("chart:1")
    assert d.role == "alternate" and needle in d.reason.conclusion and any("chart" in x for x in d.reason.downweights)


def test_chart_supported_dims_empty_means_unrestricted():
    assert rule_select(S(C(dims=("渠道",)), [chart(supported_dims=[])])).primary[0].asset_id == "chart:1"


# ── 原则 4：OneDash 高权重 ───────────────────────────────────────────────────────
def test_onedash_wins_on_broad_trend_queries_but_not_on_specific_ones():
    od, other = A("od", "核心大盘", is_onedash=True, heat=10), A("ot", "普通", heat=9999)
    broad = rule_select(S(C(query="近30天大盘DAU趋势"), [other, od]))
    assert broad.primary[0].asset_id == "od" and "OneDash high weight" in broad.primary[0].reason.rules
    assert rule_select(S(C(query="自然堂的DAU"), [other, od])).primary[0].asset_id == "ot"


def test_onedash_loses_when_coverage_is_clearly_lower_or_it_is_a_cube():
    od_low = A("od", "核心大盘", fields=("dt", "dau"), pks=(), is_onedash=True)
    full = A("ot", "普通", fields=("dt", "channel", "dau"), pks=("dt", "channel"))
    assert rule_select(S(C(dims=("渠道",), query="近30天大盘各渠道DAU趋势"), [od_low, full])).primary[0].asset_id == "ot"
    od_cube = A("odc", "核心大盘cube", is_onedash=True, cube_filters={"operate_type": "整体"}, main_table="x_cube_1d_y")
    other = A("ot", "普通")
    res = rule_select(S(C(query="大盘DAU趋势"), [od_cube, other]))
    assert res.primary[0].asset_id == "ot" and any("OneDash" in x for x in res.decision("odc").reason.downweights)


# ── 原则 5：cube 表 ─────────────────────────────────────────────────────────────
def test_non_cube_is_preferred_and_cube_goes_to_alternates():
    cube = A("cube", "DAU cube", main_table="dws_x_cube_1d_y", cube_filters={"operate_type": "整体"})
    plain = A("plain", "DAU 汇总")
    res = rule_select(S(C(), [cube, plain]))
    assert res.primary[0].asset_id == "plain" and res.decision("cube").role == "alternate"
    assert any("cube" in x for x in res.decision("cube").reason.downweights)


def test_cube_as_primary_must_pin_the_slice_filters():
    cube = A("cube", "DAU cube", main_table="dws_x_cube_1d_y", cube_filters={"operate_type": "整体"})
    res = rule_select(S(C(), [cube]))
    p = res.primary[0]
    assert "cube requires slice filter" in p.reason.rules and "Cube table; the operate_type slice must be pinned" in p.reason.conclusion
    ref = next(f for f in p.query_fields if f.field_name == "operate_type")
    assert ref.dim_values == ["整体"] and ref.role == "dimension"


def test_cube_detected_from_main_table_name_alone():
    res = rule_select(S(C(), [A("c", "C", main_table="dws_a_summary_b")]))
    assert "cube requires slice filter" in res.primary[0].reason.rules and "cube_requires_filter:c" in res.risk_flags      # 没有切面信息：标风险而不是臆测


# ── 原则 6：粒度与 scope ─────────────────────────────────────────────────────────
def test_primary_key_must_contain_the_requested_dimension():
    seller = A("s", "粒度表S", fields=("dt", "seller_id", "channel", "dau"), pks=("dt", "seller_id"))
    res = rule_select(S(C(dims=("渠道",)), [seller, A("ok", "OK", fields=("dt", "channel", "dau"), pks=("dt", "channel"))]))
    assert res.primary[0].asset_id == "ok"
    d = res.decision("s")
    assert d.role == "discarded" and "grain" in d.reason.conclusion and "渠道" in d.reason.conclusion and "pk_granularity" in d.rule_ids


def test_unknown_primary_key_is_not_a_reason_to_discard_but_is_a_blind_spot():
    res = rule_select(S(C(dims=("渠道",)), [A("x", "X", fields=("dt", "channel", "dau"), pks=())]))
    assert res.primary[0].asset_id == "x" and "Primary-key info missing" in " ".join(res.primary[0].reason.blind_spots)


def test_scope_narrower_than_the_query_is_discarded():
    live, glob = A("live", "直播流量分析表", desc="直播场 DAU"), A("glob", "全站 DAU 宽表")
    res = rule_select(S(C(query="DAU"), [live, glob]))
    assert res.primary[0].asset_id == "glob"
    d = res.decision("live")
    assert d.role == "discarded" and "scope" in d.reason.conclusion and "直播" in d.reason.conclusion and "scope_insufficient" in d.rule_ids


def test_narrow_query_prefers_matching_scope_and_demotes_the_wide_table():
    live, glob = A("live", "直播流量分析表", desc="直播场 DAU", heat=1), A("glob", "全站 DAU 宽表", heat=9999)
    res = rule_select(S(C(query="直播间DAU"), [glob, live]))
    assert res.primary[0].asset_id == "live" and res.decision("glob").role == "alternate"
    assert any("scope" in x for x in res.decision("glob").reason.downweights)
    assert rule_select(S(C(query="直播间DAU"), [glob])).primary[0].asset_id == "glob"                  # 没有更精确的才用宽表


def test_asset_coarser_than_required_time_grain_is_discarded():
    month, day = A("m", "M", grain="month"), A("d", "D", grain="day")
    res = rule_select(S(C(grain="day"), [month, day]))
    assert res.primary[0].asset_id == "d" and res.decision("m").role == "discarded" and "grain_hard" in res.decision("m").rule_ids
    assert rule_select(S(C(grain="month"), [month, day])).primary[0].asset_id in ("m", "d")


# ── 原则 7：用户粒度宽表 ─────────────────────────────────────────────────────────
def test_user_grain_wide_table_requires_slice_dims_and_sum_aggregation():
    wide = A("w", "用户宽表", fields=("dt", "uid", "uv_pv_tag", "dau"), pks=("dt", "uid"))
    res = rule_select(S(C(), [wide]))
    p = res.primary[0]
    assert "user-grain wide table: sum" in p.reason.rules
    tag = next(f for f in p.query_fields if f.field_name == "uv_pv_tag")
    assert tag.role == "dimension" and tag.dim_values == [] and "sum" in tag.note and "distinct count" in tag.note
    assert "slice_values_unknown:w" in res.risk_flags


# ── 原则 8：权限 ────────────────────────────────────────────────────────────────
def test_permission_denied_goes_to_alternates_and_is_flagged_applyable():
    best = A("best", "B", fields=("dt", "channel", "dau"), pks=("dt", "channel"))
    ok = A("ok", "O", fields=("dt", "dau"), pks=())
    res = rule_select(S(C(dims=("渠道",)), [best, ok], perms={"best": (False, "缺少 finance 权限")}))
    assert res.primary[0].asset_id == "ok"
    d = res.decision("best")
    assert d.role == "alternate" and d.needs_permission and "缺少 finance 权限" in d.reason.conclusion and "permission demotion" in d.reason.rules


def test_only_candidate_without_permission_leaves_primary_empty():
    res = rule_select(S(C(), [A("x", "X")], perms={"x": (False, "无权限")}))
    assert tiers(res) == ([], ["x"], []) and "no_asset_selected" in res.risk_flags and "needs_permission:x" in res.risk_flags


# ── 填 query_fields ─────────────────────────────────────────────────────────────
def test_query_fields_cover_metric_dimension_and_partition_with_time_range_note():
    a = A("a", "A", fields=("dt", "channel", "dau"), pks=("dt", "channel"), part="dt")
    res = rule_select(S(C(dims=("渠道",), time_range="last_7_days"), [a]))
    qf = {f.field_name: f for f in res.primary[0].query_fields}
    assert qf["dau"].role == "metric" and qf["dau"].field_alias == "DAU" and qf["channel"].role == "dimension" and qf["channel"].field_alias == "渠道"
    assert qf["dt"].role == "dimension" and "explicit time range" in qf["dt"].note and "last_7_days" in qf["dt"].note


def test_unknown_partition_is_a_risk_flag_not_a_discard():
    res = rule_select(S(C(), [A("t", "T", part=None)]))
    assert res.primary[0].asset_id == "t" and "partition_unknown:t" in res.risk_flags
    assert any("Partition" in b for b in res.primary[0].reason.blind_spots)


def test_resolved_dim_value_fills_dim_values_from_the_dimension_card():
    a = A("a", "A", fields=("dt", "seller_name", "dau"), aliases={"seller_name": "商家名称", "dau": "DAU", "dt": "日期"}, pks=("dt", "seller_name"))
    res = rule_select(S(C(dim_terms=("自然堂",)), [a], dim_refs=[DimRef(term="自然堂", field="商家名称", asset_id="a")]))
    f = next(f for f in res.primary[0].query_fields if f.field_name == "seller_name")
    assert f.dim_values == ["自然堂"] and f.field_alias == "商家名称"
    assert any(h.term == "自然堂" and h.type == "dimension" for h in res.primary[0].reason.hits)


def test_dim_term_nobody_supports_means_no_primary_and_a_clear_reason():
    res = rule_select(S(C(dim_terms=("自然堂",)), [A("a", "A"), A("b", "B")]))
    assert res.primary == [] and {d.asset_id for d in res.alternates} == {"a", "b"}
    assert "No candidate supports filtering by 自然堂" in res.alternates[0].reason.conclusion and "dim_term_unsupported:自然堂" in res.risk_flags


# ── 知识桥接 / 理由结构 / 盲区 ────────────────────────────────────────────────────
def test_knowledge_bridges_business_word_to_physical_field_and_is_referenced():
    a = A("a", "A", fields=("dt", "channel", "dau"), aliases={"dau": "DAU"}, pks=("dt", "channel"))
    k = K("k_channel", "渠道", terms=("渠道", "CHANNEL"))
    res = rule_select(S(C(dims=("渠道",)), [a], knowledge=[k, K("k_other", "无关", terms=("无关",))]))
    h = next(h for h in res.primary[0].reason.hits if h.term == "渠道")
    assert h.how == "同义" and h.location == "fields[1]" and h.matched == "★channel"
    assert res.knowledge_refs == ["k_channel"] and "k_channel" in res.primary[0].reason.conclusion


def test_hits_echo_the_candidate_field_verbatim_including_alias_and_primary_key_mark():
    a = A("a", "A", fields=("dt", "channel", "dau"), pks=("dt", "channel"))
    res = rule_select(S(C(dims=("渠道",)), [a]))
    hits = {h.term: h for h in res.primary[0].reason.hits}
    assert hits["渠道"].matched == "★渠道|channel" and hits["渠道"].location == "fields[1]" and hits["渠道"].how == "字面相同"
    assert hits["DAU"].matched == "DAU|dau" and hits["DAU"].location == "fields[2]" and hits["DAU"].type == "metric"


def test_blind_spots_are_only_objective_signals():
    a = A("a", "A", fields=("dt", "channel", "gmv", "dau"), aliases={"dau": "DAU"}, pks=(), part=None)
    sig = signals_for(a)
    assert any(s.startswith("Low Chinese-alias coverage: 1/4") for s in sig) and "Primary-key info missing (grain cannot be judged)" in sig
    res = rule_select(S(C(), [a]))
    assert res.primary[0].reason.blind_spots and set(res.primary[0].reason.blind_spots) <= set(sig)


def test_multi_metric_selects_one_primary_per_metric_and_merges_a_shared_best_asset():
    dau = A("dau_t", "DAU表", fields=("dt", "dau"))
    gmv = A("gmv_t", "GMV表", fields=("dt", "gmv"))
    both = A("both", "全能表", fields=("dt", "dau", "gmv"), heat=10)
    res = rule_select(S(C(("DAU", "GMV")), [dau, gmv]))
    assert [d.asset_id for d in res.primary] == ["dau_t", "gmv_t"] and [d.covers for d in res.primary] == [["DAU"], ["GMV"]]
    res2 = rule_select(S(C(("DAU", "GMV")), [dau, gmv, both]))
    assert [d.asset_id for d in res2.primary] == ["both"] and res2.primary[0].covers == ["DAU", "GMV"]
    assert {g.caliber: g.primary for g in res2.groups} == {"DAU": "both", "GMV": "both"}
    assert {d.asset_id for d in res2.alternates} == {"dau_t", "gmv_t"}


def test_risk_flags_for_missing_caliber_evidence_conflicts_and_intent_warnings():
    a = A("a", "A")
    ev = Evidence(id="e1", asset_id="a", kind=EvidenceKind.CALIBER, statement="x", role="r", confidence=0.8)
    assert "no_caliber_evidence:a" not in rule_select(S(C(), [a], evidences=[ev])).risk_flags
    res = rule_select(S(C(warnings=["⚠️ domain 概念冲突：x"]), [a], conflicts=[Conflict(asset_id="a", topic=EvidenceKind.GRAIN, evidence_ids=["e1"], values=["d", "m"])]))
    assert {"no_caliber_evidence:a", "conflict:a:grain", "intent_conflict"} <= set(res.risk_flags)
    assert any("概念冲突" in w for w in res.warnings)


def test_ranking_scores_every_candidate_and_primary_scores_highest_on_the_rule_order():
    res = rule_select(S(C(), [A("a", "A", heat=100), A("b", "B", heat=1), A("c", "C", fields=("dt", "gmv"))]))
    assert {r.asset_id for r in res.ranking} == {"a", "b", "c"} and res.ranking[0].asset_id == res.primary[0].asset_id
    assert all(set(r.dims) >= {"coverage", "type_rank", "layer_rank", "heat", "retrieval"} for r in res.ranking)


def test_name_inferred_hit_counts_half_and_loses_to_a_real_field_match_at_equal_type():
    unknown = A("u", "DAU 数据集", typ="dataset", fields=(), part=None)
    real = A("r", "R", typ="dataset", fields=("dt", "dau"))
    res = rule_select(S(C(), [unknown, real]))
    assert res.primary[0].asset_id == "r" and next(x for x in res.ranking if x.asset_id == "u").dims["coverage"] == 0.5
    assert rule_select(S(C(), [unknown])).primary[0].reason.hits[0].how == "推断"          # 只有它时仍可用，不被丢弃
