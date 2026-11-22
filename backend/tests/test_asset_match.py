"""字段匹配的单一实现：④ 的维度覆盖核对与 ⑤ 的字段命中共用 `asset_match`，不再各写一套子串匹配。"""
from app.asset_match import covered_dims, match_term
from app.schemas.knowledge import KnowledgeItem, KnowledgeKind, KnowledgeOrigin
from app.schemas.recall import AssetMeta, CandidateAsset


def A(fields=("dt", "dau"), aliases=None, roles=None, typ="dataset", **meta):
    return CandidateAsset(asset_id="a", name="资产", asset_type=typ, meta=AssetMeta(fields=list(fields), field_aliases=aliases or {}, field_roles=roles or {}, **meta))


def K(title, terms=(), kind=KnowledgeKind.DIMENSION):
    return KnowledgeItem(id=f"k_{title}", kind=kind, title=title, content=title, origin=KnowledgeOrigin.RETRIEVAL, terms=list(terms))


def test_literal_and_alias_hits_count_as_covered():
    assert covered_dims(A(("dt", "channel"), {"channel": "渠道"}), ["渠道"], []) == ["渠道"]
    assert covered_dims(A(("dt", "channel")), ["channel"], []) == ["channel"]


def test_synonym_bridge_through_knowledge_counts_as_covered():
    c = A(("dt", "source"))
    assert covered_dims(c, ["渠道"], []) == []
    assert covered_dims(c, ["渠道"], [K("渠道", terms=["source"])]) == ["渠道"]


def test_a_field_whose_known_role_is_metric_is_not_a_dimension_for_partial_matches():
    c = A(("dt", "gmv_by_channel"), {"gmv_by_channel": "渠道GMV"}, roles={"gmv_by_channel": "metric"})
    assert covered_dims(c, ["渠道"], []) == []                                    # 旧的子串匹配会把它误判为覆盖了「渠道」


def test_name_inference_is_not_coverage_evidence():
    stub = CandidateAsset(asset_id="s", name="渠道日活表", asset_type="table", meta=AssetMeta())
    assert covered_dims(stub, ["渠道"], []) == []                                  # 没有字段元信息，只凭名称推断：不算覆盖


def test_charts_use_supported_dims_and_empty_means_unlimited():
    chart = CandidateAsset(asset_id="c", name="图", asset_type="chart", meta=AssetMeta(supported_dims=["渠道"]))
    assert covered_dims(chart, ["渠道", "平台"], []) == ["渠道"]
    free = CandidateAsset(asset_id="c2", name="图", asset_type="chart", meta=AssetMeta(supported_dims=[]))
    assert covered_dims(free, ["渠道", "平台"], []) == ["渠道", "平台"]


def test_match_term_is_still_the_rich_matcher_used_by_selection():
    hit, fld, _ = match_term("渠道", "dimension", A(("dt", "channel"), {"channel": "渠道"}), [])
    assert hit.how == "字面相同" and hit.matched == "渠道|channel" and fld == "channel"


def test_rules_module_reuses_the_shared_matcher():
    import app.stages.s5_selection.rules as rules
    import app.asset_match as m
    assert rules.match_term is m.match_term and rules.render_field is m.render_field and rules.is_chart is m.is_chart
