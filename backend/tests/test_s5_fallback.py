"""⑤ 在元信息缺失时不能让主资产落空：未知 ≠ 不满足（维值支撑 / chart 条件），全都不满足时放宽门槛选出最好的并显式标注。"""
from app.schemas.intent import Intent
from app.schemas.recall import AssetMeta, CandidateAsset
from app.schemas.task import Slot, TaskContract, ValueSource
from app.stages.s5_selection import SelectionInput, rule_select


def C(metrics=("GMV",), dims=(), dim_terms=(), grain=None):
    q = "".join(dims) + "".join(metrics)
    return TaskContract(query=q, metrics=Slot(value=list(metrics), source=ValueSource.USER), dimensions=Slot(value=list(dims), source=ValueSource.USER),
                        grain=Slot(value=grain), intent=Intent(query=q, domain="transaction", dim_terms=list(dim_terms)))


def chart(i, name, *, metrics=(), dims=(), strict=None, score=0.8):
    return CandidateAsset(asset_id=i, name=name, asset_type="chart", retrieval_score=score,
                          meta=AssetMeta(metric_ids=list(metrics), supported_dims=list(dims), is_strict=strict))


def bare_dataset(i, name, score=0.6):
    return CandidateAsset(asset_id=i, name=name, asset_type="dataset", retrieval_score=score, meta=AssetMeta())


def S(contract, cands, **kw):
    return SelectionInput(contract=contract, candidates=list(cands), **kw)


def tiers(res):
    return ([d.asset_id for d in res.primary], [d.asset_id for d in res.alternates], [d.asset_id for d in res.discarded])


# ── 兜底：没有任何满足全部条件的候选时，放宽门槛选出最好的 ─────────────────────────────────────────
def test_when_no_candidate_meets_every_gate_the_best_relaxed_one_is_chosen_and_flagged():
    res = rule_select(S(C(), [chart("c1", "GMV趋势", metrics=["GMV"], dims=["日期"]), chart("c2", "GMV明细", metrics=["DGMV"], dims=["日期"], score=0.5)]))
    assert tiers(res)[0] == ["c1"] and "fallback_candidate" in res.primary[0].rule_ids and "fallback_primary:c1" in res.risk_flags
    assert any("relaxed gate" in x for x in res.primary[0].reason.downweights) and res.alternates and res.alternates[0].asset_id == "c2"
    assert "no_asset_selected" not in res.risk_flags


def test_the_fallback_is_not_used_when_a_fully_eligible_candidate_exists():
    ok = chart("ok", "GMV", metrics=["GMV"], dims=[], strict=True)
    res = rule_select(S(C(), [chart("weak", "GMV趋势", metrics=["GMV"], dims=["日期"], score=0.99), ok]))
    assert tiers(res)[0] == ["ok"] and not any(f.startswith("fallback_primary") for f in res.risk_flags)


def test_a_candidate_that_does_not_touch_the_metric_is_never_promoted():
    res = rule_select(S(C(), [chart("c1", "留存", metrics=["留存"], dims=["日期"])]))
    assert tiers(res)[0] == [] and "no_asset_selected" in res.risk_flags


def test_permission_denied_candidates_are_not_promoted_by_the_fallback():
    res = rule_select(S(C(), [chart("c1", "GMV趋势", metrics=["GMV"], dims=["日期"])], perms={"c1": (False, "no scope")}))
    assert tiers(res)[0] == [] and res.alternates[0].needs_permission


def test_an_unsupported_named_dimension_value_is_still_a_hard_no_when_fields_are_known():
    fielded = CandidateAsset(asset_id="d", name="GMV表", asset_type="table", retrieval_score=0.7,
                             meta=AssetMeta(fields=["dt", "gmv"], field_aliases={"gmv": "GMV"}, partition_field="dt"))
    res = rule_select(S(C(dim_terms=["自然堂"]), [fielded]))
    assert tiers(res)[0] == [] and "dim_term_unsupported:自然堂" in res.risk_flags


# ── 维值支撑：没有字段元信息 = 无法判断，不是「不支持」 ────────────────────────────────────────────
def test_assets_without_field_metadata_are_not_ruled_out_by_a_named_dimension_value():
    res = rule_select(S(C(metrics=("DAU",), dim_terms=["渠道"]), [bare_dataset("a", "DAU日活看板"), bare_dataset("b", "留存")]))
    assert tiers(res)[0] == ["a"] and "dim_term_unsupported:渠道" not in res.risk_flags
    assert "dim_value_unresolved:渠道" in res.risk_flags                           # 但如实标注：没有解析出对应字段


def test_a_real_dimension_card_still_has_to_match_a_field_when_fields_exist():
    from app.stages.s5_selection.rules import DimRef
    fielded = CandidateAsset(asset_id="d", name="GMV表", asset_type="table", retrieval_score=0.7, meta=AssetMeta(fields=["dt", "gmv", "brand"], field_aliases={"gmv": "GMV"}))
    res = rule_select(S(C(dim_terms=["自然堂"]), [fielded], dim_refs=[DimRef(term="自然堂", field="brand")]))
    assert tiers(res)[0] == ["d"] and "dim_term_unsupported:自然堂" not in res.risk_flags


# ── LLM 路径：规则放宽后的候选，LLM 选它不会被「chart 四条件」护栏误拒；选了没命中口径的 chart 仍被拒 ─────────────────────────────
import json

from app.stages.s5_selection import SelectionAgent


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply

    async def complete(self, *, prompt, system=None, json_mode=False):
        return json.dumps(self.reply, ensure_ascii=False)


def _llm_pick(pick, other):
    return {"primary": [{"asset_id": pick, "reason": {"结论": "最相关"}}], "alternates": [{"asset_id": other, "reason": "次相关"}], "discarded": [], "knowledge_refs": []}


async def test_llm_may_choose_the_relaxed_candidate():
    cands = [chart("c1", "GMV趋势", metrics=["GMV"], dims=["日期"]), chart("c2", "GMV明细", metrics=["DGMV"], dims=["日期"], score=0.5)]
    res = await SelectionAgent(FakeLLM(_llm_pick("c2", "c1"))).run(S(C(), cands))
    assert res.mode == "llm" and [d.asset_id for d in res.primary] == ["c2"] and not res.warnings[:0]


async def test_llm_still_cannot_choose_a_chart_that_does_not_touch_the_metric():
    cands = [chart("c1", "GMV趋势", metrics=["GMV"], dims=["日期"]), chart("dau", "DAU趋势", metrics=["DAU"], dims=["日期"], score=0.99)]
    res = await SelectionAgent(FakeLLM(_llm_pick("dau", "c1"))).run(S(C(), cands))
    assert res.mode == "rule" and [d.asset_id for d in res.primary] == ["c1"] and "chart" in res.groups[0].fallback_reason
