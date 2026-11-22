"""⑤ 选择 Agent：LLM 复核 + 校验 + 硬护栏 + 规则兜底；阶段接线与 HTTP。"""
import json

import pytest
from fastapi.testclient import TestClient

from app.deps import Deps
from app.errors import UpstreamError
from app.main import create_app
from app.schemas.collab import EvidenceCard, RelatedEntity
from app.schemas.recall import AssetMeta, CandidateAsset
from app.services.fakes import FakeBIRetrieval, FakeKnowledgeRetrieval, FakePermission
from app.stages import RunContext
from app.stages.s2_coordination import CoordinationStage
from app.stages.s3_planning import PlanningStage
from app.stages.s4_recall import RecallStage
from app.stages.s5_selection import SelectionAgent, SelectionInput, SelectionStage, rule_select
from tests.test_s5_rules import A, C, K


class FakeLLM:
    def __init__(self, reply=None, exc=None):
        self.reply, self.exc, self.prompts = reply, exc, []

    async def complete(self, *, prompt, system=None, json_mode=False):
        self.prompts.append(prompt)
        if self.exc:
            raise self.exc
        return self.reply if isinstance(self.reply, str) else json.dumps(self.reply, ensure_ascii=False)


def inp3(**kw):
    a = A("a", "A", fields=("dt", "channel", "dau"), pks=("dt", "channel"))
    b = A("b", "B", fields=("dt", "dau"), pks=())
    c = A("c", "C", fields=("dt", "gmv"))
    return SelectionInput(contract=C(dims=("渠道",)), candidates=[a, b, c], **kw)


GOOD = {
    "primary": [{"asset_id": "a", "reason": {"结论": "覆盖最全", "命中": [{"业务词": "渠道", "类型": "dimension", "匹配到": "★渠道|channel", "位置": "fields[1]", "匹配方式": "字面相同"}],
                                             "未命中": [], "采用规则": ["字段覆盖度第一"], "降权信号": [], "盲区": []},
                 "query_fields": [{"field_name": "dau", "field_alias": "DAU", "role": "metric"}, {"field_name": "channel", "role": "dimension"}]}],
    "alternates": [{"asset_id": "b", "reason": "缺渠道维度"}],
    "discarded": [{"asset_id": "c", "reason": {"结论": "无目标指标字段"}}],
    "knowledge_refs": [],
}


async def run(llm, inp=None):
    return await SelectionAgent(llm).run(inp or inp3())


# ── LLM 复核 ────────────────────────────────────────────────────────────
async def test_llm_result_is_used_and_prompt_contains_candidates_hints_and_guards():
    llm = FakeLLM(GOOD)
    res = await run(llm)
    assert res.mode == "llm" and [d.asset_id for d in res.primary] == ["a"] and [d.asset_id for d in res.alternates] == ["b"] and [d.asset_id for d in res.discarded] == ["c"]
    assert res.primary[0].reason.conclusion == "覆盖最全" and res.primary[0].reason.hits[0].matched == "★渠道|channel"
    assert res.alternates[0].reason.conclusion == "缺渠道维度"                       # 字符串形式的理由也接受
    p = llm.prompts[0]
    assert "DAU" in p and "★渠道|channel" in p and '"hints"' in p and "{{" not in p and "grain_hard" not in p.split("## 输入")[0]
    assert [f.field_name for f in res.primary[0].query_fields] == ["dau", "channel"] and res.primary[0].query_fields[1].field_alias == "渠道"       # LLM 没写别名：用候选的中文名补全


async def test_prompt_candidate_content_cannot_inject_placeholders():
    a = A("a", "{{knowledge_context}} 注入", fields=("dt", "dau"))
    llm = FakeLLM(GOOD)
    await run(llm, SelectionInput(contract=C(), candidates=[a], knowledge=[K("k1", "kn", content="SECRET")]))
    assert "{{knowledge_context}} 注入" in llm.prompts[0] and llm.prompts[0].count("SECRET") == 1


async def test_fenced_json_is_accepted():
    assert (await run(FakeLLM("```json\n" + json.dumps(GOOD) + "\n```"))).mode == "llm"


@pytest.mark.parametrize("llm", [FakeLLM(exc=UpstreamError("llm", "boom")), FakeLLM("not json"), FakeLLM('[1]'), FakeLLM({"primary": "x"})])
async def test_failure_falls_back_to_rules_and_records_why(llm):
    res = await run(llm)
    assert res.mode == "rule" and [d.asset_id for d in res.primary] == ["a"] and res.groups[0].fallback_reason
    assert any("fell back to rules" in w for w in res.warnings)


async def test_null_llm_is_pure_rules():
    res = await run(Deps().llm)
    assert res.mode == "rule" and [d.asset_id for d in res.primary] == ["a"]


# ── 校验 / 护栏 ──────────────────────────────────────────────────────────
async def test_made_up_asset_ids_are_dropped_and_missing_ones_filled_from_rules():
    bad = {**GOOD, "alternates": [{"asset_id": "ghost", "reason": "x"}], "discarded": []}
    res = await run(FakeLLM(bad))
    ids = [d.asset_id for d in (*res.primary, *res.alternates, *res.discarded)]
    assert sorted(ids) == ["a", "b", "c"] and len(set(ids)) == 3
    assert any("ghost" in w for w in res.warnings) and any("filled in with the rule verdict" in w for w in res.warnings)
    assert res.decision("b").role == "alternate" and res.decision("c").role == "discarded"


async def test_two_primaries_keep_the_first_only():
    two = {**GOOD, "primary": [GOOD["primary"][0], {"asset_id": "b", "reason": "x", "query_fields": []}], "alternates": []}
    res = await run(FakeLLM(two))
    assert [d.asset_id for d in res.primary] == ["a"] and res.decision("b").role == "alternate" and any("2 primaries were returned" in w for w in res.warnings)


async def test_asset_in_two_tiers_is_kept_once():
    dup = {**GOOD, "discarded": [{"asset_id": "a", "reason": "x"}, {"asset_id": "c", "reason": "y"}]}
    res = await run(FakeLLM(dup))
    assert [d.asset_id for d in res.primary] == ["a"] and all(d.asset_id != "a" for d in res.discarded)


@pytest.mark.parametrize("name,inp_kw,hard", [
    ("permission", dict(perms={"b": (False, "无权限")}), "permission"),
    ("no_metric", {}, "no_metric_field"),
])
async def test_llm_cannot_pick_a_hard_ineligible_asset_as_primary(name, inp_kw, hard):
    pick = "b" if name == "permission" else "c"
    wrong = {"primary": [{"asset_id": pick, "reason": "x", "query_fields": []}], "alternates": [], "discarded": [], "knowledge_refs": []}
    res = await run(FakeLLM(wrong), inp3(**inp_kw))
    assert res.mode == "rule" and [d.asset_id for d in res.primary] == ["a"]                     # 整个口径回退规则
    assert hard in res.groups[0].fallback_reason


async def test_chart_that_fails_the_four_conditions_cannot_be_primary():
    ch = A("chart:1", "图", typ="chart", fields=(), is_strict=False, metric_ids=["DAU"], supported_dims=[])
    ds = A("ds", "DS", fields=("dt", "dau"))
    res = await run(FakeLLM({"primary": [{"asset_id": "chart:1", "reason": "x"}], "alternates": [], "discarded": [{"asset_id": "ds", "reason": "y"}]}),
                    SelectionInput(contract=C(), candidates=[ch, ds]))
    assert res.mode == "rule" and res.primary[0].asset_id == "ds"


async def test_nonexistent_query_fields_and_fake_hits_and_blind_spots_are_dropped():
    g = json.loads(json.dumps(GOOD))
    g["primary"][0]["query_fields"] = [{"field_name": "nope", "role": "metric"}]
    g["primary"][0]["reason"]["命中"].append({"业务词": "DAU", "类型": "metric", "匹配到": "编造的字段", "位置": "fields[2]", "匹配方式": "字面相同"})
    g["primary"][0]["reason"]["盲区"] = ["missing_chinese_name:100%"]
    res = await run(FakeLLM(g))
    p = res.primary[0]
    assert [h.term for h in p.reason.hits] == ["渠道"] and p.reason.blind_spots == []
    assert {f.field_name for f in p.query_fields} >= {"dau", "channel"}                         # 全被丢弃 → 用规则生成的 query_fields
    assert sum(1 for w in res.warnings if "dropped" in w) >= 3


async def test_blind_spots_may_cite_real_signals():
    g = json.loads(json.dumps(GOOD))
    a = inp3().candidates[0]
    from app.stages.s5_selection.rules import signals_for
    real = signals_for(a)
    g["primary"][0]["reason"]["盲区"] = real
    assert (await run(FakeLLM(g))).primary[0].reason.blind_spots == real


async def test_knowledge_refs_only_keep_known_ids():
    g = {**GOOD, "knowledge_refs": ["k1", "k_fake"]}
    res = await run(FakeLLM(g), inp3(knowledge=[K("k1", "渠道", terms=("渠道",))]))
    assert res.knowledge_refs == ["k1"] and any("knowledge_refs" in w for w in res.warnings)


async def test_primary_missing_from_llm_output_is_kept_as_alternate_not_restored():
    none = {"primary": [], "alternates": [{"asset_id": "b", "reason": "x"}], "discarded": [{"asset_id": "c", "reason": "y"}]}
    res = await run(FakeLLM(none))
    assert res.primary == [] and res.decision("a").role == "alternate" and "no_asset_selected" in res.risk_flags


async def test_multi_metric_calls_llm_once_per_caliber_and_one_failing_group_falls_back_alone():
    calls = []

    class LLM:
        async def complete(self, *, prompt, system=None, json_mode=False):
            calls.append(prompt)
            if "口径：`GMV`" in prompt:
                raise UpstreamError("llm", "boom")
            return json.dumps({"primary": [{"asset_id": "d", "reason": "x"}], "alternates": [], "discarded": [{"asset_id": "g", "reason": "y"}]})

    d, g = A("d", "D", fields=("dt", "dau")), A("g", "G", fields=("dt", "gmv"))
    res = await SelectionAgent(LLM()).run(SelectionInput(contract=C(("DAU", "GMV")), candidates=[d, g]))
    assert len(calls) == 2 and res.mode == "mixed"
    assert {x.caliber: (x.mode, x.primary) for x in res.groups} == {"DAU": ("llm", "d"), "GMV": ("rule", "g")}
    assert [x.asset_id for x in res.primary] == ["d", "g"]


# ── 阶段接线 ────────────────────────────────────────────────────────────
async def through(query, deps=None, **kw):
    ctx = RunContext(query=query, deps=deps or Deps(), **kw)
    ctx.contract = await CoordinationStage().run(ctx)
    ctx.plan = await PlanningStage().run(ctx)
    ctx.recall = await RecallStage().run(ctx)
    return ctx, await SelectionStage().run(ctx)


async def test_stage_maps_permission_denied_to_alternates_and_rejects_nothing_silently():
    ctx, res = await through("昨天GMV")
    d = res.decision("ads.secret_finance")
    assert d.role == "alternate" and d.needs_permission and "needs_permission:ads.secret_finance" in res.risk_flags
    assert res.primary and res.primary[0].asset_id != "ads.secret_finance"
    assert {x.asset_id for x in (*res.primary, *res.alternates, *res.discarded)} == {c.asset_id for c in ctx.recall.candidates}


async def test_stage_time_grain_and_multi_metric():
    _, res = await through("每天GMV")
    assert res.primary[0].asset_id == "dws.trade_gmv_d" and res.decision("dws.trade_gmv_m").role == "discarded" and "grain_hard" in res.decision("dws.trade_gmv_m").rule_ids
    _, res = await through("最近7天DAU和GMV")
    assert {d.asset_id for d in res.primary} == {"dws.user_active_d", "dws.trade_gmv_d"} and [g.caliber for g in res.groups] == ["DAU", "GMV"]


async def test_stage_uses_knowledge_entities_and_dimension_cards_from_recall():
    from app.stages.s5_selection.stage import build_input
    ctx, _ = await through("最近7天DAU")
    ctx.recall.cards.append(EvidenceCard(id="c1", key="dimension:自然堂", kind="dimension", subject="自然堂", author="dim_value_resolver",
                                        entities=[RelatedEntity(kind="dimension", name="商家名称", asset_ids=["x"])]))
    inp = build_input(ctx)
    assert [(r.term, r.field) for r in inp.dim_refs] == [("自然堂", "商家名称")] and any(k.title == "商家名称" for k in inp.knowledge)


async def test_assets_without_field_metadata_are_not_discarded_but_inferred_and_flagged():
    class BI(FakeBIRetrieval):
        async def search(self, query, **kw):
            return [CandidateAsset(asset_id="dataset:9", name="DAU 数据集", asset_type="dataset", retrieval_score=0.7, meta=AssetMeta())]

    class Empty(FakeBIRetrieval):
        async def search(self, query, **kw):
            return []

    class EmptyKB(FakeKnowledgeRetrieval):
        async def search(self, query, **kw):
            return []

    from app.services.fakes import FakeLineageRetrieval
    class EmptyLN(FakeLineageRetrieval):
        async def search(self, query, **kw):
            return []

    _, res = await through("最近7天DAU", Deps(bi=BI(), lineage=EmptyLN(), caliber_kb=EmptyKB(), kh=EmptyKB()))
    p = res.primary[0]
    assert p.asset_id == "dataset:9" and p.reason.hits[0].how == "推断" and res.ranking[0].dims["coverage"] == 0.5 and p.reason.hits[0].location == "name"
    assert {"fields_unknown:dataset:9", "partition_unknown:dataset:9"} <= set(res.risk_flags) and any("Field metadata missing" in b for b in p.reason.blind_spots)


async def test_assets_with_no_signal_at_all_go_to_alternates_awaiting_metadata():
    res = rule_select(SelectionInput(contract=C(), candidates=[CandidateAsset(asset_id="x", name="某表", asset_type="table", meta=AssetMeta(extra={"stub": True}))]))
    assert res.primary == [] and res.alternates[0].asset_id == "x" and "metadata missing" in res.alternates[0].reason.conclusion


async def test_llm_stage_end_to_end_with_real_pipeline_candidates():
    class LLM:
        async def complete(self, *, prompt, system=None, json_mode=False):
            import re
            if "为一个口径选出主资产" not in prompt:                                           # 协调 / 规划阶段的调用：不响应，走各自的规则兜底
                raise UpstreamError("llm", "not for me")
            ids = re.findall(r'"id": "([^"]+)"', prompt)                                         # candidates 块里的资产 id
            first, rest = ids[0], ids[1:]
            return json.dumps({"primary": [{"asset_id": first, "reason": {"结论": "LLM 选"}}], "alternates": [{"asset_id": i, "reason": "x"} for i in rest], "discarded": []})

    _, res = await through("昨天GMV", Deps(llm=LLM()))
    assert res.mode == "llm" and res.primary and res.primary[0].reason.conclusion == "LLM 选" and res.alternates


# ── HTTP ────────────────────────────────────────────────────────────────
def test_http_stage5_returns_three_tiers_with_structured_reasons():
    c = TestClient(create_app(Deps()))
    b = c.post("/api/stage5", json={"query": "最近7天各渠道的DAU", "max_parallel": 4}).json()
    s = b["selection"]
    assert b["status"] == "ready" and len(s["primary"]) == 1 and {"alternates", "discarded", "knowledge_refs", "risk_flags", "ranking", "groups", "mode"} <= set(s)
    p = s["primary"][0]
    assert p["role"] == "primary" and p["covers"] == ["DAU"] and p["reason"]["conclusion"] and p["reason"]["hits"] and p["query_fields"]
    assert {h["how"] for h in p["reason"]["hits"]} <= {"字面相同", "同义", "部分", "推断"}
    total = {d["asset_id"] for t in ("primary", "alternates", "discarded") for d in s[t]}
    assert total == {x["asset_id"] for x in b["recall"]["candidates"]}


def test_http_stage5_runs_even_when_metric_missing():
    b = TestClient(create_app(Deps())).post("/api/stage5", json={"query": "帮我看看数据"}).json()
    assert b["status"] == "ready" and b["selection"] is not None


def test_sse_stage5_event_carries_tiers():
    ev = TestClient(create_app(Deps())).post("/api/discover/sync", json={"query": "最近7天DAU"}).json()["events"]
    s5 = next(e for e in ev if e["event"] == "stage5")["result"]
    assert s5["primary"] and "alternates" in s5 and "discarded" in s5
