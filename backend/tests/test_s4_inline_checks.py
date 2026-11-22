"""④ 选择类角色已并入召回 agent：评估维度由召回角色提供；原来的核对（新鲜度 / 维度覆盖 / 佐证）在召回 agent 里内联执行，
发现问题时仍通过转交让别的 agent 补查。L1（select_only）不再召回，只核验已知资产并按需查口径。"""
import pytest

from app.deps import Deps
from app.roles import REGISTRY, list_role_specs
from app.schemas.knowledge import KnowledgeItem, KnowledgeKind, KnowledgeOrigin, PreInjectionResult
from app.schemas.recall import AssetMeta, CandidateAsset, EvidenceKind
from app.schemas.workflow import EdgeKind, EntryLevel, WorkflowEdge, WorkflowGraph
from app.services.fakes import FakeBIRetrieval, FakeLineageRetrieval
from app.stages import RunContext
from app.stages.s2_coordination import CoordinationStage
from app.stages.s3_planning import PlanningStage
from app.stages.s3_planning.candidates import make_node
from app.stages.s4_recall import RecallStage

BI, LN, CAL, OMNI, DIMV = "bi_asset_discovery", "lineage_discovery", "caliber_lookup", "omni_discovery", "dim_value_resolver"


@pytest.fixture(autouse=True)
def _handoff_on(monkeypatch):
    from app.settings import settings as _s
    monkeypatch.setattr(_s, "COLLAB_HANDOFF_ENABLED", True)


def A(i, name, *, fields=("dt", "dau"), latest="2026-11-12", grain="day", typ="dataset"):
    return CandidateAsset(asset_id=i, name=name, asset_type=typ, retrieval_score=0.8,
                          meta=AssetMeta(grain=grain, partition_field="dt" if latest else None, partition_latest=latest, fields=list(fields),
                                         field_aliases={k: v for k, v in {"channel": "渠道", "dau": "DAU"}.items() if k in fields}))


class Spy:
    def __init__(self):
        self.bi_search, self.bi_get, self.ln, self.kb = [], [], [], []


def deps_for(spy, bi_assets, lineage_assets=None, kb_items=None):
    class BI(FakeBIRetrieval):
        async def search(self, query, **kw):
            spy.bi_search.append(query)
            return [a.model_copy(deep=True) for a in bi_assets]

        async def get(self, ids):
            spy.bi_get.append(list(ids))
            return [next((a.model_copy(deep=True) for a in bi_assets if a.asset_id == i), A(i, f"resolved {i}")) for i in ids]

    class LN(FakeLineageRetrieval):
        async def search(self, query, *, limit=10, cookie=None, params=None):
            spy.ln.append((query, list((params or {}).get("seed_assets", []))))
            return [a.model_copy(deep=True) for a in (lineage_assets or [])]

    class KB:
        async def search(self, query, **kw):
            spy.kb.append(query)
            return list(kb_items or [])

    return Deps(bi=BI(), lineage=LN(), caliber_kb=KB())


async def run(specs, deps, query="最近7天各渠道的DAU", edges=(), mode="recall_and_select", known=(), pre=None):
    ctx = RunContext(query=query, deps=deps)
    ctx.preinjection = pre
    ctx.contract = await CoordinationStage().run(ctx)
    ctx.plan = await PlanningStage().run(ctx)
    g = WorkflowGraph(nodes=[make_node(REGISTRY[r].spec, t) for r, t in specs], edges=[WorkflowEdge(src=a, dst=b, kind=EdgeKind.DEP) for a, b in edges])
    ctx.plan = ctx.plan.model_copy(update={"chosen": ctx.plan.chosen.model_copy(update={"graph": g}), "max_parallel": 8, "known_assets": list(known), "mode": mode})
    return await RecallStage().run(ctx)


GOOD = [A("dataset:1", "DAU 数据集", fields=("dt", "channel", "dau"))]


# ── 注册表与维度提供者 ───────────────────────────────────────────────────────────
def test_the_three_select_roles_are_gone_and_every_role_is_a_recall_role():
    assert set(REGISTRY) == {BI, LN, CAL, OMNI, DIMV}
    assert all(s.phase == "recall" and s.template_name for s in list_role_specs())


def test_recall_roles_provide_the_evaluation_dimensions_the_select_roles_used_to():
    d = {s.name: set(s.dims) for s in list_role_specs()}
    assert {"grain_fit", "freshness"} <= d[BI] and "dim_coverage" in d[LN] and "caliber" in d[CAL] and "dim_coverage" in d[DIMV]
    assert {"metric_match", "caliber", "grain_fit", "freshness", "dim_coverage"} <= d[OMNI]
    assert "permission" not in set().union(*d.values())                  # permission 由 ⑤ 的确定性权限过滤负责，不再要求节点提供


# ── 内联核对：新鲜度（BI 角色）──────────────────────────────────────────────────────
async def test_bi_agent_checks_freshness_of_its_own_candidates_and_reports_the_check():
    rec = await run([(BI, ["DAU"])], deps_for(Spy(), GOOD))
    fr = [e for e in rec.evidences if e.kind == EvidenceKind.FRESHNESS]
    assert fr and fr[0].role == BI and "2026-11-12" in fr[0].statement
    checks = [e for e in rec.collab.events if e.kind == "check"]
    assert [e.a for e in checks] == [f"{BI}@DAU"] and "checked 1" in checks[0].note
    info = {a.agent_id: a for a in rec.collab.agents}[f"{BI}@DAU"]
    assert info.checked == 1 and info.issues == 0


async def test_missing_partition_metadata_is_handed_to_bi_to_resolve():
    spy = Spy()
    thin = [A("dataset:9", "DAU 数据集", fields=("dt", "dau"), latest=None, grain=None)]
    rec = await run([(BI, ["DAU"])], deps_for(spy, thin))
    hs = [(e.a, e.via) for e in rec.collab.events if e.kind in ("ask", "probe") and e.via.startswith("freshness:")]
    assert hs and hs[0][1] == "freshness:dataset:9"
    assert any("dataset:9" in ids for ids in spy.bi_get)
    assert {a.agent_id: a for a in rec.collab.agents}[f"{BI}@DAU"].issues == 1


async def test_complete_metadata_triggers_no_handoff():
    rec = await run([(BI, ["DAU"])], deps_for(Spy(), GOOD))
    assert [e for e in rec.collab.events if e.via.startswith("freshness:")] == []


# ── 内联核对：维度覆盖（血缘角色）─────────────────────────────────────────────────────
async def test_lineage_agent_hands_an_uncovered_dimension_back_to_lineage_with_candidates_as_seeds():
    spy = Spy()
    no_channel = [A("dataset:1", "DAU 数据集", fields=("dt", "dau"))]
    rec = await run([(BI, ["DAU"]), (LN, ["DAU"])], deps_for(spy, no_channel, [A("table:t", "dws_dau_channel", typ="table")]),
                    edges=[(f"{BI}@DAU", f"{LN}@DAU")])
    assert [e for e in rec.collab.events if e.via.startswith("dimension:") and e.need_key == "lineage:渠道"]
    q, seeds = [x for x in spy.ln if x[0] == "渠道"][0]
    assert "dataset:1" in seeds
    assert {a.agent_id: a for a in rec.collab.agents}[f"{LN}@DAU"].issues >= 1
    assert "table:t" in {c.asset_id for c in rec.candidates}


async def test_no_dimension_handoff_when_a_candidate_covers_every_dimension():
    rec = await run([(BI, ["DAU"]), (LN, ["DAU"])], deps_for(Spy(), GOOD, []), edges=[(f"{BI}@DAU", f"{LN}@DAU")])
    assert [e for e in rec.collab.events if e.via.startswith("dimension:")] == []


async def test_dimension_check_is_skipped_when_the_question_has_no_dimension():
    spy = Spy()
    rec = await run([(BI, ["DAU"]), (LN, ["DAU"])], deps_for(spy, [A("dataset:1", "d", fields=("dt", "dau"))], []), query="DAU", edges=[(f"{BI}@DAU", f"{LN}@DAU")])
    assert [e for e in rec.collab.events if e.via.startswith("dimension:")] == []


# ── 内联核对：佐证（口径角色）─────────────────────────────────────────────────────────
async def test_caliber_agent_hands_a_candidate_sharing_no_field_back_for_caliber_lookup():
    odd = [A("dataset:1", "DAU 数据集", fields=("dt", "dau")), A("dataset:2", "另一张表", fields=("zz", "yy"))]
    rec = await run([(BI, ["DAU"]), (CAL, ["DAU"])], deps_for(Spy(), odd), edges=[(f"{BI}@DAU", f"{CAL}@DAU")])
    assert [e for e in rec.collab.events if e.via.startswith("crosscheck:")]
    other = [e for e in rec.evidences if e.source == "cross_check"]
    assert other and all(e.role == CAL for e in other)


async def test_single_candidate_cannot_be_cross_checked():
    rec = await run([(BI, ["DAU"]), (CAL, ["DAU"])], deps_for(Spy(), GOOD), edges=[(f"{BI}@DAU", f"{CAL}@DAU")])
    assert any("仅有单一候选" in e.statement for e in rec.evidences if e.source == "cross_check")


# ── 全通道角色：三类核对都做 ──────────────────────────────────────────────────────────
async def test_omni_agent_runs_all_three_checks():
    thin = [A("dataset:1", "DAU 数据集", fields=("dt", "dau"), latest=None, grain=None), A("dataset:2", "另一张表", fields=("zz",))]
    rec = await run([(OMNI, ["DAU"])], deps_for(Spy(), thin, []))
    vias = {e.via.split(":")[0] for e in rec.collab.events if e.via}
    assert {"dimension", "crosscheck"} <= vias                                                           # 新鲜度的转交可能因同名的 assets 卡已被别的探查产出而去重（见 BI 角色的专门用例）
    info = {a.agent_id: a for a in rec.collab.agents}[f"{OMNI}@DAU"]
    assert info.issues >= 3                                                                              # 但三类问题都被发现并计入：缺分区 1 + 缺维度 1 + 对不上 2
    assert any(e.kind.value == "freshness" for e in rec.evidences)
    assert sum(1 for e in rec.collab.events if e.kind == "check" and e.a == f"{OMNI}@DAU") == 1          # 只汇报一次


# ── 约束 ────────────────────────────────────────────────────────────────────────────
async def test_checks_still_report_but_never_hand_off_when_handoff_is_disabled():
    spy = Spy()
    deps = deps_for(spy, [A("dataset:9", "DAU 数据集", fields=("dt", "dau"), latest=None, grain=None)])
    deps.settings = deps.settings.model_copy(update={"COLLAB_HANDOFF_ENABLED": False})
    rec = await run([(BI, ["DAU"])], deps)
    assert any(e.kind == "check" for e in rec.collab.events) and not [e for e in rec.collab.events if e.via]


async def test_handoffs_from_checks_obey_the_shared_budget():
    rec = await run([(BI, ["DAU"])], deps_for(Spy(), [A(f"dataset:{i}", f"d{i}", latest=None, grain=None) for i in range(5)]))
    assert rec.collab.stats["handoffs"] <= 12


# ── L1（select_only）：不召回，只核验已知资产并按需查口径 ───────────────────────────────────
PRE_L1 = PreInjectionResult(injected=[KnowledgeItem(
    id="k", kind=KnowledgeKind.METRIC_DEF, title="DAU", content="日活", origin=KnowledgeOrigin.RETRIEVAL, asset_ids=["dataset:1"], consistency="possible_inconsistent")])


async def test_l1_plan_is_a_verification_graph_made_of_recall_roles_that_provide_every_required_dim():
    ctx = RunContext(query="DAU", deps=Deps())
    ctx.preinjection = PRE_L1
    ctx.contract = await CoordinationStage().run(ctx)
    plan = await PlanningStage().run(ctx)
    assert plan.level == EntryLevel.L1 and plan.mode == "select_only" and plan.chosen.feasible and not plan.degraded
    provided = {d for n in plan.chosen.graph.nodes for d in REGISTRY[n.role].spec.dims}
    assert plan.chosen.graph.nodes and set(plan.context.required_dims) <= provided
    assert all(n.phase == "recall" for n in plan.chosen.graph.nodes)


async def test_l1_does_not_search_bi_or_lineage_but_reads_knowledge_and_checks_the_known_assets():
    spy = Spy()
    deps = deps_for(spy, GOOD, [], kb_items=[KnowledgeItem(id="k2", kind=KnowledgeKind.METRIC_DEF, title="DAU", content="日活口径", origin=KnowledgeOrigin.RETRIEVAL,
                                                           asset_ids=["dataset:1"], relevance=0.8)])
    ctx = RunContext(query="DAU", deps=deps)
    ctx.preinjection = PRE_L1
    ctx.contract = await CoordinationStage().run(ctx)
    ctx.plan = await PlanningStage().run(ctx)
    assert ctx.plan.mode == "select_only"
    rec = await RecallStage().run(ctx)
    assert spy.bi_search == [] and spy.ln == [] and spy.kb                      # 不再检索 BI / 血缘；口径存疑 → 查知识全文
    assert any(e.kind == EvidenceKind.FRESHNESS for e in rec.evidences)         # 对已知资产（seed）做了核验
    assert [c.asset_id for c in rec.candidates] == ["dataset:1"] and all(r.status == "ok" for r in rec.role_reports)


async def test_seed_candidates_reach_every_node_not_just_checkers():
    spy = Spy()
    rec = await run([(BI, ["DAU"])], deps_for(spy, GOOD), mode="select_only", known=["dataset:1"])
    assert spy.bi_search == [] and any(e.kind == EvidenceKind.FRESHNESS for e in rec.evidences)


async def test_l0_graph_stays_empty():
    pre = PreInjectionResult(injected=[KnowledgeItem(id="k", kind=KnowledgeKind.METRIC_DEF, title="DAU", content="日活", origin=KnowledgeOrigin.RETRIEVAL, asset_ids=["dws.user_active_d"])])
    ctx = RunContext(query="DAU", deps=Deps(), confirmed_assets={"DAU": ["dws.user_active_d"]})
    ctx.preinjection = pre
    ctx.contract = await CoordinationStage().run(ctx)
    ctx.plan = await PlanningStage().run(ctx)
    assert ctx.plan.mode == "execute" and ctx.plan.chosen.graph.nodes == []


# ── 维度覆盖核对与 ⑤ 共用同一个字段匹配器 ──────────────────────────────────────────────────────
async def test_dimension_check_does_not_count_a_metric_role_field_as_covering_the_dimension():
    trap = [A("dataset:1", "DAU 数据集", fields=("dt", "gmv_by_channel", "dau"))]
    trap[0].meta.field_aliases = {"gmv_by_channel": "渠道GMV"}
    trap[0].meta.field_roles = {"gmv_by_channel": "metric"}
    rec = await run([(BI, ["DAU"]), (LN, ["DAU"])], deps_for(Spy(), trap, [A("table:t", "dws_channel", typ="table")]), edges=[(f"{BI}@DAU", f"{LN}@DAU")])
    assert [e for e in rec.collab.events if e.via.startswith("dimension:")]          # 指标字段不算覆盖「渠道」→ 转交血缘去找


async def test_dimension_check_accepts_a_knowledge_synonym_like_selection_does():
    syn = [A("dataset:1", "DAU 数据集", fields=("dt", "source", "dau"))]
    pre = PreInjectionResult(injected=[KnowledgeItem(id="k", kind=KnowledgeKind.DIMENSION, title="渠道", content="渠道", origin=KnowledgeOrigin.RETRIEVAL, terms=["source"])])
    rec = await run([(BI, ["DAU"]), (LN, ["DAU"])], deps_for(Spy(), syn, []), edges=[(f"{BI}@DAU", f"{LN}@DAU")], pre=pre)
    assert [e for e in rec.collab.events if e.via.startswith("dimension:")] == []
