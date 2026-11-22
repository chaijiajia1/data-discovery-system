import asyncio

from app.schemas.recall import CandidateAsset, Evidence, EvidenceKind, RoleReport
from app.stages.s4_recall.merger import merge


def _ev(i, aid, kind, val, role):
    return Evidence(id=i, asset_id=aid, kind=kind, statement=f"{kind.value}={val}", value=val, role=role, confidence=0.5)


def test_merge_dedup_and_conflict_preserved():
    a = CandidateAsset(asset_id="t1", name="T1")
    r1 = RoleReport(node_id="n1", role="r1", candidates=[a], evidences=[_ev("e1", "t1", EvidenceKind.GRAIN, "day", "r1")])
    r2 = RoleReport(node_id="n2", role="r2", candidates=[a], evidences=[_ev("e2", "t1", EvidenceKind.GRAIN, "month", "r2"),
                                                                    _ev("e3", "t1", EvidenceKind.GRAIN, "day", "r2")])
    res = merge([r1, r2])
    assert len(res.candidates) == 1 and res.dedup_removed == 1
    assert res.candidates[0].found_by == ["r1", "r2"]
    assert len(res.evidences) == 2                      # e3 与 e1 同陈述 → 去重
    assert len(res.conflicts) == 1 and set(res.conflicts[0].values) == {"day", "month"}


# ── ④ 按 ③ 的计划执行：已知资产物化 / 空图 / 选择类角色消费上游 ────────────────
from app.deps import Deps
from app.schemas.knowledge import KnowledgeItem, KnowledgeKind, KnowledgeOrigin, PreInjectionResult
from app.stages import RunContext
from app.stages.s2_coordination import CoordinationStage
from app.stages.s3_planning import PlanningStage
from app.stages.s4_recall import RecallStage


async def _plan_and_recall(query, pre=None, deps=None, **kw):
    ctx = RunContext(query=query, deps=deps or Deps(), **kw)
    ctx.preinjection = pre
    ctx.contract = await CoordinationStage().run(ctx)
    ctx.plan = await PlanningStage().run(ctx)
    return ctx, await RecallStage().run(ctx)


async def test_recall_executes_planned_roles_and_roles_check_what_they_found():
    ctx, rec = await _plan_and_recall("最近7天各渠道DAU")
    roles = {r.role for r in rec.role_reports}
    assert roles == {n.role for n in ctx.plan.chosen.graph.nodes} and rec.candidates
    assert all(r.status == "ok" for r in rec.role_reports)
    bi = next(r for r in rec.role_reports if r.role in ("bi_asset_discovery", "omni_discovery"))
    assert bi.candidates and any(e.kind.value == "freshness" for e in bi.evidences)       # 召回角色内联做了新鲜度核对


async def test_known_assets_are_materialized_without_recall_roles():
    pre = PreInjectionResult(injected=[KnowledgeItem(
        id="k", kind=KnowledgeKind.METRIC_DEF, title="DAU", content="日活", origin=KnowledgeOrigin.RETRIEVAL, asset_ids=["dws.user_active_d"])])
    ctx, rec = await _plan_and_recall("DAU", pre, confirmed_assets={"DAU": ["dws.user_active_d"]})
    assert ctx.plan.mode == "execute" and rec.role_reports == []
    assert [c.asset_id for c in rec.candidates] == ["dws.user_active_d"] and rec.candidates[0].found_by == ["preinjection"]
    assert rec.candidates[0].meta.grain == "day"                          # 经 bi.get 取回完整元信息，供 ⑤ 硬过滤使用
    assert any(e.kind.value == "caliber" for e in rec.evidences) or not pre.injected[0].asset_ids


async def test_l1_verification_nodes_consume_seed_candidates():
    pre = PreInjectionResult(injected=[KnowledgeItem(
        id="k", kind=KnowledgeKind.METRIC_DEF, title="DAU", content="日活", origin=KnowledgeOrigin.RETRIEVAL,
        asset_ids=["dataset:1"], consistency="possible_inconsistent")])
    ctx, rec = await _plan_and_recall("DAU", pre)
    assert ctx.plan.mode == "select_only"
    assert rec.role_reports and all(r.status == "ok" for r in rec.role_reports) and rec.candidates
    assert any(e.kind.value == "freshness" and e.asset_id == "dataset:1" for r in rec.role_reports for e in r.evidences)      # 对已知资产（seed）做了核验


async def test_lineage_and_dim_roles_run_via_deps():
    from app.services.fakes import FakeLineageRetrieval
    calls = []

    class SpyLineage(FakeLineageRetrieval):
        async def search(self, query, *, limit=10, cookie=None, params=None):
            calls.append((query, cookie, params and params.get("graph_domain")))
            return await super().search(query, limit=limit)

    ctx, rec = await _plan_and_recall("最近7天各渠道DAU", deps=Deps(lineage=SpyLineage()), cookie="common-internal-access-token-prod=t")
    assert any(n.role == "lineage_discovery" for n in ctx.plan.chosen.graph.nodes) == bool(calls)
    for q, ck, dom in calls:
        assert "DAU" in q and ck and dom                                   # 目标指标 / cookie / 检索参数都透传到通道


async def test_batched_node_queries_each_caliber_and_omni_runs_all_three_channels():
    from app.services.fakes import FakeBIRetrieval, FakeKnowledgeRetrieval, FakeLineageRetrieval
    from app.schemas.workflow import PlanningResult
    seen = {"bi": [], "lineage": [], "kb": []}

    class BI(FakeBIRetrieval):
        async def search(self, query, **kw):
            seen["bi"].append(query)
            return await super().search(query, limit=5)

    class LN(FakeLineageRetrieval):
        async def search(self, query, **kw):
            seen["lineage"].append(query)
            return await super().search(query, limit=5)

    class KB(FakeKnowledgeRetrieval):
        async def search(self, query, **kw):
            seen["kb"].append(query)
            return await super().search(query, limit=5)

    deps = Deps(bi=BI(), lineage=LN(), caliber_kb=KB())
    ctx = RunContext(query="DAU和GMV", deps=deps)
    ctx.contract = await CoordinationStage().run(ctx)
    ctx.plan = await PlanningStage().run(ctx)
    from app.stages.s3_planning.candidates import make_node
    from app.roles import REGISTRY
    node = make_node(REGISTRY["omni_discovery"].spec, ["DAU", "GMV"])
    ctx.plan = ctx.plan.model_copy(update={"chosen": ctx.plan.chosen.model_copy(update={"graph": type(ctx.plan.chosen.graph)(nodes=[node], edges=[])})})
    rec = await RecallStage().run(ctx)
    assert seen["bi"] == ["DAU", "GMV"] and seen["lineage"] == ["DAU", "GMV"] and seen["kb"] == ["DAU", "GMV"]     # 一个 Agent、每口径逐通道
    assert [r.role for r in rec.role_reports] == ["omni_discovery"] and rec.candidates
