"""④ 调度：按依赖尽早开始（ASAP），与 ③ 评分时的排程模型一致；不再按层设屏障。"""
import asyncio
import time

import pytest

from app.deps import Deps
from app.errors import ContractValidationError
from app.roles import REGISTRY
from app.schemas.workflow import EdgeKind, WorkflowEdge, WorkflowGraph
from app.services.fakes import FakeBIRetrieval
from app.stages import RunContext
from app.stages.s2_coordination import CoordinationStage
from app.stages.s3_planning import PlanningStage
from app.stages.s3_planning.candidates import make_node
from app.stages.s4_recall import RecallStage, scheduler

BI, SCOUT = "bi_asset_discovery", "caliber_lookup"


class SlowBI(FakeBIRetrieval):
    """按 query 决定延迟：GMV 慢，其余快。"""

    def __init__(self, slow=0.2, fast=0.02, boom=()):
        super().__init__()
        self.slow, self.fast, self.boom = slow, fast, set(boom)

    async def search(self, query, **kw):
        if query in self.boom:
            raise RuntimeError("down")
        await asyncio.sleep(self.slow if query == "GMV" else self.fast)
        return await super().search(query, limit=5)


@pytest.fixture
def spans(monkeypatch):
    out: dict[str, list[float]] = {}
    active = {"n": 0, "peak": 0}
    orig = scheduler._run_node

    async def spy(role, rc, timeout):
        out[rc.node.id] = [time.perf_counter(), 0.0]
        active["n"] += 1
        active["peak"] = max(active["peak"], active["n"])
        try:
            return await orig(role, rc, timeout)
        finally:
            active["n"] -= 1
            out[rc.node.id][1] = time.perf_counter()

    monkeypatch.setattr(scheduler, "_run_node", spy)
    out["_active"] = active          # type: ignore[assignment]
    return out


async def run(specs, edges, deps, a_max=8, query="DAU和GMV和CTR"):
    ctx = RunContext(query=query, deps=deps)
    ctx.contract = await CoordinationStage().run(ctx)
    ctx.plan = await PlanningStage().run(ctx)
    g = WorkflowGraph(nodes=[make_node(REGISTRY[r].spec, t) for r, t in specs], edges=[WorkflowEdge(src=a, dst=b, kind=EdgeKind.DEP) for a, b in edges])
    ctx.plan = ctx.plan.model_copy(update={"chosen": ctx.plan.chosen.model_copy(update={"graph": g}), "max_parallel": a_max, "known_assets": []})
    return ctx, await RecallStage().run(ctx)


async def test_a_dependent_node_starts_as_soon_as_its_own_predecessors_finish(spans):
    """A 快、B 慢，C 只依赖 A：整层屏障下 C 要等 B；ASAP 下 C 在 A 完成后立刻开始。"""
    t0 = time.perf_counter()
    _, rec = await run([(BI, ["DAU"]), (BI, ["GMV"]), (SCOUT, [])], [(f"{BI}@DAU", SCOUT)], Deps(bi=SlowBI()))
    a, b, c = spans[f"{BI}@DAU"], spans[f"{BI}@GMV"], spans[SCOUT]
    assert c[0] >= a[1] - 1e-3 and c[0] < b[1] - 0.05                       # C 在 A 之后、B 结束之前就开始了
    assert time.perf_counter() - t0 < 0.2 + 0.1                             # 总时长 ≈ 最慢的那条链，而不是 B + C
    assert [r.status for r in rec.role_reports] == ["ok", "ok", "ok"]


async def test_a_node_with_two_predecessors_waits_for_both_and_sees_both_reports(spans):
    _, rec = await run([(BI, ["DAU"]), (BI, ["GMV"]), (SCOUT, [])], [(f"{BI}@DAU", SCOUT), (f"{BI}@GMV", SCOUT)], Deps(bi=SlowBI()))
    c = spans[SCOUT]
    assert c[0] >= spans[f"{BI}@GMV"][1] - 1e-3 and c[0] >= spans[f"{BI}@DAU"][1] - 1e-3
    node = next(r for r in rec.role_reports if r.node_id == SCOUT)
    assert node.status == "ok" and {r.node_id for r in rec.role_reports if r.node_id != SCOUT} == {f"{BI}@DAU", f"{BI}@GMV"}


async def test_a_max_still_caps_concurrent_nodes(spans):
    await run([(BI, ["DAU"]), (BI, ["GMV"]), (BI, ["CTR"])], [], Deps(bi=SlowBI(slow=0.05)), a_max=2)
    assert spans["_active"]["peak"] == 2


async def test_a_failed_node_still_releases_its_dependents(spans):
    _, rec = await run([(BI, ["DAU"]), (SCOUT, [])], [(f"{BI}@DAU", SCOUT)], Deps(bi=SlowBI(boom={"DAU"})))
    assert [r.status for r in rec.role_reports] == ["failed", "ok"]


async def test_reports_keep_graph_node_order_regardless_of_finish_order():
    _, rec = await run([(BI, ["GMV"]), (BI, ["DAU"]), (BI, ["CTR"])], [], Deps(bi=SlowBI()))
    assert [r.node_id for r in rec.role_reports] == [f"{BI}@GMV", f"{BI}@DAU", f"{BI}@CTR"]


async def test_cycles_are_still_rejected():
    ctx = RunContext(query="DAU", deps=Deps())
    ctx.contract = await CoordinationStage().run(ctx)
    ctx.plan = await PlanningStage().run(ctx)
    g = WorkflowGraph(nodes=[make_node(REGISTRY[BI].spec, ["DAU"]), make_node(REGISTRY[BI].spec, ["GMV"])],
                      edges=[WorkflowEdge(src=f"{BI}@DAU", dst=f"{BI}@GMV", kind=EdgeKind.DEP), WorkflowEdge(src=f"{BI}@GMV", dst=f"{BI}@DAU", kind=EdgeKind.DEP)])
    ctx.plan = ctx.plan.model_copy(update={"chosen": ctx.plan.chosen.model_copy(update={"graph": g}), "known_assets": []})
    with pytest.raises(ContractValidationError):
        await asyncio.wait_for(RecallStage().run(ctx), timeout=5)


async def test_empty_graph_returns_immediately():
    _, rec = await run([], [], Deps())
    assert rec.role_reports == []
