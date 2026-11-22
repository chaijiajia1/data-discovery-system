from app.deps import Deps
from app.services.fakes import FakeEngine
from app.stages.s6_execution.attribution import attribute
from app.schemas.execution import CheckResult, VerificationReport
from app.pipeline import stream_pipeline
from app.stages import RunContext


async def _run(deps, query="最近7天DAU"):
    ctx = RunContext(query=query, deps=deps)
    events = [e async for e in stream_pipeline(ctx)]
    return ctx, events


async def test_success_writes_verified_pair():
    deps = Deps()
    ctx, events = await _run(deps)
    assert events[-1][0] == "done" and events[-1][1]["status"] == "success"
    assert ctx.execution.feedback_written
    assert [p.pair_id for p in deps.pair_store.all()] == [ctx.execution.verified_pair.pair_id]


async def test_failure_rolls_back_to_min_step_and_retains_upstream():
    deps = Deps(engine=FakeEngine(fail_at={("run_query", 1): "timeout"}))
    ctx, _ = await _run(deps)
    ex = ctx.execution
    assert ex.status == "success" and len(ex.attempts) == 2
    a1 = ex.attempts[0]
    assert a1.attribution.suspect_step == "run_query" and a1.attribution.rollback_to == "run_query"
    assert a1.attribution.retained_steps == ["resolve_fields", "build_query"]
    # 第二轮只重跑 run_query：上游步骤不重复执行
    assert deps.engine._calls == {"resolve_fields": 1, "build_query": 1, "run_query": 2}


async def test_persistent_failure_gives_up_without_writing_pair():
    deps = Deps(engine=FakeEngine(fail_at={("run_query", n): "boom" for n in range(1, 10)}))
    deps.settings = deps.settings.model_copy(update={"EXEC_MAX_ATTEMPTS": 5})
    ctx, events = await _run(deps)
    assert events[-1][1]["status"] == "failed"
    assert deps.pair_store.all() == []


def test_schema_mismatch_switches_asset():
    rep = VerificationReport(checks=[CheckResult(name="schema_match", passed=False, detail="x", step_id="resolve_fields")])
    a = attribute(step_order=["resolve_fields", "build_query", "run_query"], report=rep, step_failures={}, has_fallback_asset=True)
    assert a.next_action == "switch_asset" and a.retained_steps == []


async def test_multi_primary_all_executed_and_written_back():
    deps = Deps()
    ctx, events = await _run(deps, "最近30天各渠道的DAU和GMV")
    assert events[-1][1]["status"] == "success"
    assert {a.asset_id for a in ctx.execution.attempts} == {"dws.user_active_d", "dws.trade_gmv_d"}
    assert len(deps.pair_store.all()) == 2


async def test_preinjected_metric_def_becomes_caliber_evidence():
    deps = Deps()
    ctx, _ = await _run(deps, "最近7天DAU")
    assert any(e.kind.value == "caliber" and e.asset_id == "dws.user_active_d" for e in ctx.recall.evidences)      # 口径证据来自口径查找（预注入知识的同一条定义去重后保留先产出者）
    assert ctx.selection.risk_flags == []
