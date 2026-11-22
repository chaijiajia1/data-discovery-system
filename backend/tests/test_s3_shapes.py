"""③ 候选图的结构灵活性：种子依赖边 / 按通道拆分或批量的组合 / 风格多样 / 按指标并行的流水线。"""
import pytest

from app.deps import Deps
from app.roles import list_role_specs
from app.schemas.workflow import EdgeKind
from app.stages.s3_planning.candidates import graph_style, generate_candidates, pick_top
from app.stages.s3_planning.context import build_plan_context
from app.stages.s3_planning.levels import grade_task
from app.stages.s3_planning.scorer import evaluate
from app.stages import RunContext
from app.stages.s2_coordination import CoordinationStage
from app.stages.s3_planning import PlanningStage
from tests.test_s3_planning import _contract, _pre, _plan

SPECS = {s.name: s for s in list_role_specs()}


def pc_for(metrics, dims=(), a_max=4, **kw):
    c = _contract(list(metrics), list(dims))
    return build_plan_context(c, _pre(), grade_task(c, _pre()), max_parallel=a_max, **kw)


def by_name(pc, **kw):
    return {c.name: c for c in generate_candidates(pc, SPECS, **kw)}


def recall_ids(c):
    return sorted(n.id for n in c.graph.nodes if n.phase == "recall")


def deps(c):
    return {(e.src, e.dst) for e in c.graph.edges if e.kind == EdgeKind.DEP}


# ── 种子依赖：血缘以 BI 召回为种子，所以 lineage 必须在对应 BI 之后 ────────────────
def test_lineage_depends_on_the_bi_node_covering_the_same_caliber():
    cands = by_name(pc_for(["DAU", "GMV"]))
    lanes = cands["lanes"]
    assert {("bi_asset_discovery@DAU", "lineage_discovery@DAU"), ("bi_asset_discovery@GMV", "lineage_discovery@GMV")} <= deps(lanes)
    assert ("bi_asset_discovery@DAU", "lineage_discovery@GMV") not in deps(lanes)               # 只依赖同口径的 BI，不跨口径互等
    batch = cands["per_channel"]
    assert ("bi_asset_discovery@DAU+GMV", "lineage_discovery@DAU+GMV") in deps(batch)


def test_batched_lineage_waits_for_a_split_bi_that_covers_any_of_its_calibers():
    c = by_name(pc_for(["DAU", "GMV"]))["split_bi"]                                              # BI 拆分，血缘 / 口径批量
    assert {("bi_asset_discovery@DAU", "lineage_discovery@DAU+GMV"), ("bi_asset_discovery@GMV", "lineage_discovery@DAU+GMV")} <= deps(c)


def test_seed_dependency_makes_the_batched_critical_path_longer_than_per_caliber_lanes():
    pc = pc_for(["DAU", "GMV"], ["渠道"])
    cands = by_name(pc)
    lanes, batch = cands["lanes"].score_breakdown if cands["lanes"].score_breakdown else None, None
    ev = {n: evaluate(c.graph, pc, name=n) for n, c in cands.items()}
    assert ev["per_channel"].score_breakdown["makespan"] > ev["lanes"].score_breakdown["makespan"]      # bi(2)→lineage(2) 比 bi(1)→lineage(1) 长
    assert ev["lanes"].score_breakdown["parallelism"] > ev["per_channel"].score_breakdown["parallelism"]
    assert ev["lanes"].score > ev["per_channel"].score                                                   # 默认权重下，按指标并行的流水线胜出


# ── 通道级的拆分 / 批量组合 ───────────────────────────────────────────────────────
def test_every_combination_of_split_or_batch_per_channel_is_generated_for_multi_caliber():
    names = set(by_name(pc_for(["DAU", "GMV"])))
    assert {"lanes", "per_channel", "split_bi", "split_graph", "split_knowledge", "split_bi_graph", "split_bi_knowledge", "split_graph_knowledge",
            "per_caliber", "single_agent"} == names


def test_split_means_one_node_per_caliber_and_batch_means_one_node_for_all():
    c = by_name(pc_for(["DAU", "GMV"]))["split_bi"]
    assert recall_ids(c) == ["bi_asset_discovery@DAU", "bi_asset_discovery@GMV", "caliber_lookup@DAU+GMV", "lineage_discovery@DAU+GMV"]


def test_single_caliber_collapses_split_and_batch_into_one_graph():
    cands = by_name(pc_for(["DAU"]))
    assert set(cands) == {"lanes", "per_caliber"} or set(cands) <= {"lanes", "per_caliber", "per_channel", "single_agent"}
    assert len({tuple(recall_ids(c)) for c in cands.values()}) == len(cands)                    # 去重后互不相同


def test_three_calibers_split_lanes_run_in_parallel_up_to_a_max():
    pc = pc_for(["DAU", "GMV", "CTR"], a_max=6)
    c = evaluate(by_name(pc)["lanes"].graph, pc, name="lanes")
    assert c.feasible and c.score_breakdown["agents"] == 9                                       # 3 口径 × (bi, lineage, caliber)；评估维度由这些召回角色提供，没有额外的选择节点
    assert c.score_breakdown["makespan"] == 3                                                    # lineage 角色同时最多 2 个副本：第三个血缘排到下一波


# ── 风格 / 多样性 ────────────────────────────────────────────────────────────────
def test_graph_style_classification():
    cands = by_name(pc_for(["DAU", "GMV"]))
    assert graph_style(cands["lanes"].graph) == "parallel" and graph_style(cands["per_channel"].graph) == "batched"
    assert graph_style(cands["split_bi"].graph) == "batched"                                      # 只要有一个节点处理多个口径就是 batched
    assert graph_style(cands["per_caliber"].graph) == "omni" and graph_style(cands["single_agent"].graph) == "batched"


def test_pick_top_guarantees_a_per_metric_parallel_graph_is_shown_even_if_batched_ones_score_higher():
    pc = pc_for(["DAU", "GMV"], a_max=4, w_coverage=0.6, w_parallel=0.05, w_cost=0.35)         # 重成本：批量 / omni 方案全面领先
    ev = [evaluate(c.graph, pc, name=c.name) for c in generate_candidates(pc, SPECS)]
    ranked = sorted(ev, key=lambda c: -c.score)
    assert graph_style(ranked[0].graph) != "parallel"
    top = pick_top(ranked, 3)
    assert len(top) == 3 and {graph_style(c.graph) for c in top} == {"parallel", "batched", "omni"}
    assert top[0].name == ranked[0].name                                                       # 但推荐（第一个）仍是分数最高的


def test_pick_top_fills_remaining_slots_by_score_and_never_returns_more_than_k():
    pc = pc_for(["DAU", "GMV"])
    ev = [evaluate(c.graph, pc, name=c.name) for c in generate_candidates(pc, SPECS)]
    ranked = sorted(ev, key=lambda c: -c.score)
    assert [c.name for c in pick_top(ranked, 5)][:1] == [ranked[0].name] and len(pick_top(ranked, 5)) == 5 and len(pick_top(ranked, 99)) == len(ranked)


# ── 阶段：你的查询 ───────────────────────────────────────────────────────────────
async def test_stage_recommends_per_metric_parallel_lanes_for_two_metrics(make_ctx):
    plan = await _plan(make_ctx, _contract(["DAU", "GMV"], ["channel"]), _pre(), max_parallel=4)
    best = plan.chosen
    assert graph_style(best.graph) in ("parallel", "omni") and len(plan.candidates) == 3                       # 推荐的是按指标拆分、指标之间并行的图
    assert not any(len(n.targets) > 1 for n in best.graph.nodes if n.phase == "recall")                       # 没有一个节点同时查两个指标
    assert {graph_style(c.graph) for c in plan.candidates} == {"parallel", "omni", "batched"}                  # 三种风格都能看到
    lanes = next(c for c in plan.candidates if c.name == "lanes")
    assert {n.id for n in lanes.graph.nodes if n.phase == "recall"} == {
        "bi_asset_discovery@DAU", "bi_asset_discovery@GMV", "lineage_discovery@DAU", "lineage_discovery@GMV", "caliber_lookup@DAU", "caliber_lookup@GMV"}


async def test_stage_for_the_query_in_the_issue_end_to_end():
    ctx = RunContext(query="Last 7 days DAU and GMV by channel", deps=Deps(), max_parallel=4)
    ctx.contract = await CoordinationStage().run(ctx)
    plan = await PlanningStage().run(ctx)
    assert ctx.contract.metrics.value == ["DAU", "GMV"]
    assert graph_style(plan.chosen.graph) in ("parallel", "omni") and any(graph_style(c.graph) == "batched" for c in plan.candidates)
    assert not any(len(n.targets) > 1 for n in plan.chosen.graph.nodes if n.phase == "recall")


async def test_k_limits_the_shown_candidates_but_the_best_is_always_first(make_ctx, deps):
    deps.settings = deps.settings.model_copy(update={"PLANNER_NUM_CANDIDATES": 1})
    plan = await _plan(make_ctx, _contract(["DAU", "GMV"]), _pre(), deps=deps)
    assert len([c for c in plan.candidates if c.source == "rule"]) == 1 and plan.chosen.name == plan.candidates[0].name
