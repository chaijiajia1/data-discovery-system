"""③ 动态工作流规划：规划 Agent 接收任务契约与预注入覆盖情况，决定后续阶段如何执行。

    入口定级（子目标覆盖状态[LLM 判定，规则兜底] → 确定性汇总为唯一的任务级别 L0/L1/L2）→ 覆盖单元 R/C₀/R′ 与选择维度 D → 生成 K 个候选 AOV 图
    → 可行性过滤（A_max / 角色并发 / 选择维度 D 有提供者）→ 评分择优：Score = α·覆盖度 + β·并行度 − γ·Agent 消耗

mode：execute（全 L0，无 Agent 调用）/ select_only（无 L2，跳过召回）/ recall_and_select。
"""
from __future__ import annotations

from ...llm_client import NullLLM
from ...roles import list_role_specs
from ...schemas.workflow import CandidateWorkflow, EntryLevel, GoalStatus, PlanContext, PlanningResult, WorkflowGraph
from ...services.recall_channels import intent_to_params
from ..base import RunContext, Stage
from .agent import PlanningAgent
from .candidates import generate_candidates, pick_top
from .context import build_plan_context
from .level_agent import LevelAgent
from .levels import finalize_grade, rule_goals
from .scorer import evaluate, pick_best, rank_candidates

BLOCKED_DIM_VALUE_DOMAINS = {"resource_cost", "overseas"}      # 维值库仅覆盖三大业务域，这两域触发维值解析必 miss


MODE_BY_LEVEL = {EntryLevel.L0: "execute", EntryLevel.L1: "select_only", EntryLevel.L2: "recall_and_select"}


class PlanningStage(Stage[PlanningResult]):
    name = "stage3_planning"

    async def run(self, ctx: RunContext) -> PlanningResult:
        assert ctx.contract is not None, "planning requires contract"
        s, c = ctx.deps.settings, ctx.contract
        a_max = max(1, min(16, int(ctx.max_parallel or s.MAX_PARALLEL_AGENTS)))
        notes: list[str] = []
        llm_used = False
        goals = rule_goals(c, ctx.preinjection, confirmed=ctx.confirmed_assets, require_bi=s.PLANNER_REQUIRE_BI_ASSET)
        if s.PLANNER_LLM_LEVEL_ENABLED:
            # 子目标覆盖状态由 LLM 判定；级别仍由确定性汇总。LLM 不可用 / 输出不可解析 → 回落规则（铁律 3）
            llm_goals = await LevelAgent(ctx.deps.llm).regrade(c, ctx.preinjection, goals, confirmed=ctx.confirmed_assets, require_bi=s.PLANNER_REQUIRE_BI_ASSET)
            if llm_goals is None:
                if not isinstance(ctx.deps.llm, NullLLM):
                    notes.append("Entry level: LLM grading failed; fell back to rules")
            else:
                goals, llm_used = llm_goals, True
                notes.append("Entry level: sub-goal coverage graded by the LLM; the task level is aggregated deterministically")
        grade = finalize_grade(goals)
        goals, known = grade.goals, grade.known_assets
        specs = list_role_specs()
        by_name = {sp.name: sp for sp in specs}
        pc = build_plan_context(c, ctx.preinjection, grade, max_parallel=a_max, w_coverage=s.PLANNER_W_COVERAGE, w_parallel=s.PLANNER_W_PARALLEL, w_cost=s.PLANNER_W_COST, specs=specs)
        mode = MODE_BY_LEVEL[grade.level]
        domain = c.intent.domain if c.intent else None
        dim_resolve = bool(c.intent and c.intent.dim_terms and domain not in BLOCKED_DIM_VALUE_DOMAINS
                           and any(g.kind == "dimension" and g.status == GoalStatus.UNCOVERED for g in goals))
        if mode == "execute":
            # 全 L0：没有 Agent 调用，选择阶段的评估维度（粒度 / 新鲜度 / 权限）由 ⑤ 的确定性硬过滤与排序完成，不要求节点提供
            pc = pc.model_copy(update={"satisfied_dims": pc.satisfied_dims + pc.required_dims, "required_dims": []})
            cands = [evaluate(WorkflowGraph(nodes=[]), pc, name="direct_execute", rationale="Entry level L0: no recall, no verification, straight to execution (Phases 4-5 only materialise known assets and apply the permission filter)")]
        else:
            allc = [evaluate(r.graph, pc, name=r.name, rationale=r.rationale) for r in generate_candidates(pc, by_name, dim_resolve=dim_resolve)]
            cands = pick_top(rank_candidates(allc), s.PLANNER_NUM_CANDIDATES)
            rules = cands
            extra = await PlanningAgent(ctx.deps.llm).propose(pc, by_name, limit=s.PLANNER_LLM_EXTRA, taken_names={r.name for r in rules})
            llm_used = llm_used or bool(extra)
            cands += [evaluate(x.graph, pc, name=x.name, source="llm", rationale=x.rationale) for x in extra]
        ranked = rank_candidates(cands)
        chosen = pick_best(cands)
        degraded = chosen is None
        if degraded:
            chosen = ranked[0]
            notes.append("No candidate satisfied every feasibility constraint; fell back to the one with the fewest violations: " + "; ".join(chosen.violations))
        return PlanningResult(
            chosen=chosen, candidates=ranked, max_parallel=a_max, mode=mode, level=grade.level, level_rule_id=grade.rule_id, level_reason=grade.reason, level_basis=grade.basis, sub_goals=goals, known_assets=known, context=pc,  # type: ignore[arg-type]
            recall_params=intent_to_params(c.intent) if c.intent else {}, degraded=degraded, llm_used=llm_used, notes=notes,
        )


def evaluate_custom(pc: PlanContext, graph: WorkflowGraph) -> CandidateWorkflow:
    """用户在前端自定义的图：用同一套可行性与评分评估。"""
    return evaluate(graph, pc, name="custom", source="custom", rationale="User-defined workflow")
