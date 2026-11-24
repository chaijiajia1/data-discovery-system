"""④ 并行召回：按 ③ 的工作流图多角色并行探查 → 共享黑板协作 → 证据合并去重（冲突显式保留）。

混合检索：BI 通道（真实可用的基础资产）与知识通道（向量 + 图谱，带出血缘关联的资产 / 指标 / 维度实体）归一化后进同一个候选池。
协作：证据卡复用优先 → 认领等待 → 定向询问 / 黑板广播（被点名角色可婉拒并降级）→ 采纳 / 忽略 / 婉拒写回能力画像。

L0 / L1 子目标跳过召回：③ 给出的已知资产经 `deps.bi.get` 物化为候选（found_by=preinjection），
作为各节点的 seed 上游（L1 时召回角色只核验它们）；全 L0 时图为空，不产生任何 Agent 调用。
"""
from __future__ import annotations

from ...schemas.knowledge import PreInjectionResult
from ...schemas.recall import RecallResult, RoleReport
from ..base import RunContext, Stage
from .merger import merge
from .scheduler import run_workflow


def views_for_hub(pre: PreInjectionResult | None) -> dict:
    """① 分发给 ④ 各角色（及内联核对）的知识视图：consumer（recall.*）→ 知识。没有分发视图 → 空（各角色回落全量）。"""
    return {c: pre.view(c) for c in (pre.views if pre else {}) if c.startswith("recall.")}


class RecallStage(Stage[RecallResult]):
    name = "stage4_recall"

    async def run(self, ctx: RunContext) -> RecallResult:
        assert ctx.contract is not None and ctx.plan is not None, "recall requires contract & plan"
        knowledge = ctx.preinjection.injected if ctx.preinjection else []
        seed = None
        if ctx.plan.known_assets:
            known = await ctx.deps.bi.get(list(ctx.plan.known_assets))
            seed = RoleReport(node_id="seed", role="preinjection", candidates=[c.model_copy(update={"found_by": []}) for c in known])
        reports, hub = await run_workflow(
            ctx.plan.chosen.graph, contract=ctx.contract, knowledge=knowledge, deps=ctx.deps, max_parallel=ctx.plan.max_parallel,
            seed=seed, cookie=ctx.cookie, recall_params=ctx.plan.recall_params, known_assets=ctx.plan.known_assets, mode=ctx.plan.mode,
            views=views_for_hub(ctx.preinjection),
        )
        result = merge(([seed] if seed else []) + reports, knowledge, board=hub.board, role_of=hub.role_of)
        result.role_reports = reports            # seed 不是角色，不进角色报告
        result.board = hub.board.snapshot()
        result.collab = hub.finalize()
        return result
