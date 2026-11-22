"""可行性过滤：硬约束，不进入评分公式。违反任一条的候选图先被剔除（保留原因供展示）。

  - 峰值并发 ≤ A_max（来自前端用户输入）——按 ASAP 排程逐时间片检查，不是只看拓扑层
  - 各角色同时运行的节点数 ≤ 自身并发上限
  - 选择阶段所需的评估维度 D 必须都有节点提供（预检不能靠删节点省成本）
  - 图合法：角色已注册、无环

覆盖度（R′ 被覆盖多少）不是硬约束，而是评分里的最大化目标。
"""
from __future__ import annotations

from ...errors import ContractValidationError
from ...schemas.workflow import PlanContext, WorkflowGraph
from .coverage import asap, concurrency_violations


def check_feasibility(graph: WorkflowGraph, pc: PlanContext) -> list[str]:
    out: list[str] = []
    for r in sorted({n.role for n in graph.nodes if n.role not in pc.role_phase}):
        out.append(f"Role {r} is not registered")
    try:
        span = asap(graph, pc)
    except ContractValidationError as e:
        return out + [f"Invalid workflow graph: {e.message} (cycle or dangling edge)"]
    nodes = {n.id: n for n in graph.nodes}
    peak = max((len(ids) for _, kind, ids in concurrency_violations(span, nodes, pc) if kind == "peak"), default=0)
    if peak:
        out.append(f"Peak concurrency {peak} exceeds A_max={pc.max_parallel}")
    for _, kind, ids in concurrency_violations(span, nodes, pc):
        if kind != "peak":
            out.append(f"Role {kind} runs {len(ids)} nodes at once, above its concurrency limit {pc.role_limits[kind]}")
    provided = {d for n in graph.nodes for d in pc.role_dims.get(n.role, [])}
    for d in pc.required_dims:
        if d not in provided:
            out.append(f"No node provides the evaluation dimension {d}")
    return list(dict.fromkeys(out))
