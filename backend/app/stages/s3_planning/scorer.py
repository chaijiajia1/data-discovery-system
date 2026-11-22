"""候选图评分与择优。

    Score(G) = α·Cov(G) + β·Par(G) − γ·Cost(G)          三项都已归一，α+β+γ 的相对大小就是取舍偏好

    Cov   覆盖度 = (C₀ ∪ 图命中) 的「口径 × 通道」对数 / (口径数 × 通道数)
    Par   并行度 = (W − M) / (W − 1)
            W = 总工作量 = Σ 节点工作量（处理的「口径 × 通道」对数；选择类节点为 1）
            M = 完工时间 = ASAP 排程的关键路径长度
          全并行 M = 单个最长节点 → 接近 1；全串行 M = W → 0；单个节点 (W=1) 无并行可言，记 1；空图记 0
    Cost  Agent 消耗 = N / (|R′| + |D|)
            N = Agent 数（节点数）；分母是最细粒度拆法的 Agent 数（R′ 每对一个、D 每个维度一个）

默认 α=0.6, β=0.2, γ=0.2：覆盖优先；覆盖相同时，并行与成本分胜负。
批量节点（一个 Agent 带多个口径 / 多个通道）省 Agent 但时长变长，会同时拉高 Cost 项收益、拉低 Par 项——由权重决定是否划算。
分数只用于排序；硬约束在 feasibility.py。并列时取节点度数标准差更小的图。
"""
from __future__ import annotations

import statistics
from typing import Any

from ...errors import ContractValidationError
from ...schemas.workflow import CandidateWorkflow, PlanContext, WorkflowGraph
from .coverage import covered_pairs, node_duration, pending_keys, schedule
from .feasibility import check_feasibility

ROUND = 9


def score_graph(graph: WorkflowGraph, pc: PlanContext) -> dict[str, Any]:
    sched = schedule(graph, pc)                       # 有环抛 ContractValidationError
    N = len(graph.nodes)
    W = sum(node_duration(n, pc) for n in graph.nodes)
    M = max((s.end for s in sched), default=0)
    hit = covered_pairs(graph, pc)
    total = len(pc.units)
    c0 = sum(1 for u in pc.units if u.covered)
    cov = (c0 + len(hit)) / total if total else 1.0
    par = 0.0 if W <= 0 else (1.0 if W == 1 else (W - M) / (W - 1))
    n_ref = max(1, len(pending_keys(pc)) + len(pc.required_dims))
    cost = N / n_ref
    terms = {"coverage_term": pc.w_coverage * cov, "parallel_term": pc.w_parallel * par, "cost_term": pc.w_cost * cost}
    return {
        "score": round(terms["coverage_term"] + terms["parallel_term"] - terms["cost_term"], ROUND),
        "coverage": round(cov, ROUND), "parallelism": round(par, ROUND), "cost": round(cost, ROUND),
        "agents": N, "work": W, "makespan": M, "n_ref": n_ref,
        "covered_total": c0 + len(hit), "units_total": total,
        **{k: round(v, ROUND) for k, v in terms.items()},
        "schedule": sched, "covered_pairs": sorted(hit),
    }


def degree_std(graph: WorkflowGraph) -> float:
    """节点度数（入度 + 出度）的总体标准差：越小说明图越均衡（并列时的 tie-break）。"""
    if not graph.nodes:
        return 0.0
    deg = {n.id: 0 for n in graph.nodes}
    for e in graph.edges:
        for i in (e.src, e.dst):
            if i in deg:
                deg[i] += 1
    return statistics.pstdev(deg.values())


_BREAKDOWN = ("coverage", "parallelism", "cost", "agents", "work", "makespan", "n_ref", "covered_total", "units_total",
              "coverage_term", "parallel_term", "cost_term")


def evaluate(graph: WorkflowGraph, pc: PlanContext, *, name: str, source: str = "rule", rationale: str = "") -> CandidateWorkflow:
    violations = check_feasibility(graph, pc)
    try:
        s = score_graph(graph, pc)
    except ContractValidationError:
        return CandidateWorkflow(name=name, graph=graph, source=source, rationale=rationale, feasible=False, violations=violations,
                                 score_breakdown={"degree_std": 0.0})
    return CandidateWorkflow(
        name=name, graph=graph, source=source, rationale=rationale, score=s["score"], schedule=s["schedule"], covered_pairs=s["covered_pairs"],
        feasible=not violations, violations=violations,
        score_breakdown={k: float(s[k]) for k in _BREAKDOWN} | {"degree_std": round(degree_std(graph), ROUND)},
    )


def _key(c: CandidateWorkflow):
    return (-round(c.score, ROUND), round(degree_std(c.graph), ROUND), len(c.graph.nodes), c.name)


def pick_best(cands: list[CandidateWorkflow]) -> CandidateWorkflow | None:
    """可行候选里取分数最高者；并列取度数标准差更小者（再并列取节点更少、名字字典序）。无可行候选返回 None。"""
    feas = [c for c in cands if c.feasible]
    return min(feas, key=_key) if feas else None


def rank_candidates(cands: list[CandidateWorkflow]) -> list[CandidateWorkflow]:
    return sorted([c for c in cands if c.feasible], key=_key) + sorted([c for c in cands if not c.feasible], key=lambda c: (len(c.violations), c.name))
