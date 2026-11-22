"""覆盖与排程。

覆盖单元 = 口径 × 通道。节点查询一个或多个口径（targets，空 = 全部口径），经角色的一个或多个通道；
命中 R′ 里对应的 (口径, 通道) 对。单元只要被任意节点命中就算覆盖（0 / 1），重复命中不再增加覆盖，只增加成本。

排程：节点按依赖尽早开始（ASAP）。节点工作量 = 处理的「口径 × 通道」对数（不在覆盖网格内的节点、以及没有待覆盖对时只做核验的节点为 1）。
所以「一个超大节点」不会被误当成并行：它的时长就是它内部要串行处理的对数。
"""
from __future__ import annotations

from ...schemas.workflow import NodeSchedule, PlanContext, WorkflowGraph, WorkflowNode


def unit_key(caliber: str, channel: str) -> str:
    return f"{caliber}×{channel}"


def pending_keys(pc: PlanContext) -> list[str]:
    return [u.key for u in pc.pending_units()]


def _grid_channels(node: WorkflowNode, pc: PlanContext) -> list[str]:
    if pc.role_phase.get(node.role) != "recall":
        return []
    return [c for c in pc.role_channels.get(node.role, []) if c in pc.channels]


def node_pairs(node: WorkflowNode, pc: PlanContext) -> set[str]:
    """节点命中的 R′ 单元。"""
    pending = set(pending_keys(pc))
    return {k for m in (node.targets or pc.metrics) for ch in _grid_channels(node, pc) if (k := unit_key(m, ch)) in pending}


def node_duration(node: WorkflowNode, pc: PlanContext) -> int:
    ch = _grid_channels(node, pc)
    if not ch or not pending_keys(pc):                # 没有待覆盖的对（L0 / L1）：召回角色只做核验，工作量为 1
        return 1
    return max(1, len(node.targets or pc.metrics) * len(ch))


def asap(graph: WorkflowGraph, pc: PlanContext) -> dict[str, tuple[int, int]]:
    """id → (start, end)。有环抛 ContractValidationError。"""
    nodes = {n.id: n for n in graph.nodes}
    span: dict[str, tuple[int, int]] = {}
    for layer in graph.layers():
        for nid in layer:
            s = max((span[p][1] for p in graph.predecessors(nid)), default=0)
            span[nid] = (s, s + node_duration(nodes[nid], pc))
    return span


def schedule(graph: WorkflowGraph, pc: PlanContext) -> list[NodeSchedule]:
    span = asap(graph, pc)
    nodes = {n.id: n for n in graph.nodes}
    seen: set[str] = set()
    out: list[NodeSchedule] = []
    for nid in sorted(span, key=lambda i: (span[i][0], i)):
        new = node_pairs(nodes[nid], pc) - seen
        seen |= new
        s, e = span[nid]
        out.append(NodeSchedule(id=nid, start=s, end=e, duration=e - s, covers=sorted(new)))
    return out


def covered_pairs(graph: WorkflowGraph, pc: PlanContext) -> set[str]:
    hit: set[str] = set()
    for n in graph.nodes:
        hit |= node_pairs(n, pc)
    return hit


def concurrency_violations(span: dict[str, tuple[int, int]], nodes: dict[str, WorkflowNode], pc: PlanContext) -> list[tuple[int, str, list[str]]]:
    """按排程逐个时间片检查：(t, 'peak'|角色名, 同时运行的节点 id，按 (结束时间, id) 排序)。"""
    out: list[tuple[int, str, list[str]]] = []
    horizon = max((e for _, e in span.values()), default=0)
    for t in range(horizon):
        active = sorted((i for i, (s, e) in span.items() if s <= t < e), key=lambda i: (span[i][1], i))
        if len(active) > pc.max_parallel:
            out.append((t, "peak", active))
        by_role: dict[str, list[str]] = {}
        for i in active:
            by_role.setdefault(nodes[i].role, []).append(i)
        for role, ids in by_role.items():
            lim = pc.role_limits.get(role)
            if lim is not None and len(ids) > lim:
                out.append((t, role, ids))
    return out
