"""候选图生成：在约束下生成结构不同的候选 AOV 图 G=(V,E,A)。

  V  子任务节点：(角色, 口径集合)。一个节点可带一个或多个口径；角色决定它查一个还是多个通道
  E  依赖边：dep（等待上游产出）/ then（仅顺序，用于并发修复）
  A  每个节点关联一个角色（见 roles/builtin.py）

**种子依赖**：血缘通道以已有召回内容（BI 通道的资产）为种子，所以 `lineage` 节点必须在覆盖其任一口径的 `bi` 节点之后（dep 边）。
规划图如实包含这条边，批量节点的关键路径才会如实变长——否则「一个节点串行查多个口径」看起来没有代价。

**拆分 / 批量**：对每个通道独立选择——拆分（每个待覆盖口径一个节点，口径之间并行）或批量（一个节点串行处理全部口径，Agent 更少）。
3 个通道 × 2 种选择 = 8 种组合，再加两种全通道 Agent（`omni`）方案：
  lanes          全部拆分：每个口径一条并行流水线（bi@M → lineage@M，caliber@M 并行），口径之间互不等待
  per_channel    全部批量：每个通道一个 Agent 处理全部口径
  split_<通道…>  只拆分所列通道（如 split_bi：BI 按口径并行，血缘 / 口径批量）
  per_caliber    每个口径一个全通道 Agent（依次查 bi → graph → knowledge）
  single_agent   一个全通道 Agent 处理全部口径：Agent 最少，完全串行
评估维度 D 由召回角色内联提供（原选择类角色已并入）；图里还缺的维度用最小集合覆盖补节点——L1 没有待覆盖的对时，整张图就由这些只核验、不召回的节点组成。
全部生成并评分后，由 `pick_top` 取 top K：第一个始终是分数最高的，并保证 parallel / omni / batched 三种风格各有最优代表（口径按节点拆分的方案不会被批量方案挤出候选）。
并发修复：按 ASAP 排程逐时间片检查，超过 A_max 或角色并发上限时，把溢出节点用 then 边排到其后。
"""
from __future__ import annotations

from ...roles import RoleSpec
from ...schemas.workflow import CandidateWorkflow, EdgeKind, PlanContext, WorkflowEdge, WorkflowGraph, WorkflowNode
from .coverage import asap, concurrency_violations

DIMVAL = "dim_value_resolver"


def node_id(role: str, targets: list[str]) -> str:
    return f"{role}@{'+'.join(targets)}" if targets else role


def make_node(spec: RoleSpec, targets: list[str]) -> WorkflowNode:
    sub = spec.description.split("：")[0] + (f"（{'、'.join(targets)}）" if targets else "")
    return WorkflowNode(id=node_id(spec.name, targets), role=spec.name, subtask=sub, phase=spec.phase,  # type: ignore[arg-type]
                        channels=sorted(spec.channels), targets=list(targets))


def repair_width(graph: WorkflowGraph, pc: PlanContext) -> WorkflowGraph:
    """按 ASAP 排程逐时间片修复超限：把溢出节点用 then 边排到「同时运行的最早结束者」之后，直到收敛。"""
    nodes = {n.id: n for n in graph.nodes}
    for _ in range(len(graph.nodes) ** 2 + 8):
        viol = concurrency_violations(asap(graph, pc), nodes, pc)
        if not viol:
            break
        _, kind, active = viol[0]
        limit = max(1, pc.max_parallel if kind == "peak" else pc.role_limits[kind])
        graph.edges.append(WorkflowEdge(src=active[limit - 1], dst=active[limit], kind=EdgeKind.THEN))
    return graph


def _signature(g: WorkflowGraph) -> tuple:
    return (tuple(sorted((n.id, tuple(n.targets)) for n in g.nodes)), tuple(sorted((e.src, e.dst, e.kind.value) for e in g.edges)))


class _Builder:
    def __init__(self, pc: PlanContext, specs: dict[str, RoleSpec], dim_resolve: bool):
        self.pc, self.specs, self.dim_resolve = pc, specs, dim_resolve
        pend: dict[str, list[str]] = {ch: [] for ch in pc.channels}          # 通道 → 该通道上待覆盖的口径
        for u in pc.pending_units():
            pend[u.channel].append(u.caliber)
        self.pend = pend
        self.calibers = [m for m in pc.metrics if any(m in v for v in pend.values())]
        recall = [s for s in specs.values() if s.phase == "recall"]
        self.role_of = {ch: s.name for ch in pc.channels for s in recall if set(s.channels) == {ch}}
        self.omni = next((s.name for s in recall if set(pc.channels) <= set(s.channels) and len(s.channels) > 1), None)
        self.active = [ch for ch in pc.channels if self.role_of.get(ch) and pend[ch]]

    def extra_roles(self, provided: set[str], used: set[str]) -> list[str]:
        """图里还没人提供的评估维度，用最小集合覆盖补节点（L1 没有待覆盖的对时，整张图就由它们组成：只核验、不召回）。"""
        sel = sorted((s for s in self.specs.values() if s.name not in used and s.name != DIMVAL), key=lambda s: s.name)
        remaining = {d for d in self.pc.required_dims if d not in provided}
        chosen: list[str] = []
        while remaining:
            best = max((s for s in sel if s.name not in chosen), key=lambda s: (len(set(s.dims) & remaining), -len(s.dims)), default=None)
            if not best or not (set(best.dims) & remaining):
                break
            chosen.append(best.name)
            remaining -= set(best.dims)
        return chosen

    def recall_nodes(self, split: set[str] | str) -> list[tuple[str, list[str]]]:
        """(角色, 口径列表)。split：要拆分的通道集合，或 "omni_split" / "omni_batch"。"""
        if split in ("omni_split", "omni_batch"):
            if not self.omni:
                return []
            return [(self.omni, [m]) for m in self.calibers] if split == "omni_split" else ([(self.omni, list(self.calibers))] if self.calibers else [])
        out: list[tuple[str, list[str]]] = []
        for ch in self.active:
            cals = self.pend[ch]
            out += [(self.role_of[ch], [c]) for c in cals] if ch in split else [(self.role_of[ch], list(cals))]
        return out

    def build(self, split: set[str] | str) -> WorkflowGraph | None:
        recall = self.recall_nodes(split)
        if self.calibers and not recall:
            return None                                   # 该方案在当前上下文里不可用（如没有全通道角色）
        if self.dim_resolve and not isinstance(split, str) and DIMVAL in self.specs:
            recall.append((DIMVAL, []))
        provided = {d for r, _ in recall for d in self.specs[r].dims}
        sel = self.extra_roles(provided, {r for r, _ in recall})
        nodes = [make_node(self.specs[r], t) for r, t in recall] + [make_node(self.specs[r], list(self.pc.metrics) if self.specs[r].phase == "recall" else []) for r in sel]
        if not nodes:
            return None
        ids = {n.id for n in nodes}
        edges: list[WorkflowEdge] = []

        def dep(a: str, b: str):
            if a in ids and b in ids and a != b and not any(e.src == a and e.dst == b for e in edges):
                edges.append(WorkflowEdge(src=a, dst=b, kind=EdgeKind.DEP))

        bi_role, ln_role = self.role_of.get("bi"), self.role_of.get("graph")
        for r, t in recall:                               # 种子依赖：血缘节点等待覆盖其任一口径的 BI 节点
            if r == ln_role:
                for r2, t2 in recall:
                    if r2 == bi_role and set(t) & set(t2):
                        dep(node_id(r2, t2), node_id(r, t))
        return WorkflowGraph(nodes=nodes, edges=edges)


def graph_style(g: WorkflowGraph) -> str:
    """batched：存在一个节点处理多个口径（节点内串行）；omni：每个口径各用一个全通道 Agent（口径之间并行，节点内串行）；
    parallel：每个口径、每个通道各自一个节点（口径之间、通道之间都并行）。"""
    recall = [n for n in g.nodes if n.phase == "recall" and n.role != DIMVAL]
    if any(len(n.targets) > 1 for n in recall):
        return "batched"
    return "omni" if any(len(n.channels) > 1 for n in recall) else "parallel"


def _combo_name(split: set[str], channels: list[str]) -> str:
    if len(split) == len(channels):
        return "lanes"
    if not split:
        return "per_channel"
    return "split_" + "_".join(ch for ch in channels if ch in split)


_RATIONALE = {
    "lanes": "每个口径一条并行流水线（bi@M → lineage@M，caliber@M 并行）：口径之间互不等待，Agent 较多",
    "per_channel": "每个通道一个 Agent，批量处理全部口径：Agent 少，但节点内串行处理多个口径，血缘要等整批 BI 结果",
    "per_caliber": "每个口径一个全通道 Agent（依次查 bi → graph → knowledge）：口径之间并行，节点内串行",
    "single_agent": "一个全通道 Agent 处理全部口径：Agent 最少，完全串行",
}


def generate_candidates(pc: PlanContext, specs: dict[str, RoleSpec], k: int | None = None, *, dim_resolve: bool = False) -> list[CandidateWorkflow]:
    """生成全部结构不同的候选（去重）。k 给定时只取前 k 个（生成顺序，不看分数；通常传 None，由 pick_top 在评分后挑）。"""
    b = _Builder(pc, specs, dim_resolve)
    plans: list[tuple[str, set[str] | str]] = []
    multi = len(b.calibers) > 1
    chans = b.active
    if chans:
        for mask in range(2 ** len(chans) - 1, -1, -1):          # 全拆分在前，全批量在后
            split = {ch for i, ch in enumerate(chans) if mask >> i & 1}
            if not multi and split:                              # 单口径：拆分 == 批量，只留一个
                if len(split) != len(chans):
                    continue
            plans.append((_combo_name(split, chans), split))
    plans += [("per_caliber", "omni_split"), ("single_agent", "omni_batch")]
    out: list[CandidateWorkflow] = []
    seen: set[tuple] = set()
    for name, split in plans:
        g = b.build(split)
        if g is None:
            continue
        repair_width(g, pc)
        sig = _signature(g)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(CandidateWorkflow(name=name, graph=g, rationale=_RATIONALE.get(name, f"仅拆分 {name.removeprefix('split_')} 通道（口径之间并行），其余通道批量处理")))
        if k is not None and len(out) >= k:
            break
    return out


def pick_top(ranked: list[CandidateWorkflow], k: int) -> list[CandidateWorkflow]:
    """从已按优劣排序的候选里取 k 个：第一个始终是最优；然后让 parallel / omni / batched 三种风格各有最优代表；余下按分数补满。"""
    if k >= len(ranked):
        return list(ranked)
    feasible = [c for c in ranked if c.feasible]
    pool = feasible if len(feasible) >= k else ranked
    top: list[CandidateWorkflow] = [pool[0]] if pool else []
    seen = {graph_style(pool[0].graph)} if pool else set()
    for c in pool[1:]:
        st = graph_style(c.graph)
        if st not in seen and len(top) < k:
            top.append(c)
            seen.add(st)
    for c in pool:
        if len(top) >= k:
            break
        if c not in top:
            top.append(c)
    return sorted(top, key=lambda c: pool.index(c))
