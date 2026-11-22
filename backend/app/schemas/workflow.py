"""③ 动态工作流规划产物。"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from ..errors import ContractValidationError


class EdgeKind(str, Enum):
    THEN = "then"  # 顺序：dst 在 src 之后
    DEP = "dep"    # 依赖：dst 等待 src 的产出（可消费其结果）


class EntryLevel(str, Enum):
    """任务的入口级别（整个任务只有一个）：先按子目标（指标、维度）判覆盖状态，再取最差者。"""

    L0 = "L0"   # 全部子目标都已覆盖（会话确认的资产，或知识指明资产且口径一致）→ 不召回、不核验，直接进入执行
    L1 = "L1"   # 没有未覆盖的，但有口径存疑的 → 跳过召回，只进入选择（带佐证核验）
    L2 = "L2"   # 有子目标没被知识覆盖 → 完整召回 + 选择


class GoalStatus(str, Enum):
    """子目标（指标 / 维度）相对注入知识的覆盖状态。它不是级别——整个任务只有一个入口级别，由全部子目标的状态汇总得出。"""

    COVERED = "covered"        # 会话内已确认资产，或知识指明了所需资产且口径一致
    DOUBTFUL = "doubtful"      # 知识指明了资产，但口径存疑（冲突 / 不一致 / 缺口径证据）
    UNCOVERED = "uncovered"    # 知识没有指明所需的全部资产


class SubGoal(BaseModel):
    kind: Literal["metric", "dimension"]
    name: str
    status: GoalStatus
    reason: str
    rule_id: str = ""                                    # 判定依据，可追溯（铁律 4）
    asset_ids: list[str] = Field(default_factory=list)   # 知识 / 会话指明的资产


class CoverageUnit(BaseModel):
    """覆盖单元 = 口径 × 通道。covered=True 即属于 C₀（预注入已有证据）。"""

    caliber: str               # 指标口径
    channel: str               # 检索通道：bi | graph | knowledge
    covered: bool = False
    evidence: str = ""         # covered 时：证据来自哪里

    @property
    def key(self) -> str:
        return f"{self.caliber}×{self.channel}"


class WorkflowNode(BaseModel):
    id: str
    role: str                    # 角色名，对应 roles registry
    subtask: str                 # 该角色要探查的子任务描述
    params: dict = Field(default_factory=dict)
    phase: Literal["recall", "select"] = "recall"
    channels: list[str] = Field(default_factory=list)  # 该节点查询的通道（由角色决定；选择阶段角色为空）
    targets: list[str] = Field(default_factory=list)   # 该节点负责的口径（一个或多个）；空 = 全部


class WorkflowEdge(BaseModel):
    src: str
    dst: str
    kind: EdgeKind = EdgeKind.DEP


class WorkflowGraph(BaseModel):
    nodes: list[WorkflowNode]
    edges: list[WorkflowEdge] = Field(default_factory=list)

    def layers(self) -> list[list[str]]:
        """拓扑分层：同层节点互不依赖，可并行。有环则抛 ContractValidationError。"""
        ids = {n.id for n in self.nodes}
        indeg = {i: 0 for i in ids}
        out: dict[str, list[str]] = {i: [] for i in ids}
        for e in self.edges:
            if e.src not in ids or e.dst not in ids:
                raise ContractValidationError("s3_planning", f"edge references unknown node: {e.src}->{e.dst}")
            indeg[e.dst] += 1
            out[e.src].append(e.dst)
        layers: list[list[str]] = []
        frontier = sorted(i for i, d in indeg.items() if d == 0)
        seen = 0
        while frontier:
            layers.append(frontier)
            seen += len(frontier)
            nxt: list[str] = []
            for u in frontier:
                for v in out[u]:
                    indeg[v] -= 1
                    if indeg[v] == 0:
                        nxt.append(v)
            frontier = sorted(nxt)
        if seen != len(ids):
            raise ContractValidationError("s3_planning", "workflow graph has a cycle")
        return layers

    def max_width(self) -> int:
        return max((len(l) for l in self.layers()), default=0)

    def predecessors(self, node_id: str) -> list[str]:
        return [e.src for e in self.edges if e.dst == node_id]


class NodeSchedule(BaseModel):
    """ASAP 排程：节点按依赖尽早开始，工作量 = 处理的「口径×通道」对数（选择类节点为 1）。"""

    id: str
    start: int
    end: int
    duration: int
    covers: list[str] = Field(default_factory=list)   # 该节点新覆盖的 R′ 单元


class CandidateWorkflow(BaseModel):
    name: str
    graph: WorkflowGraph
    source: Literal["rule", "llm", "custom"] = "rule"
    score: float = 0.0
    # Score = α·Cov + β·Par − γ·Cost；各项（均已归一）与原始量都记录在这里，便于前端展示与回放
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    schedule: list[NodeSchedule] = Field(default_factory=list)
    covered_pairs: list[str] = Field(default_factory=list)              # 本图覆盖的 R′ 单元 key
    feasible: bool = True
    violations: list[str] = Field(default_factory=list)                 # 不可行原因（可行性过滤，不进入公式）
    rationale: str = ""


class PlanContext(BaseModel):
    """评分 / 可行性判定所需的全部输入。随 PlanningResult 返回，自定义图评分可据此复现。"""

    metrics: list[str] = Field(default_factory=list)                    # 口径（指标）
    dimensions: list[str] = Field(default_factory=list)
    domain: str | None = None
    channels: list[str] = Field(default_factory=list)                   # 通道集合：R = 口径 × 通道
    units: list[CoverageUnit] = Field(default_factory=list)             # R（covered=True 的构成 C₀）
    required_dims: list[str] = Field(default_factory=list)              # D：选择阶段仍需评估的维度（必须有节点提供）
    satisfied_dims: list[str] = Field(default_factory=list)             # 预注入已满足、不计入 D
    max_parallel: int = 4                                               # A_max（前端用户输入）
    w_coverage: float = 0.6                                             # α：覆盖度权重
    w_parallel: float = 0.2                                             # β：并行度权重
    w_cost: float = 0.2                                                 # γ：Agent 消耗权重
    role_channels: dict[str, list[str]] = Field(default_factory=dict)   # 角色 → 它查询的通道
    role_dims: dict[str, list[str]] = Field(default_factory=dict)       # 角色能提供的选择维度
    role_limits: dict[str, int] = Field(default_factory=dict)           # 各角色并发上限
    role_phase: dict[str, str] = Field(default_factory=dict)

    def pending_units(self) -> list[CoverageUnit]:
        """R′ = R \\ C₀"""
        return [u for u in self.units if not u.covered]


class PlanningResult(BaseModel):
    chosen: CandidateWorkflow
    candidates: list[CandidateWorkflow] = Field(default_factory=list)   # 含不可行者（feasible=False + violations）
    max_parallel: int = 4
    mode: Literal["execute", "select_only", "recall_and_select"] = "recall_and_select"
    level: EntryLevel = EntryLevel.L2                                   # 任务的唯一入口级别 = 全部子目标覆盖状态的汇总
    level_rule_id: str = ""
    level_reason: str = ""
    level_basis: list[str] = Field(default_factory=list)                # 决定该级别的子目标（如 L2 = 未覆盖的那些）
    sub_goals: list[SubGoal] = Field(default_factory=list)              # 各子目标的覆盖状态（级别的依据，不是各自的级别）
    known_assets: list[str] = Field(default_factory=list)               # L0/L1 子目标已知资产：④ 直接物化为候选
    context: PlanContext = Field(default_factory=PlanContext)
    recall_params: dict = Field(default_factory=dict)                   # 由 intent 映射的检索参数（见 services/recall_channels）
    degraded: bool = False                                              # 没有任何可行候选，退而取覆盖最高者
    llm_used: bool = False
    notes: list[str] = Field(default_factory=list)
