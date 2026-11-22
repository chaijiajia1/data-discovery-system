"""PlanContext：覆盖单元 R / C₀ / R′、选择维度 D、角色通道与约束 / 权重参数。

    R   覆盖网格 = 口径 × 通道（bi / graph / knowledge），总数 = 口径数 × 通道数
    C₀  全有或全无，由任务的唯一入口级别决定：L0 / L1（入口定级本就跳过召回）→ 全部 口径×通道 都在 C₀；
        L2 → 没有 C₀——不默认知识已命中，预注入的资产只作血缘通道的种子
    R′  = R \\ C₀，待覆盖
    D   选择阶段仍需评估的维度（扣掉预注入已满足的）；必须都有节点提供（硬约束）
"""
from __future__ import annotations

from ...roles import RoleSpec, list_role_specs
from ...schemas.knowledge import PreInjectionResult
from ...schemas.task import TaskContract
from ...schemas.workflow import CoverageUnit, EntryLevel, GoalStatus, PlanContext
from ...roles.builtin import GRID_CHANNELS
from .levels import TaskGrade, caliber_state, metric_names

DIM_ORDER = ("metric_match", "caliber", "grain_fit", "dim_coverage", "freshness")      # permission 由 ⑤ 的确定性权限过滤负责，不要求节点提供


def build_plan_context(
    contract: TaskContract, pre: PreInjectionResult | None, grade: TaskGrade, *, max_parallel: int,
    w_coverage: float = 0.6, w_parallel: float = 0.2, w_cost: float = 0.2, specs: list[RoleSpec] | None = None,
) -> PlanContext:
    specs = specs or list_role_specs()
    domain = contract.intent.domain if contract.intent else None
    metrics = metric_names(contract)
    dims = [str(d) for d in (contract.dimensions.value or [])]
    level, goals = grade.level, grade.goals
    present = {c for sp in specs if sp.phase == "recall" for c in sp.channels}
    channels = [c for c in GRID_CHANNELS if c in present]

    units: list[CoverageUnit] = []
    skip_recall = level in (EntryLevel.L0, EntryLevel.L1)          # 任务只有一个级别：要么全部口径都跳过召回，要么全部都要真正去查
    for m in metrics:
        doubt = caliber_state(m, contract, pre) == "doubt"
        for ch in channels:
            ev = f"{level.value}：{grade.reason.split('（')[0]}" + ("（口径存疑，交由选择阶段核验）" if ch == "knowledge" and doubt else "") if skip_recall else ""
            units.append(CoverageUnit(caliber=m, channel=ch, covered=skip_recall, evidence=ev))

    needed = {"metric_match", "caliber", "grain_fit", "freshness"}
    if dims:
        needed.add("dim_coverage")
    satisfied: set[str] = set()
    if level != EntryLevel.L2:
        satisfied.add("metric_match")
    if level == EntryLevel.L0:                                  # 口径维度只在 L0 时由知识满足；L1 / L2 都要有节点去查 / 核验
        satisfied.add("caliber")
    dim_goals = [g for g in goals if g.kind == "dimension"]
    if dims and all(g.status == GoalStatus.COVERED for g in dim_goals):
        satisfied.add("dim_coverage")
    return PlanContext(
        metrics=metrics, dimensions=dims, domain=domain, units=units,
        required_dims=[d for d in DIM_ORDER if d in needed and d not in satisfied],
        satisfied_dims=[d for d in DIM_ORDER if d in needed and d in satisfied],
        channels=channels, max_parallel=max(1, int(max_parallel)), w_coverage=w_coverage, w_parallel=w_parallel, w_cost=w_cost,
        role_channels={s.name: sorted(s.channels) for s in specs}, role_dims={s.name: sorted(s.dims) for s in specs},
        role_limits={s.name: s.max_concurrency for s in specs}, role_phase={s.name: s.phase for s in specs},
    )
