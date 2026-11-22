"""入口定级：整个任务只有一个级别 L0 / L1 / L2。

两步：
  ① 按子目标（指标、维度）分别判**覆盖状态**——covered / doubtful / uncovered（这不是级别，只是依据）；
     指标：会话内已确认资产 → covered；知识没指明任何资产 / 缺所需的某类资产 → uncovered；
           指明了所需资产且口径一致 → covered；口径冲突 / 不一致 / 缺口径证据 → doubtful
     维度：知识把它绑定到了资产字段 → covered；只有定义没绑定 → doubtful；没覆盖 → uncovered
  ② 汇总成唯一的任务级别（取最差者）：
     L2  有任一子目标 uncovered                → 完整召回 + 选择（对全部口径，不为已覆盖的口径开例外）
     L1  没有 uncovered，但有 doubtful          → 跳过召回，只进入选择（带佐证核验）
     L0  全部 covered                          → 直接进入执行

「所需资产」按资产类型衡量：指标至少要一个 BI 资产（数据集 / 图表 / 看板）；需要拆维度或做分析时还要明细表。
每个定级都带 rule_id，可追溯（铁律 4）。知识缺失资产类型时优先判 L2：跳过召回会让缺的那类永远补不上。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ...schemas.knowledge import KnowledgeItem, KnowledgeKind, PreInjectionResult
from ...schemas.task import TaskContract, ValueSource
from ...schemas.workflow import EntryLevel, GoalStatus, SubGoal

BI_PREFIXES = ("dataset:", "chart:", "dashboard:", "metric:")
TABLE_PREFIXES = ("hive_table:", "table:")


def asset_kind(asset_id: str) -> str:
    """资产 id → 覆盖单元的资产类型。"""
    a = (asset_id or "").lower()
    if a.startswith(BI_PREFIXES):
        return "bi_dataset"
    if a.startswith(TABLE_PREFIXES) or "." in a:
        return "detail_table"
    return "bi_dataset"


def mentions(item: KnowledgeItem, name: str) -> bool:
    n = name.strip().casefold()
    if not n:
        return False
    if n == item.title.strip().casefold() or n in {t.strip().casefold() for t in item.terms}:
        return True
    return len(n) >= 2 and n in f"{item.title} {item.content}".casefold()


def metric_names(contract: TaskContract) -> list[str]:
    ms = [str(m) for m in (contract.metrics.value or [])]
    return ms or [contract.query]            # 没有指标：整句话作为唯一子目标


def required_asset_types(contract: TaskContract, *, require_bi: bool = False) -> list[str]:
    """知识至少要指明哪些类型的资产才算「已指明资产」。默认不限类型：知识指明了任何数据源资产（BI 资产或明细表）就算——
    明细表（hive_table）本来就是可用的数据源，BI 资产优先只影响 ⑤ 选哪个，不决定要不要召回。`require_bi` 才要求必须有 BI 资产。"""
    return ["bi_dataset"] if require_bi else []


def _injected(pre: PreInjectionResult | None) -> list[KnowledgeItem]:
    return list(pre.injected) if pre else []


def known_assets_for(metric: str, pre: PreInjectionResult | None) -> list[str]:
    out: list[str] = []
    for it in _injected(pre):
        if it.kind != KnowledgeKind.DIMENSION and mentions(it, metric):
            out += [a for a in it.asset_ids if a not in out]
    return out


def caliber_state(metric: str, contract: TaskContract, pre: PreInjectionResult | None) -> str:
    """ok：有一致的口径证据（或用户明确指定）；doubt：口径冲突 / 不一致；none：没有口径证据。"""
    if contract.caliber.source == ValueSource.USER and contract.caliber.value:
        return "ok"
    n = metric.strip().casefold()
    defs = [it for it in _injected(pre) if it.kind == KnowledgeKind.METRIC_DEF and mentions(it, metric)]
    conflict = any(d.reason == "conflict" and n and (n in d.title.casefold() or d.title.casefold() in n) for d in (pre.dropped if pre else []))
    if conflict or any(it.consistency == "possible_inconsistent" for it in _injected(pre) if mentions(it, metric)):
        return "doubt"
    return "ok" if defs else "none"


def _goal(kind: str, name: str, status: GoalStatus, rule_id: str, reason: str, assets: list[str] | None = None) -> SubGoal:
    return SubGoal(kind=kind, name=name, status=status, rule_id=rule_id, reason=reason, asset_ids=list(assets or []))  # type: ignore[arg-type]


def _grade_metric(metric: str, contract: TaskContract, pre: PreInjectionResult | None, confirmed: dict[str, list[str]], require_bi: bool = False) -> SubGoal:
    sess = [a for a in (confirmed or {}).get(metric, []) if a]
    if sess:
        return _goal("metric", metric, GoalStatus.COVERED, "goal.covered.session_confirmed", f"会话内已确认资产 {sess}", sess)
    known = known_assets_for(metric, pre)
    if not known:
        return _goal("metric", metric, GoalStatus.UNCOVERED, "goal.uncovered.no_knowledge", "注入知识没有指明任何资产")
    have = {asset_kind(a) for a in known}
    missing = [t for t in required_asset_types(contract, require_bi=require_bi) if t not in have]
    if missing:
        return _goal("metric", metric, GoalStatus.UNCOVERED, "goal.uncovered.incomplete_assets", f"注入知识指明了 {known}，但缺少资产类型 {missing}", known)
    state = caliber_state(metric, contract, pre)
    if state == "ok":
        return _goal("metric", metric, GoalStatus.COVERED, "goal.covered.knowledge_assets", f"注入知识指明了所需资产 {known}，口径一致", known)
    why = "口径冲突或不一致" if state == "doubt" else "没有口径证据"
    return _goal("metric", metric, GoalStatus.DOUBTFUL, "goal.doubtful.caliber_doubt" if state == "doubt" else "goal.doubtful.no_caliber_evidence",
                 f"注入知识指明了资产 {known}，但{why}", known)


def _grade_dimension(dim: str, pre: PreInjectionResult | None) -> SubGoal:
    hits = [it for it in _injected(pre) if it.kind in (KnowledgeKind.DIMENSION, KnowledgeKind.ASSET_META) and mentions(it, dim)]
    bound = [a for it in hits for a in it.asset_ids]
    if bound:
        return _goal("dimension", dim, GoalStatus.COVERED, "goal.covered.dim_bound", "注入知识已把该维度绑定到资产字段", list(dict.fromkeys(bound)))
    if hits:
        return _goal("dimension", dim, GoalStatus.DOUBTFUL, "goal.doubtful.dim_unbound", "注入知识给出了维度定义，但没有绑定资产")
    return _goal("dimension", dim, GoalStatus.UNCOVERED, "goal.uncovered.dim_unknown", "注入知识没有覆盖该维度")


@dataclass
class TaskGrade:
    goals: list[SubGoal]
    level: EntryLevel
    rule_id: str
    reason: str
    basis: list[str]
    known_assets: list[str] = field(default_factory=list)    # L0 / L1 才有：知识 / 会话指明的资产（④ 直接物化为候选，不再召回）


def assess_level(goals: list[SubGoal]) -> tuple[EntryLevel, str, str, list[str]]:
    """子目标覆盖状态 → 唯一的任务级别（取最差者）。返回 (级别, rule_id, 原因, 决定该级别的子目标)。"""
    names = lambda st: [g.name for g in goals if g.status == st]
    unc, dbt = names(GoalStatus.UNCOVERED), names(GoalStatus.DOUBTFUL)
    if unc:
        why = "；".join(f"{g.name}：{g.reason}" for g in goals if g.status == GoalStatus.UNCOVERED)
        return EntryLevel.L2, "level.l2.uncovered_goals", f"{'、'.join(unc)} 没有被注入知识覆盖（{why}）→ 完整召回 + 选择", unc
    if dbt:
        why = "；".join(f"{g.name}：{g.reason}" for g in goals if g.status == GoalStatus.DOUBTFUL)
        return EntryLevel.L1, "level.l1.doubtful_goals", f"{'、'.join(dbt)} 口径存疑（{why}）→ 跳过召回，只进入选择（带佐证核验）", dbt
    metrics = [g for g in goals if g.kind == "metric"]
    sess = bool(metrics) and all(g.rule_id == "goal.covered.session_confirmed" for g in metrics)
    return (EntryLevel.L0, "level.l0.session_confirmed" if sess else "level.l0.all_covered",
            "会话内已确认全部指标的资产，直接进入执行" if sess else "全部子目标都已被注入知识覆盖且口径一致 → 直接进入执行", [g.name for g in goals])


def rule_goals(contract: TaskContract, pre: PreInjectionResult | None, *, confirmed: dict[str, list[str]] | None = None, require_bi: bool = False) -> list[SubGoal]:
    """规则版子目标覆盖状态（LLM 定级的兜底，也是 LLM 输出非法时逐目标的回落值）。"""
    goals = [_grade_metric(m, contract, pre, confirmed or {}, require_bi) for m in metric_names(contract)]
    goals += [_grade_dimension(str(d), pre) for d in (contract.dimensions.value or [])]
    return goals


def finalize_grade(goals: list[SubGoal]) -> TaskGrade:
    """子目标覆盖状态（无论来自规则还是 LLM）→ 唯一的任务级别 + 已知资产。级别始终由确定性汇总得出。"""
    level, rule_id, reason, basis = assess_level(goals)
    known: list[str] = []
    if level != EntryLevel.L2:                  # L2 要对全部口径完整召回；已知资产只作血缘通道的种子（经 env.knowledge / env.known_assets 传给 ④）
        for g in goals:
            if g.kind == "metric":
                known += [a for a in g.asset_ids if a not in known]
    return TaskGrade(goals=goals, level=level, rule_id=rule_id, reason=reason, basis=basis, known_assets=known)


def grade_task(contract: TaskContract, pre: PreInjectionResult | None, *, confirmed: dict[str, list[str]] | None = None, require_bi: bool = False) -> TaskGrade:
    return finalize_grade(rule_goals(contract, pre, confirmed=confirmed, require_bi=require_bi))
