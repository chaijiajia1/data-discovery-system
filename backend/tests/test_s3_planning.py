"""③ 动态工作流规划：入口定级 / 覆盖单元 / 评分公式 / 可行性过滤 / 候选图 / LLM 提案 / 阶段。"""
import json

import pytest

from app.deps import Deps
from app.errors import ContractValidationError, UpstreamError
from app.schemas.intent import Intent
from app.schemas.knowledge import DroppedKnowledge, KnowledgeItem, KnowledgeKind, KnowledgeOrigin, PreInjectionResult
from app.schemas.task import Slot, TaskContract, ValueSource
from app.schemas.workflow import (
    CandidateWorkflow, CoverageUnit, EdgeKind, EntryLevel, GoalStatus, PlanContext, WorkflowEdge, WorkflowGraph, WorkflowNode,
)
from app.stages.s3_planning import PlanningStage
from app.stages.s3_planning.candidates import node_id, repair_width
from app.stages.s3_planning.context import build_plan_context
from app.stages.s3_planning.feasibility import check_feasibility
from app.stages.s3_planning.levels import asset_kind, grade_task
from app.stages.s3_planning.scorer import degree_std, evaluate, pick_best, score_graph


# ── 构造辅助 ────────────────────────────────────────────────────────────────
def _contract(metrics, dims=None, *, intent_type="data_fetch", domain="transaction", warnings=(), caliber=None, dim_terms=()):
    return TaskContract(
        query="q", metrics=Slot(value=list(metrics), source=ValueSource.USER), dimensions=Slot(value=list(dims or []), source=ValueSource.USER),
        caliber=Slot(value=caliber, source=ValueSource.USER if caliber else ValueSource.DEFAULT_RULE),
        intent=Intent(query="q", intent_type=intent_type, domain=domain, sub_domain=["交易驾驶舱"], warnings=list(warnings), dim_terms=list(dim_terms)),
    )


def _k(title, kind=KnowledgeKind.METRIC_DEF, assets=(), consistency="none", content=None, i=None):
    return KnowledgeItem(id=i or f"k_{title}_{kind.value}", kind=kind, title=title, content=content or f"{title} 的定义", origin=KnowledgeOrigin.RETRIEVAL,
                         asset_ids=list(assets), consistency=consistency, relevance=0.9, token_cost=10)


def _pre(*items, dropped=()):
    return PreInjectionResult(injected=list(items), dropped=list(dropped))


def _g(nodes, edges=()):
    """nodes: [(id, role, targets)]；edges: [(src, dst)] 均为 DEP。"""
    return WorkflowGraph(
        nodes=[WorkflowNode(id=i, role=r, subtask="", targets=list(t)) for i, r, t in nodes],
        edges=[WorkflowEdge(src=s, dst=d, kind=EdgeKind.DEP) for s, d in edges],
    )


# ── 入口定级 ────────────────────────────────────────────────────────────────
def test_asset_kind_by_id():
    assert asset_kind("dataset:1") == "bi_dataset" and asset_kind("chart:9") == "bi_dataset"
    assert asset_kind("hive_table:dws.x") == "detail_table" and asset_kind("dws.user_active_d") == "detail_table"


def G(c, pre=None, **kw):
    return grade_task(c, pre if pre is not None else _pre(), **kw)


def st(grade, name):
    return next(g for g in grade.goals if g.name == name)


# ── 子目标：只判覆盖状态（不是级别）──────────────────────────────────────────────
def test_metric_covered_when_knowledge_names_assets_with_consistent_caliber():
    g = G(_contract(["DAU"]), _pre(_k("DAU", assets=["dataset:1"])))
    s = st(g, "DAU")
    assert (s.kind, s.status, s.rule_id, s.asset_ids) == ("metric", GoalStatus.COVERED, "goal.covered.knowledge_assets", ["dataset:1"])
    assert not hasattr(s, "level")                                                              # 子目标没有自己的级别


def test_metric_covered_when_session_already_confirmed_asset():
    g = G(_contract(["DAU"]), confirmed={"DAU": ["chart:9"]})
    assert st(g, "DAU").status == GoalStatus.COVERED and st(g, "DAU").rule_id == "goal.covered.session_confirmed" and g.known_assets == ["chart:9"]
    assert g.level == EntryLevel.L0 and g.rule_id == "level.l0.session_confirmed"


def test_metric_doubtful_when_caliber_is_possible_inconsistent():
    g = G(_contract(["DAU"]), _pre(_k("DAU", assets=["dataset:1"], consistency="possible_inconsistent")))
    assert st(g, "DAU").status == GoalStatus.DOUBTFUL and st(g, "DAU").rule_id == "goal.doubtful.caliber_doubt"
    assert g.level == EntryLevel.L1 and g.known_assets == ["dataset:1"]


def test_metric_doubtful_when_conflicting_definitions_were_dropped():
    pre = _pre(_k("DAU", assets=["dataset:1"]), dropped=[DroppedKnowledge(item_id="x", title="DAU", reason="conflict")])
    assert st(G(_contract(["DAU"]), pre), "DAU").status == GoalStatus.DOUBTFUL


def test_metric_doubtful_when_assets_known_but_no_caliber_evidence():
    g = G(_contract(["DAU"]), _pre(_k("DAU", KnowledgeKind.PAIR, assets=["dataset:1"], content="历史相似问题 DAU 选用资产 dataset:1")))
    assert st(g, "DAU").status == GoalStatus.DOUBTFUL and st(g, "DAU").rule_id == "goal.doubtful.no_caliber_evidence"


def test_user_given_caliber_counts_as_consistent():
    g = G(_contract(["DAU"], caliber="去重口径"), _pre(_k("DAU", KnowledgeKind.PAIR, assets=["dataset:1"])))
    assert st(g, "DAU").status == GoalStatus.COVERED and g.level == EntryLevel.L0


def test_a_table_named_by_knowledge_is_enough_by_default():
    """知识指明了数据源资产就算指明了资产——明细表（hive_table）也是数据源，BI 资产优先只影响 ⑤ 选哪个，不决定要不要召回。"""
    g = G(_contract(["DAU"], ["渠道"]), _pre(_k("DAU", assets=["dws.user_active_d"])))
    s = st(g, "DAU")
    assert s.status == GoalStatus.COVERED and s.rule_id == "goal.covered.knowledge_assets" and s.asset_ids == ["dws.user_active_d"]


def test_requiring_a_bi_asset_is_an_opt_in_and_a_table_only_hint_then_stays_uncovered():
    c, pre = _contract(["DAU"], ["渠道"]), _pre(_k("DAU", assets=["dws.user_active_d"]))
    g = grade_task(c, pre, require_bi=True)
    s = st(g, "DAU")
    assert s.status == GoalStatus.UNCOVERED and s.rule_id == "goal.uncovered.incomplete_assets" and "bi_dataset" in s.reason
    assert st(grade_task(c, _pre(_k("DAU", assets=["dataset:1"])), require_bi=True), "DAU").status == GoalStatus.COVERED


def test_demo_knowledge_with_table_assets_and_an_unbound_dimension_is_l1_not_a_full_recall():
    """DAU / GMV 的知识只指到 dws 表、维度「渠道」只有定义没绑定资产：以前整个任务被判 L2（完整召回），现在是 L1（跳过召回，只核验）。"""
    pre = _pre(_k("DAU", assets=["dws.user_active_d"]), _k("GMV", assets=["dws.trade_gmv_d"]), _k("渠道", KnowledgeKind.DIMENSION))
    g = G(_contract(["DAU", "GMV"], ["渠道"]), pre)
    assert {s.name: s.status for s in g.goals} == {"DAU": GoalStatus.COVERED, "GMV": GoalStatus.COVERED, "渠道": GoalStatus.DOUBTFUL}
    assert g.level == EntryLevel.L1 and sorted(g.known_assets) == ["dws.trade_gmv_d", "dws.user_active_d"]


def test_metric_uncovered_when_nothing_known():
    g = G(_contract(["DAU"]))
    assert st(g, "DAU").status == GoalStatus.UNCOVERED and st(g, "DAU").rule_id == "goal.uncovered.no_knowledge" and g.known_assets == []


def test_dimension_statuses():
    c = _contract(["DAU"], ["渠道", "城市", "平台"])
    pre = _pre(_k("DAU", assets=["dataset:1", "dws.t"]), _k("渠道", KnowledgeKind.DIMENSION, assets=["dataset:1"]), _k("城市", KnowledgeKind.DIMENSION))
    g = G(c, pre)
    assert {s.name: s.status for s in g.goals if s.kind == "dimension"} == {"渠道": GoalStatus.COVERED, "城市": GoalStatus.DOUBTFUL, "平台": GoalStatus.UNCOVERED}


def test_empty_metrics_falls_back_to_whole_query():
    g = G(_contract([]))
    assert g.goals[0].name == "q" and g.goals[0].status == GoalStatus.UNCOVERED and g.level == EntryLevel.L2


# ── 汇总：整个任务只有一个级别（取最差者）─────────────────────────────────────────
def test_single_level_is_the_worst_sub_goal_status():
    pre = _pre(_k("DAU", assets=["dataset:1"]))
    g = G(_contract(["DAU", "GMV"]), pre)                                                      # DAU 已覆盖，GMV 没有
    assert {s.name: s.status for s in g.goals} == {"DAU": GoalStatus.COVERED, "GMV": GoalStatus.UNCOVERED}
    assert g.level == EntryLevel.L2 and g.rule_id == "level.l2.uncovered_goals" and g.basis == ["GMV"] and "GMV" in g.reason and "DAU" not in g.basis


def test_all_covered_is_l0_and_names_every_goal_as_basis():
    g = G(_contract(["DAU", "GMV"]), _pre(_k("DAU", assets=["dataset:1"]), _k("GMV", assets=["dataset:2"])))
    assert g.level == EntryLevel.L0 and g.rule_id == "level.l0.all_covered" and set(g.basis) == {"DAU", "GMV"}
    assert g.known_assets == ["dataset:1", "dataset:2"]


def test_doubtful_without_uncovered_is_l1_and_uncovered_beats_doubtful():
    doubt = _k("DAU", assets=["dataset:1"], consistency="possible_inconsistent")
    assert G(_contract(["DAU"]), _pre(doubt)).level == EntryLevel.L1
    both = G(_contract(["DAU", "GMV"]), _pre(doubt))                                           # DAU 存疑、GMV 未覆盖
    assert both.level == EntryLevel.L2 and both.basis == ["GMV"]


def test_dimension_participates_in_the_level():
    c = _contract(["DAU"], ["渠道"])
    pre = _pre(_k("DAU", assets=["dataset:1", "dws.t"]))                                       # 指标已覆盖，维度「渠道」知识没覆盖
    g = G(c, pre)
    assert st(g, "DAU").status == GoalStatus.COVERED and st(g, "渠道").status == GoalStatus.UNCOVERED and g.level == EntryLevel.L2 and g.basis == ["渠道"]
    bound = G(c, _pre(_k("DAU", assets=["dataset:1", "dws.t"]), _k("渠道", KnowledgeKind.DIMENSION, assets=["dataset:1"])))
    assert bound.level == EntryLevel.L0


def test_known_assets_only_materialised_when_recall_is_skipped():
    pre = _pre(_k("DAU", assets=["dataset:1"]))
    assert G(_contract(["DAU"]), pre).known_assets == ["dataset:1"]                           # L0：直接物化
    assert G(_contract(["DAU", "GMV"]), pre).known_assets == []                               # L2：完整召回，已知资产只作血缘种子


# ── 覆盖单元 R = 口径 × 通道 / C₀ / R′ 与选择维度 D ───────────────────────────────
def _ctx_for(c, pre, **kw):
    return build_plan_context(c, pre, grade_task(c, pre), max_parallel=kw.pop("max_parallel", 4), **kw)


def test_grid_is_calibers_times_channels():
    pc = _ctx_for(_contract(["DAU", "GMV"]), _pre())
    assert pc.channels == ["bi", "graph", "knowledge"] and len(pc.units) == 2 * 3
    assert [u.key for u in pc.units] == ["DAU×bi", "DAU×graph", "DAU×knowledge", "GMV×bi", "GMV×graph", "GMV×knowledge"]
    assert all(not u.covered for u in pc.units) and len(pc.pending_units()) == 6
    assert pc.required_dims == ["metric_match", "caliber", "grain_fit", "freshness"] and pc.satisfied_dims == []          # permission 不在 D：由 ⑤ 的权限过滤负责


def test_knowledge_hits_never_cover_a_channel_for_an_l2_caliber():
    """不默认知识已命中：L2 口径的每个 口径×通道 都要真正去查，预注入的资产只作后续召回的种子。"""
    pre = _pre(_k("DAU", assets=["dws.user_active_d"]))
    c = _contract(["DAU", "GMV"], ["渠道"])
    assert G(c, pre).level == EntryLevel.L2
    pc = _ctx_for(c, pre)
    assert not any(u.covered for u in pc.units) and len(pc.pending_units()) == 6


def test_l2_caliber_with_knowledge_naming_assets_and_consistent_caliber_still_needs_all_three_channels():
    pre = _pre(_k("DAU", assets=["dws.user_active_d"]))                      # 指标已覆盖，但维度「渠道」知识没覆盖 → L2
    c = _contract(["DAU"], ["渠道"])
    assert G(c, pre).level == EntryLevel.L2
    pc = _ctx_for(c, pre)
    assert {u.key: u.covered for u in pc.units} == {"DAU×bi": False, "DAU×graph": False, "DAU×knowledge": False}
    assert "caliber" in pc.required_dims and "metric_match" in pc.required_dims          # 口径维度也不再被知识抵扣


def test_l0_and_l1_calibers_have_all_channels_covered():
    l0 = _ctx_for(_contract(["DAU"]), _pre(_k("DAU", assets=["dataset:1"])))
    assert l0.pending_units() == [] and "caliber" in l0.satisfied_dims and "metric_match" in l0.satisfied_dims
    assert l0.required_dims == ["grain_fit", "freshness"]
    l1 = _ctx_for(_contract(["DAU"]), _pre(_k("DAU", assets=["dataset:1"], consistency="possible_inconsistent")))
    assert l1.pending_units() == [] and "metric_match" in l1.satisfied_dims
    assert l1.required_dims == ["caliber", "grain_fit", "freshness"]          # 口径存疑：仍需口径证据，由口径角色（或全通道角色）提供


def test_dim_coverage_required_only_when_dimensions_not_all_l0():
    base = _pre(_k("DAU", assets=["dataset:1", "dws.t"]))
    assert "dim_coverage" in _ctx_for(_contract(["DAU"], ["渠道"]), base).required_dims
    ok = _pre(_k("DAU", assets=["dataset:1", "dws.t"]), _k("渠道", KnowledgeKind.DIMENSION, assets=["dataset:1"]))
    pc = _ctx_for(_contract(["DAU"], ["渠道"]), ok)
    assert "dim_coverage" not in pc.required_dims and "dim_coverage" in pc.satisfied_dims
    assert "dim_coverage" not in _ctx_for(_contract(["DAU"]), base).required_dims


def test_context_carries_role_channels_limits_phases_and_weights():
    pc = _ctx_for(_contract(["DAU"]), _pre(), w_coverage=0.5, w_parallel=0.3, w_cost=0.1)
    assert pc.role_channels["bi_asset_discovery"] == ["bi"] and pc.role_channels["lineage_discovery"] == ["graph"]
    assert pc.role_channels["caliber_lookup"] == ["knowledge"] and pc.role_channels["omni_discovery"] == ["bi", "graph", "knowledge"]
    assert pc.role_channels["dim_value_resolver"] == ["dim_value"] and pc.role_limits["lineage_discovery"] == 2
    assert set(pc.role_phase.values()) == {"recall"} and "freshness_precheck" not in pc.role_phase          # 选择类角色已并入召回角色
    assert (pc.w_coverage, pc.w_parallel, pc.w_cost) == (0.5, 0.3, 0.1)
    assert not hasattr(pc, "lam") and not hasattr(pc, "theta") and not hasattr(pc, "high_risk")


# ── 评分：Score = α·Cov + β·Par − γ·Cost ─────────────────────────────────────
def _pc(units, dims=(), *, a_max=8, w=(0.6, 0.2, 0.2), channels=("bi", "graph", "knowledge")):
    from app.roles import list_role_specs
    specs = list_role_specs()
    return PlanContext(
        metrics=list(dict.fromkeys(u.caliber for u in units)), units=list(units), required_dims=list(dims), max_parallel=a_max, channels=list(channels),
        w_coverage=w[0], w_parallel=w[1], w_cost=w[2],
        role_channels={s.name: sorted(s.channels) for s in specs}, role_dims={s.name: sorted(s.dims) for s in specs},
        role_limits={s.name: s.max_concurrency for s in specs}, role_phase={s.name: s.phase for s in specs},
    )


def _grid(calibers, *, covered=()):
    return [CoverageUnit(caliber=m, channel=ch, covered=f"{m}×{ch}" in covered) for m in calibers for ch in ("bi", "graph", "knowledge")]


FAN = [("bi@M", "bi_asset_discovery", ["M"]), ("ln@M", "lineage_discovery", ["M"]), ("cal@M", "caliber_lookup", ["M"])]


def test_score_fanout_hand_computed():
    s = score_graph(_g(FAN), _pc(_grid(["M"])))
    # 3 个单通道节点并行：Cov=3/3，W=3, M=1 → Par=(3−1)/(3−1)=1，N=3, N_ref=|R′|=3 → Cost=1
    assert (s["coverage"], s["parallelism"], s["cost"]) == (1.0, 1.0, 1.0) and (s["agents"], s["work"], s["makespan"]) == (3, 3, 1)
    assert s["score"] == pytest.approx(0.6 * 1 + 0.2 * 1 - 0.2 * 1)


def test_serial_chain_loses_parallelism_only():
    serial = _g(FAN, [("bi@M", "ln@M"), ("ln@M", "cal@M")])
    s = score_graph(serial, _pc(_grid(["M"])))
    assert s["makespan"] == 3 and s["parallelism"] == 0.0 and s["coverage"] == 1.0
    assert s["score"] == pytest.approx(0.6 - 0.2) and s["score"] < score_graph(_g(FAN), _pc(_grid(["M"])))["score"]


def test_omni_node_is_one_agent_but_its_work_is_serial_inside():
    s = score_graph(_g([("o@M", "omni_discovery", ["M"])]), _pc(_grid(["M"])))
    assert (s["agents"], s["work"], s["makespan"]) == (1, 3, 3)                      # 一个 Agent 查 3 个通道：时长 3
    assert s["coverage"] == 1.0 and s["parallelism"] == 0.0 and s["cost"] == pytest.approx(1 / 3)
    assert s["score"] == pytest.approx(0.6 + 0 - 0.2 / 3)


def test_partial_coverage_is_scored_not_forbidden():
    s = score_graph(_g([FAN[0]]), _pc(_grid(["M"])))
    assert s["coverage"] == pytest.approx(1 / 3) and s["covered_pairs"] == ["M×bi"]
    assert s["parallelism"] == 1.0                                                    # 单个节点：无并行可言，不扣分
    assert s["score"] == pytest.approx(0.6 / 3 + 0.2 - 0.2 / 3)
    assert score_graph(_g(FAN), _pc(_grid(["M"])))["score"] > s["score"]             # 覆盖更全的赢


def test_redundant_agent_adds_cost_only():
    dup = _g(FAN + [("bi2@M", "bi_asset_discovery", ["M"])])
    base, d = score_graph(_g(FAN), _pc(_grid(["M"]))), score_graph(dup, _pc(_grid(["M"])))
    assert d["coverage"] == base["coverage"] and d["parallelism"] == base["parallelism"] == 1.0
    assert d["score"] - base["score"] == pytest.approx(-0.2 / 3)                     # N 多 1，N_ref 不变


def test_batching_calibers_beats_fanout_when_agents_matter():
    units = _grid(["A", "B"])
    batched = _g([("bi@A+B", "bi_asset_discovery", ["A", "B"]), ("ln@A+B", "lineage_discovery", ["A", "B"]), ("cal@A+B", "caliber_lookup", ["A", "B"])])
    fan = _g([(f"{r}@{m}", r, [m]) for m in "AB" for r in ("bi_asset_discovery", "lineage_discovery", "caliber_lookup")])
    sb, sf = score_graph(batched, _pc(units)), score_graph(fan, _pc(units))
    assert (sb["agents"], sb["work"], sb["makespan"]) == (3, 6, 2) and sb["parallelism"] == pytest.approx((6 - 2) / 5)
    assert (sf["agents"], sf["work"], sf["makespan"]) == (6, 6, 1) and sf["parallelism"] == 1.0
    assert sb["score"] == pytest.approx(0.6 + 0.2 * 0.8 - 0.2 * 0.5) and sf["score"] == pytest.approx(0.6 + 0.2 - 0.2 * 1.0)
    assert sb["score"] > sf["score"]                                                  # 省 3 个 Agent 的收益 > 少 0.2 的并行度
    assert sb["score"] < score_graph(batched, _pc(units, w=(0.6, 0.2, 0.0)))["score"]


def test_weights_move_the_optimum():
    units = _grid(["M"])
    parallel, omni = _g(FAN), _g([("o@M", "omni_discovery", ["M"])])
    fast = _pc(units, w=(0.4, 0.5, 0.1))
    cheap = _pc(units, w=(0.4, 0.05, 0.55))
    assert score_graph(parallel, fast)["score"] > score_graph(omni, fast)["score"]            # 重并行：多 Agent 并行胜
    assert score_graph(omni, cheap)["score"] > score_graph(parallel, cheap)["score"]          # 重成本：单 Agent 胜


def test_dependent_recall_nodes_count_in_work_makespan_and_cost():
    g = _g([("bi@M", "bi_asset_discovery", ["M"]), ("ln@M", "lineage_discovery", ["M"])], [("bi@M", "ln@M")])
    s = score_graph(g, _pc([CoverageUnit(caliber="M", channel="bi"), CoverageUnit(caliber="M", channel="graph")], ["freshness", "dim_coverage"]))
    assert (s["agents"], s["work"], s["makespan"]) == (2, 2, 2) and s["parallelism"] == 0.0
    assert s["n_ref"] == 2 + 2 and s["cost"] == pytest.approx(2 / 4)
    assert s["score"] == pytest.approx(0.6 - 0.2 * 2 / 4)


def test_recall_nodes_with_nothing_pending_only_verify_so_their_work_is_one():
    g = _g([("o@M", "omni_discovery", ["M", "N"])])
    s = score_graph(g, _pc(_grid(["M", "N"], covered=tuple(f"{m}×{c}" for m in "MN" for c in ("bi", "graph", "knowledge"))), ["caliber"]))
    assert s["work"] == 1 and s["makespan"] == 1


def test_c0_counts_toward_coverage_but_not_toward_gain():
    units = _grid(["M"], covered=("M×bi",))
    s = score_graph(_g([FAN[1], FAN[2]]), _pc(units))
    assert s["coverage"] == 1.0 and s["covered_total"] == 3 and s["covered_pairs"] == ["M×graph", "M×knowledge"]
    assert score_graph(WorkflowGraph(nodes=[]), _pc(units))["coverage"] == pytest.approx(1 / 3)           # 空图也有 C₀ 的覆盖


def test_node_only_covers_its_own_targets_and_channels():
    units = _grid(["A", "B"])
    s = score_graph(_g([("bi@A", "bi_asset_discovery", ["A"])]), _pc(units))
    assert s["covered_pairs"] == ["A×bi"] and s["coverage"] == pytest.approx(1 / 6)
    s = score_graph(_g([("bi@all", "bi_asset_discovery", [])]), _pc(units))                              # targets 为空 = 全部口径
    assert s["covered_pairs"] == ["A×bi", "B×bi"]
    assert score_graph(_g([("d", "dim_value_resolver", [])]), _pc(units))["covered_pairs"] == []         # 不在覆盖网格内的通道


def test_empty_graph_and_degenerate_denominators():
    assert score_graph(WorkflowGraph(nodes=[]), _pc([]))["score"] == pytest.approx(0.6)                  # 无单元：覆盖度记 1
    s = score_graph(_g([("f", "dim_value_resolver", [])]), _pc([]))                                      # R′ 与 D 都空：N_ref 取 1
    assert s["cost"] == 1.0 and s["parallelism"] == 1.0


def test_schedule_is_asap_and_attributes_new_pairs():
    g = _g([("bi@M", "bi_asset_discovery", ["M"]), ("o@M", "omni_discovery", ["M"]), ("dv", "dim_value_resolver", [])], [("bi@M", "dv"), ("o@M", "dv")])
    sched = {x.id: x for x in score_graph(g, _pc(_grid(["M"]), ["freshness"]))["schedule"]}
    assert (sched["bi@M"].start, sched["bi@M"].end) == (0, 1) and (sched["o@M"].start, sched["o@M"].end) == (0, 3)
    assert (sched["dv"].start, sched["dv"].end) == (3, 4)                                                # 等最慢的上游
    assert sched["bi@M"].covers == ["M×bi"] and sched["o@M"].covers == ["M×graph", "M×knowledge"]       # bi 先到先得，omni 只记新增的


# ── 可行性过滤（不进入公式）────────────────────────────────────────────────
def test_feasible_graph_has_no_violations():
    assert check_feasibility(_g(FAN), _pc(_grid(["M"]))) == []


def test_peak_concurrency_uses_the_schedule_not_just_layers():
    assert any("Peak concurrency" in v for v in check_feasibility(_g(FAN), _pc(_grid(["M"]), a_max=2)))
    # 拓扑上分两层，但按排程 a(时长 3) 仍在运行时 c 已开始：同时运行 3 个 → 超限
    g = _g([("a", "omni_discovery", ["M"]), ("b", "dim_value_resolver", []), ("c", "dim_value_resolver", [])], [("b", "c")])
    assert check_feasibility(g, _pc(_grid(["M"]), a_max=1))
    ok = check_feasibility(_g([("a", "omni_discovery", ["M"]), ("b", "dim_value_resolver", [])], [("a", "b")]), _pc(_grid(["M"]), a_max=1))
    assert ok == []


def test_role_concurrency_limit_is_infeasible():
    units = _grid(["A", "B", "C"])
    g = _g([(f"l@{m}", "lineage_discovery", [m]) for m in "ABC"])
    v = check_feasibility(g, _pc(units))                                               # lineage 同时运行上限 2
    assert any("lineage_discovery" in x and "concurrency limit" in x for x in v)


def test_required_dims_must_have_a_provider_among_the_recall_nodes():
    g = _g([FAN[1], FAN[2]])                                                              # 血缘 + 口径：没有 BI 角色 → 没人提供 grain_fit / freshness
    v = check_feasibility(g, _pc(_grid(["M"]), ["metric_match", "caliber", "grain_fit", "freshness"]))
    assert v == ["No node provides the evaluation dimension grain_fit", "No node provides the evaluation dimension freshness"]
    assert check_feasibility(_g(FAN), _pc(_grid(["M"]), ["metric_match", "caliber", "grain_fit", "freshness", "dim_coverage"])) == []   # bi + lineage + caliber 齐了


def test_low_coverage_graph_is_still_feasible():
    c = evaluate(_g([FAN[0]]), _pc(_grid(["M"])), name="thin")
    assert c.feasible and c.violations == [] and c.score_breakdown["coverage"] == pytest.approx(1 / 3)


def test_unknown_role_and_cycle_are_infeasible_not_crashing():
    assert any("not registered" in v for v in check_feasibility(_g([("x", "ghost", [])]), _pc([])))
    cyc = _g([("a", "bi_asset_discovery", ["M"]), ("b", "caliber_lookup", ["M"])], [("a", "b"), ("b", "a")])
    assert any("cycle" in v for v in check_feasibility(cyc, _pc(_grid(["M"]))))
    assert evaluate(cyc, _pc(_grid(["M"])), name="cyc").feasible is False


# ── 并列取度数标准差更小者 ─────────────────────────────────────────────────
def test_degree_std():
    star = _g([(x, "bi_asset_discovery", []) for x in "abcd"], [("a", "b"), ("a", "c"), ("a", "d")])
    chain = _g([(x, "bi_asset_discovery", []) for x in "abcd"], [("a", "b"), ("b", "c"), ("c", "d")])
    assert degree_std(chain) == pytest.approx(0.5) and degree_std(star) == pytest.approx(0.8660254, rel=1e-5)
    assert degree_std(WorkflowGraph(nodes=[])) == 0.0


def test_pick_best_prefers_higher_score_then_smaller_degree_std():
    star = CandidateWorkflow(name="star", score=0.5, graph=_g([(x, "bi_asset_discovery", []) for x in "abcd"], [("a", "b"), ("a", "c"), ("a", "d")]))
    chain = CandidateWorkflow(name="chain", score=0.5, graph=_g([(x, "bi_asset_discovery", []) for x in "abcd"], [("a", "b"), ("b", "c"), ("c", "d")]))
    assert pick_best([star, chain]).name == "chain" and pick_best([chain, star]).name == "chain"
    better = star.model_copy(update={"name": "better", "score": 0.51})
    assert pick_best([chain, better]).name == "better"
    assert pick_best([chain.model_copy(update={"feasible": False, "score": 9}), star]).name == "star"
    assert pick_best([chain.model_copy(update={"feasible": False})]) is None


# ── 候选图生成 / 并发修复 ───────────────────────────────────────────────────
def test_node_id_convention():
    assert node_id("bi_asset_discovery", ["DAU"]) == "bi_asset_discovery@DAU"
    assert node_id("bi_asset_discovery", ["DAU", "GMV"]) == "bi_asset_discovery@DAU+GMV"
    assert node_id("freshness_precheck", []) == "freshness_precheck"


def test_repair_width_serializes_overflow_by_schedule():
    g = _g([(f"n{i}", "bi_asset_discovery", [str(i)]) for i in range(3)])
    pc = _pc(_grid(["0", "1", "2"]), a_max=2)
    repair_width(g, pc)
    assert any(e.kind == EdgeKind.THEN for e in g.edges) and check_feasibility(g, pc) == []


def test_repair_width_respects_role_limit_and_unequal_durations():
    g = _g([(f"n{i}", "lineage_discovery", [str(i)]) for i in range(3)])
    repair_width(g, _pc(_grid(["0", "1", "2"]), a_max=8))
    assert check_feasibility(g, _pc(_grid(["0", "1", "2"]), a_max=8)) == []
    mixed = _g([("a", "omni_discovery", ["M"]), ("b", "bi_asset_discovery", ["M"]), ("c", "lineage_discovery", ["M"]), ("d", "caliber_lookup", ["M"])], [("b", "c")])
    pc = _pc(_grid(["M"]), a_max=2)
    repair_width(mixed, pc)
    assert check_feasibility(mixed, pc) == []


# ── 阶段 ───────────────────────────────────────────────────────────────────
async def _plan(make_ctx, contract, pre, *, deps=None, **ctxkw):
    ctx = make_ctx("q", **({"deps": deps} if deps else {}), **ctxkw)
    ctx.contract, ctx.preinjection = contract, pre
    return await PlanningStage().run(ctx)


def _sig(c):
    return sorted((n.id, tuple(n.targets)) for n in c.graph.nodes)


async def test_default_plan_has_three_distinct_candidates_and_picks_the_best_feasible(make_ctx):
    plan = await _plan(make_ctx, _contract(["DAU", "GMV", "CTR"]), _pre(), max_parallel=4)
    assert plan.mode == "recall_and_select" and plan.max_parallel == 4 and plan.context.max_parallel == 4
    rules = [c for c in plan.candidates if c.source == "rule"]
    assert len(rules) == 3 and len({str(_sig(c)) + str(sorted((e.src, e.dst) for e in c.graph.edges)) for c in rules}) == 3
    feas = [c for c in plan.candidates if c.feasible]
    assert plan.chosen.feasible and plan.chosen.score == max(c.score for c in feas) and not plan.degraded
    assert set(plan.chosen.score_breakdown) >= {"coverage", "parallelism", "cost", "agents", "work", "makespan", "coverage_term", "parallel_term", "cost_term"}
    assert plan.chosen.score_breakdown["coverage"] == 1.0                          # 所有候选族都取满覆盖
    assert plan.recall_params["asset_types"] and plan.recall_params["graph_domain"] == "交易"


async def test_candidates_trade_agents_against_parallelism_and_all_reach_full_coverage(make_ctx):
    plan = await _plan(make_ctx, _contract(["DAU", "GMV", "CTR"]), _pre(), max_parallel=4)
    names = {c.name: c for c in plan.candidates}
    assert all(c.score_breakdown["coverage"] == 1.0 for c in plan.candidates)
    lanes = names.get("lanes")
    if lanes is not None:
        assert lanes.score_breakdown["agents"] > plan.chosen.score_breakdown["agents"] or lanes is plan.chosen          # 同样覆盖，Agent 更多


async def test_single_caliber_has_fewer_distinct_shapes_but_still_offers_a_choice(make_ctx):
    plan = await _plan(make_ctx, _contract(["DAU"], ["渠道"]), _pre())
    assert len(plan.candidates) == 2 and {c.name for c in plan.candidates} == {"lanes", "per_caliber"}                 # 拆分 == 批量，去重后只剩两种结构
    assert any(n.role == "omni_discovery" for c in plan.candidates for n in c.graph.nodes)


async def test_a_max_one_makes_everything_serial(make_ctx):
    plan = await _plan(make_ctx, _contract(["DAU"]), _pre(), max_parallel=1)
    assert all(c.feasible for c in plan.candidates)
    assert all(c.score_breakdown["makespan"] == c.score_breakdown["work"] for c in plan.candidates)       # 峰值并发 1 → 完工时间 = 总工作量


async def test_a_max_limits_peak_concurrency_of_every_candidate(make_ctx):
    plan = await _plan(make_ctx, _contract(["DAU", "GMV", "CTR"]), _pre(), max_parallel=2)
    for c in plan.candidates:
        assert not any("Peak concurrency" in v for v in c.violations)
        span = {x.id: (x.start, x.end) for x in c.schedule}
        for t in range(max((e for _, e in span.values()), default=0)):
            assert sum(1 for s, e in span.values() if s <= t < e) <= 2


async def test_execute_mode_when_all_l0_has_no_agent_calls(make_ctx):
    plan = await _plan(make_ctx, _contract(["DAU"]), _pre(_k("DAU", assets=["dataset:1"])))
    assert plan.mode == "execute" and plan.level == EntryLevel.L0 and {g.status for g in plan.sub_goals} == {GoalStatus.COVERED}
    assert plan.known_assets == ["dataset:1"] and plan.chosen.graph.nodes == [] and plan.chosen.feasible
    assert plan.context.pending_units() == [] and len(plan.candidates) == 1 and plan.chosen.score_breakdown["coverage"] == 1.0


async def test_l1_plan_skips_recall_and_verifies_with_recall_roles_that_provide_the_caliber_dimension(make_ctx):
    pre = _pre(_k("DAU", assets=["dataset:1"], consistency="possible_inconsistent"))
    plan = await _plan(make_ctx, _contract(["DAU"]), pre)
    assert plan.mode == "select_only" and plan.context.pending_units() == [] and plan.chosen.feasible
    nodes = plan.chosen.graph.nodes
    assert nodes and {"caliber"} <= {d for n in nodes for d in plan.context.role_dims[n.role]} and all(n.targets == ["DAU"] for n in nodes)


async def test_session_confirmed_assets_flow_from_ctx(make_ctx):
    plan = await _plan(make_ctx, _contract(["DAU"]), _pre(), confirmed_assets={"DAU": ["dataset:7"]})
    assert plan.known_assets == ["dataset:7"] and plan.level == EntryLevel.L0 and plan.sub_goals[0].status == GoalStatus.COVERED


async def test_dim_value_resolver_only_with_dim_terms_and_allowed_domain(make_ctx, deps):
    deps.settings = deps.settings.model_copy(update={"PLANNER_NUM_CANDIDATES": 5})
    with_terms = await _plan(make_ctx, _contract(["DGMV"], ["品牌"], dim_terms=["自然堂"]), _pre(), deps=deps, max_parallel=8)
    assert any(n.role == "dim_value_resolver" for c in with_terms.candidates for n in c.graph.nodes)
    no_terms = await _plan(make_ctx, _contract(["DGMV"], ["品牌"]), _pre(), deps=deps, max_parallel=8)
    assert not any(n.role == "dim_value_resolver" for c in no_terms.candidates for n in c.graph.nodes)
    cost = await _plan(make_ctx, _contract(["H200"], ["项目"], domain="resource_cost", dim_terms=["x"]), _pre(), deps=deps, max_parallel=8)
    assert not any(n.role == "dim_value_resolver" for c in cost.candidates for n in c.graph.nodes)


async def test_degraded_when_nothing_is_feasible(make_ctx, monkeypatch):
    import app.stages.s3_planning as s3
    from app.roles import list_role_specs
    monkeypatch.setattr(s3, "list_role_specs", lambda: [sp for sp in list_role_specs() if "freshness" not in sp.dims])   # 没有角色能提供 freshness
    plan = await _plan(make_ctx, _contract(["DAU"]), _pre())
    assert plan.degraded and any("No node provides" in v for v in plan.chosen.violations) and any("fell back" in n for n in plan.notes)


async def test_settings_drive_weights_k_and_defaults(make_ctx, deps):
    deps.settings = deps.settings.model_copy(update={"MAX_PARALLEL_AGENTS": 2, "PLANNER_W_COVERAGE": 0.5, "PLANNER_W_PARALLEL": 0.3, "PLANNER_W_COST": 0.1, "PLANNER_NUM_CANDIDATES": 2})
    plan = await _plan(make_ctx, _contract(["DAU"]), _pre(), deps=deps)
    pc = plan.context
    assert plan.max_parallel == 2 and (pc.w_coverage, pc.w_parallel, pc.w_cost) == (0.5, 0.3, 0.1)
    assert len([c for c in plan.candidates if c.source == "rule"]) <= 2


# ── 规划 Agent：LLM 只做提案，定级 / 过滤 / 评分仍是确定性的 ─────────────────
class FakeLLM:
    def __init__(self, reply=None, exc=None):
        self.reply, self.exc, self.prompts = reply, exc, []

    async def complete(self, *, prompt, system=None, json_mode=False):
        self.prompts.append(prompt)
        if self.exc:
            raise self.exc
        return self.reply if isinstance(self.reply, str) else json.dumps(self.reply, ensure_ascii=False)


LLM_PLAN = {"candidates": [
    {"name": "llm_batched", "rationale": "批量口径", "nodes": [
        {"role": "bi_asset_discovery", "targets": ["DAU", "GMV"]}, {"role": "lineage_discovery", "targets": ["DAU", "GMV"]},
        {"role": "caliber_lookup", "targets": ["DAU", "GMV"]}],
     "edges": [{"src": "bi_asset_discovery@DAU+GMV", "dst": "lineage_discovery@DAU+GMV", "kind": "dep"}]},
    {"name": "llm_no_checks", "nodes": [{"role": "caliber_lookup", "targets": ["DAU", "GMV"]}], "edges": []},
    {"name": "llm_bad_role", "nodes": [{"role": "ghost"}], "edges": []},
]}


async def test_llm_proposals_are_scored_and_filtered_like_rule_candidates(make_ctx):
    llm = FakeLLM(LLM_PLAN)
    plan = await _plan(make_ctx, _contract(["DAU", "GMV"]), _pre(), deps=Deps(llm=llm), max_parallel=8)
    by = {c.name: c for c in plan.candidates}
    assert plan.llm_used and by["llm_batched"].source == "llm" and by["llm_batched"].feasible
    assert by["llm_batched"].score_breakdown["coverage"] == 1.0 and by["llm_batched"].graph.nodes[0].targets == ["DAU", "GMV"]
    assert "llm_bad_role" not in by                                                          # 非法角色整张图丢弃
    nc = by["llm_no_checks"]
    assert nc.feasible is False and any("grain_fit" in v or "freshness" in v for v in nc.violations)   # 省掉提供评估维度的角色不能换来高分
    assert "DAU" in llm.prompts[0] and "omni_discovery" in llm.prompts[0] and "{{" not in llm.prompts[0]
    assert plan.chosen.score == max(c.score for c in plan.candidates if c.feasible)


@pytest.mark.parametrize("llm", [FakeLLM(exc=UpstreamError("llm", "boom")), FakeLLM("not json"), FakeLLM({"candidates": "x"}), FakeLLM({"candidates": []})])
async def test_llm_failure_falls_back_to_rule_candidates(make_ctx, llm):
    plan = await _plan(make_ctx, _contract(["DAU"]), _pre(), deps=Deps(llm=llm))
    assert not plan.llm_used and plan.candidates and all(c.source == "rule" for c in plan.candidates)


async def test_default_null_llm_is_pure_rules(make_ctx):
    assert not (await _plan(make_ctx, _contract(["DAU"]), _pre())).llm_used


# ── 契约 ───────────────────────────────────────────────────────────────────
def test_cycle_rejected_and_layers():
    g = _g([("a", "x", []), ("b", "x", []), ("c", "x", [])], [("a", "b"), ("a", "c")])
    assert g.layers() == [["a"], ["b", "c"]]
    with pytest.raises(ContractValidationError):
        _g([("a", "x", []), ("b", "x", [])], [("a", "b"), ("b", "a")]).layers()


async def test_plan_for_an_l2_caliber_always_includes_graph_and_caliber_roles_even_when_knowledge_hit(make_ctx):
    pre = _pre(_k("DAU", assets=["dws.user_active_d"]))
    plan = await _plan(make_ctx, _contract(["DAU"], ["渠道"]), pre, max_parallel=8)
    roles = {n.role for c in plan.candidates if c.feasible for n in c.graph.nodes}
    assert {"bi_asset_discovery", "lineage_discovery", "caliber_lookup"} <= roles or "omni_discovery" in roles
    best = {n.role for n in plan.chosen.graph.nodes}
    assert ("lineage_discovery" in best or "omni_discovery" in best) and ("caliber_lookup" in best or "omni_discovery" in best)
    assert plan.chosen.score_breakdown["coverage"] == 1.0


# ── 规划结果里只有一个级别，且级别决定 C₀（全有或全无）──────────────────────────────
async def test_plan_has_one_level_with_reason_and_basis_and_it_decides_the_mode(make_ctx):
    plan = await _plan(make_ctx, _contract(["DAU", "GMV"], ["渠道"]), _pre(_k("DAU", assets=["dataset:1", "dws.t"])))
    assert plan.level == EntryLevel.L2 and plan.mode == "recall_and_select" and plan.level_rule_id == "level.l2.uncovered_goals"
    assert set(plan.level_basis) == {"GMV", "渠道"} and "GMV" in plan.level_reason
    assert {s.name: s.status for s in plan.sub_goals}["DAU"] == GoalStatus.COVERED                  # 子目标状态保留作依据
    l1 = await _plan(make_ctx, _contract(["DAU"]), _pre(_k("DAU", assets=["dataset:1"], consistency="possible_inconsistent")))
    assert l1.level == EntryLevel.L1 and l1.mode == "select_only"


async def test_an_l2_task_recalls_every_metric_even_the_ones_knowledge_already_covers(make_ctx):
    plan = await _plan(make_ctx, _contract(["DAU", "GMV"]), _pre(_k("DAU", assets=["dataset:1"])), max_parallel=8)
    assert plan.level == EntryLevel.L2
    assert {u.caliber for u in plan.context.pending_units()} == {"DAU", "GMV"} and not any(u.covered for u in plan.context.units)      # 不为已覆盖的 DAU 开例外
    assert {t for n in plan.chosen.graph.nodes if n.phase == "recall" for t in n.targets} == {"DAU", "GMV"}


async def test_l0_and_l1_tasks_cover_every_pair_without_recall(make_ctx):
    for pre in (_pre(_k("DAU", assets=["dataset:1"])), _pre(_k("DAU", assets=["dataset:1"], consistency="possible_inconsistent"))):
        plan = await _plan(make_ctx, _contract(["DAU"]), pre)
        assert plan.context.pending_units() == [] and all(u.covered for u in plan.context.units)
        if plan.mode == "execute":
            assert plan.chosen.graph.nodes == []
        else:                                                                            # L1：不召回，只由召回角色核验（每个节点的工作量为 1）
            assert plan.chosen.score_breakdown["work"] == len(plan.chosen.graph.nodes)
