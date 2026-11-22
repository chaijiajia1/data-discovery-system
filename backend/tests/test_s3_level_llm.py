"""③ 入口定级：LLM 判子目标覆盖状态；任务级别仍由确定性汇总（取最差者）；LLM 不可用 / 输出非法时回落规则。"""
import json

import pytest

from app.deps import Deps
from app.errors import UpstreamError
from app.schemas.intent import Intent
from app.schemas.knowledge import KnowledgeItem, KnowledgeKind, KnowledgeOrigin, PreInjectionResult
from app.schemas.task import Slot, TaskContract, ValueSource
from app.schemas.workflow import EntryLevel, GoalStatus
from app.settings import settings as _settings
from app.stages.s3_planning import PlanningStage
from app.stages.s3_planning.level_agent import LevelAgent, build_prompt


class FakeLLM:
    def __init__(self, reply=None, exc=None):
        self.reply, self.exc, self.prompts = reply, exc, []

    async def complete(self, *, prompt, system=None, json_mode=False):
        self.prompts.append(prompt)
        if self.exc:
            raise self.exc
        return self.reply if isinstance(self.reply, str) else json.dumps(self.reply, ensure_ascii=False)


class RoutedLLM(FakeLLM):
    """按 prompt 区分：定级调用返回 level_reply；规划提案调用返回空候选。"""
    def __init__(self, level_reply=None, exc=None):
        super().__init__(level_reply, exc)

    async def complete(self, *, prompt, system=None, json_mode=False):
        if "子目标覆盖状态判定" in prompt:
            return await super().complete(prompt=prompt, system=system, json_mode=json_mode)
        return '{"candidates": []}'


def _contract(metrics, dims=()):
    return TaskContract(query="q", metrics=Slot(value=list(metrics), source=ValueSource.USER),
                        dimensions=Slot(value=list(dims), source=ValueSource.USER),
                        intent=Intent(query="q", intent_type="data_fetch", domain="transaction"))


def _k(title, kind=KnowledgeKind.METRIC_DEF, assets=(), i=None):
    return KnowledgeItem(id=i or f"k_{title}", kind=kind, title=title, content=f"{title} 的定义", origin=KnowledgeOrigin.RETRIEVAL,
                         asset_ids=list(assets), relevance=0.9, token_cost=10)


def _pre(*items):
    return PreInjectionResult(injected=list(items))


def _reply(*goals):
    return {"goals": list(goals)}


def _goal(kind, name, status, reason="理由", assets=()):
    return {"kind": kind, "name": name, "status": status, "reason": reason, "asset_ids": list(assets)}


async def _regrade(llm, contract, pre, confirmed=None):
    from app.stages.s3_planning.levels import grade_task
    base = grade_task(contract, pre, confirmed=confirmed)
    return await LevelAgent(llm).regrade(contract, pre, base.goals, confirmed=confirmed or {}), base


# ── prompt ────────────────────────────────────────────────────────────────
async def test_prompt_contains_goals_knowledge_and_no_unreplaced_placeholders():
    pre = _pre(_k("DAU", assets=["dataset:1"], i="k_dau"))
    p = build_prompt(_contract(["DAU"], ["渠道"]), pre, confirmed={})
    assert "子目标覆盖状态判定" in p and "DAU" in p and "渠道" in p and "k_dau" in p and "dataset:1" in p and "{{" not in p
    assert "任意数据源资产" in p and "明细表" in p                        # 默认：BI 资产或明细表都算指明了资产
    from app.stages.s3_planning.level_agent import build_prompt as bp
    assert "BI 资产（bi_dataset）" in bp(_contract(["DAU"], ["渠道"]), pre, confirmed={}, require_bi=True)


async def test_prompt_query_cannot_inject_placeholders():
    c = _contract(["{{knowledge}}"])
    assert "{{knowledge}}" in build_prompt(c, _pre(), confirmed={})   # 原样出现，不被二次替换


# ── LLM 判定被采用 ─────────────────────────────────────────────────────────────
async def test_llm_status_overrides_rule_with_traceable_rule_id():
    pre = _pre(_k("DAU", assets=["dataset:1"], i="k_dau"))
    goals, base = await _regrade(FakeLLM(_reply(_goal("metric", "DAU", "doubtful", "口径描述含糊", ["dataset:1"]))), _contract(["DAU"]), pre)
    g = goals[0]
    assert g.status == GoalStatus.DOUBTFUL and g.rule_id == "llm.goal.doubtful" and g.reason == "口径描述含糊" and g.asset_ids == ["dataset:1"]


async def test_llm_can_cover_what_rules_would_not():
    """规则按词面匹配：知识标题是『日活』，指标写 DAU，规则判 uncovered；LLM 认出同义，判 covered。"""
    pre = _pre(_k("日活", assets=["dataset:1"], i="k_dh"))
    goals, _ = await _regrade(FakeLLM(_reply(_goal("metric", "DAU", "covered", "日活即 DAU", ["dataset:1"]))), _contract(["DAU"]), pre)
    assert goals[0].status == GoalStatus.COVERED and goals[0].asset_ids == ["dataset:1"]


# ── 回落 / 校验 ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("llm", [FakeLLM(exc=UpstreamError("llm", "boom")), FakeLLM("not json"), FakeLLM({"goals": "x"}), FakeLLM({})])
async def test_llm_failure_returns_none_for_rule_fallback(llm):
    goals, _ = await _regrade(llm, _contract(["DAU"]), _pre())
    assert goals is None


async def test_invalid_goal_entries_keep_rule_result_for_that_goal():
    pre = _pre(_k("DAU", assets=["dataset:1"]), _k("GMV", assets=["dataset:2"]))
    reply = _reply(_goal("metric", "DAU", "super_covered"),                 # 非法状态 → 该目标保留规则结果
                   _goal("metric", "GHOST", "covered", assets=["dataset:1"]),  # 未知子目标 → 忽略
                   _goal("metric", "GMV", "doubtful", assets=["dataset:2"]))
    goals, base = await _regrade(FakeLLM(reply), _contract(["DAU", "GMV"]), pre)
    by = {g.name: g for g in goals}
    assert set(by) == {"DAU", "GMV"}
    assert by["DAU"].rule_id == next(g for g in base.goals if g.name == "DAU").rule_id          # 保留规则
    assert by["GMV"].status == GoalStatus.DOUBTFUL


async def test_missing_goal_in_reply_keeps_rule_result():
    pre = _pre(_k("DAU", assets=["dataset:1"]), _k("GMV", assets=["dataset:2"]))
    goals, base = await _regrade(FakeLLM(_reply(_goal("metric", "DAU", "doubtful", assets=["dataset:1"]))), _contract(["DAU", "GMV"]), pre)
    assert next(g for g in goals if g.name == "GMV").rule_id == next(g for g in base.goals if g.name == "GMV").rule_id


async def test_hallucinated_asset_ids_are_dropped_and_covered_without_assets_keeps_rule():
    pre = _pre(_k("DAU", assets=["dataset:1"]))
    goals, base = await _regrade(FakeLLM(_reply(_goal("metric", "DAU", "covered", assets=["dataset:999"]))), _contract(["DAU"]), pre)
    assert goals[0].rule_id == base.goals[0].rule_id                       # 资产 id 不在注入知识里 → 不可信，保留规则


async def test_llm_may_keep_valid_subset_of_assets():
    pre = _pre(_k("DAU", assets=["dataset:1", "hive_table:dws.x"]))
    goals, _ = await _regrade(FakeLLM(_reply(_goal("metric", "DAU", "covered", assets=["dataset:1", "dataset:999"]))), _contract(["DAU"]), pre)
    assert goals[0].asset_ids == ["dataset:1"]


async def test_uncovered_needs_no_assets():
    goals, _ = await _regrade(FakeLLM(_reply(_goal("metric", "DAU", "uncovered", "没有相关知识"))), _contract(["DAU"]), _pre(_k("DAU", assets=["dataset:1"])))
    assert goals[0].status == GoalStatus.UNCOVERED and goals[0].rule_id == "llm.goal.uncovered"


async def test_session_confirmed_goal_is_never_overridden_by_llm():
    goals, _ = await _regrade(FakeLLM(_reply(_goal("metric", "DAU", "uncovered"))), _contract(["DAU"]), _pre(), confirmed={"DAU": ["chart:9"]})
    assert goals[0].status == GoalStatus.COVERED and goals[0].rule_id == "goal.covered.session_confirmed"


async def test_dimension_goal_graded_by_llm():
    pre = _pre(_k("渠道", kind=KnowledgeKind.DIMENSION, assets=["dws.user_active_d"], i="k_ch"))
    goals, _ = await _regrade(FakeLLM(_reply(_goal("dimension", "渠道", "covered", "已绑定", ["dws.user_active_d"]))), _contract(["DAU"], ["渠道"]), pre)
    d = next(g for g in goals if g.kind == "dimension")
    assert d.status == GoalStatus.COVERED and d.rule_id == "llm.goal.covered"


# ── 阶段：任务级别仍由确定性汇总 ──────────────────────────────────────────────────
@pytest.fixture
def llm_level_on(monkeypatch):
    monkeypatch.setattr(_settings, "PLANNER_LLM_LEVEL_ENABLED", True)


async def _plan(make_ctx, llm, contract, pre):
    ctx = make_ctx("q", deps=Deps(llm=llm))
    ctx.contract, ctx.preinjection = contract, pre
    return await PlanningStage().run(ctx)


async def test_stage_level_is_worst_of_llm_goals(make_ctx, llm_level_on):
    pre = _pre(_k("DAU", assets=["dataset:1"]), _k("GMV", assets=["dataset:2"]))
    llm = RoutedLLM(_reply(_goal("metric", "DAU", "covered", assets=["dataset:1"]), _goal("metric", "GMV", "uncovered", "知识里没有 GMV 口径")))
    plan = await _plan(make_ctx, llm, _contract(["DAU", "GMV"]), pre)
    assert plan.level == EntryLevel.L2 and plan.mode == "recall_and_select" and plan.level_basis == ["GMV"]
    assert plan.llm_used and any("LLM" in n for n in plan.notes)
    assert {g.name: g.rule_id for g in plan.sub_goals} == {"DAU": "llm.goal.covered", "GMV": "llm.goal.uncovered"}


async def test_stage_all_covered_by_llm_gives_l0_and_known_assets(make_ctx, llm_level_on):
    pre = _pre(_k("日活", assets=["dataset:1"]))
    llm = RoutedLLM(_reply(_goal("metric", "DAU", "covered", "日活即 DAU", ["dataset:1"])))
    plan = await _plan(make_ctx, llm, _contract(["DAU"]), pre)
    assert plan.level == EntryLevel.L0 and plan.mode == "execute" and plan.known_assets == ["dataset:1"]


@pytest.mark.parametrize("llm", [RoutedLLM(exc=UpstreamError("llm", "boom")), RoutedLLM("garbage")])
async def test_stage_falls_back_to_rules_when_llm_fails(make_ctx, llm_level_on, llm):
    pre = _pre(_k("DAU", assets=["dataset:1"]))
    plan = await _plan(make_ctx, llm, _contract(["DAU"]), pre)
    assert plan.level == EntryLevel.L0 and plan.sub_goals[0].rule_id == "goal.covered.knowledge_assets"      # 规则结果
    assert all(not g.rule_id.startswith("llm.") for g in plan.sub_goals)
    assert any("fell back" in n for n in plan.notes)


async def test_stage_null_llm_is_pure_rules_without_notes(make_ctx, llm_level_on):
    plan = await _plan(make_ctx, Deps().llm, _contract(["DAU"]), _pre())
    assert all(not g.rule_id.startswith("llm.") for g in plan.sub_goals)


async def test_stage_flag_off_never_calls_llm(make_ctx, monkeypatch):
    monkeypatch.setattr(_settings, "PLANNER_LLM_LEVEL_ENABLED", False)
    llm = RoutedLLM(_reply(_goal("metric", "DAU", "covered", assets=["dataset:1"])))
    await _plan(make_ctx, llm, _contract(["DAU"]), _pre(_k("DAU", assets=["dataset:1"])))
    assert not any("子目标覆盖状态判定" in p for p in llm.prompts)
