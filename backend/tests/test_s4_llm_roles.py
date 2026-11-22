"""④ 角色由 LLM 驱动：角色模板 / 改写检索 query / 评估结果 / 换角度重试 / 节点内并发；LLM 不可用时回落规则。"""
import asyncio
import dataclasses
import json

import pytest

from app.deps import Deps
from app.errors import UpstreamError
from app.llm_client import NullLLM
from app.prompts import load_prompt
from app.roles import REGISTRY, list_role_specs
from app.schemas.knowledge import KnowledgeItem, KnowledgeKind, KnowledgeOrigin
from app.schemas.recall import AssetMeta, CandidateAsset
from app.schemas.workflow import EdgeKind, WorkflowEdge, WorkflowGraph
from app.services.fakes import FakeBIRetrieval, FakeKnowledgeRetrieval, FakeLineageRetrieval
from app.settings import settings as _settings
from app.stages import RunContext
from app.stages.s2_coordination import CoordinationStage
from app.stages.s3_planning import PlanningStage
from app.stages.s3_planning.candidates import make_node
from app.stages.s4_recall import RecallStage

BI, LN, CAL, OMNI = "bi_asset_discovery", "lineage_discovery", "caliber_lookup", "omni_discovery"
MARKERS = {"plan": "任务：改写检索 query", "annotate": "任务：评估检索结果", "selfeval": "任务：自评是否接活", "ask": "任务：撰写求助问题"}


class ScriptedLLM:
    """按 prompt 里的任务标记分流；值可以是 dict / callable(prompt)->dict / Exception。未配置的任务返回 {}（可解析但没有内容）。"""

    def __init__(self, **handlers):
        self.handlers = handlers
        self.calls: list[tuple[str, str]] = []
        self.active = self.peak = 0
        self.delay = handlers.pop("delay", 0.0) if "delay" in handlers else 0.0

    def of(self, purpose):
        return [p for k, p in self.calls if k == purpose]

    async def complete(self, *, prompt, system=None, json_mode=False):
        purpose = next((k for k, m in MARKERS.items() if m in prompt), "other")
        self.calls.append((purpose, prompt))
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            h = self.handlers.get(purpose, {})
            h = h(prompt) if callable(h) else h
            if isinstance(h, Exception):
                raise h
            return h if isinstance(h, str) else json.dumps(h, ensure_ascii=False)
        finally:
            self.active -= 1


def cfg(**kw):
    return _settings.model_copy(update={"RECALL_LLM_ENABLED": True, "COLLAB_HANDOFF_ENABLED": True, **kw})


class SpyBI(FakeBIRetrieval):
    def __init__(self, delay=0.0, only=None):
        super().__init__()
        self.calls: list[str] = []
        self.delay, self.only = delay, only
        self.active = self.peak = 0

    async def search(self, query, **kw):
        self.calls.append(query)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
        finally:
            self.active -= 1
        if self.only is not None and self.only not in query:
            return []
        return await super().search(query, limit=5)


def plan_reply(*plans):
    return {"plans": [{"metric": p[0], "query": p[1], "angles": list(p[2]) if len(p) > 2 else []} for p in plans]}


async def run_graph(specs, deps, query="最近7天DAU", edges=(), a_max=8):
    ctx = RunContext(query=query, deps=dataclasses.replace(deps, llm=NullLLM()), cookie="c=t")        # ②③ 不用 LLM：这里只数 ④ 里的调用
    ctx.contract = await CoordinationStage().run(ctx)
    ctx.plan = await PlanningStage().run(ctx)
    ctx.deps = deps
    g = WorkflowGraph(nodes=[make_node(REGISTRY[r].spec, t) for r, t in specs], edges=[WorkflowEdge(src=a, dst=b, kind=EdgeKind.DEP) for a, b in edges])
    ctx.plan = ctx.plan.model_copy(update={"chosen": ctx.plan.chosen.model_copy(update={"graph": g}), "max_parallel": a_max, "known_assets": []})
    return ctx, await RecallStage().run(ctx)


# ── 角色模板 ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("role", [BI, LN, CAL, OMNI, "dim_value_resolver"])
def test_recall_roles_have_templates_that_include_the_blackboard_protocol(role):
    t = load_prompt(f"roles/{role}")
    assert "{{ include" not in t and "{{include" not in t
    assert "黑板协作协议" in t and REGISTRY[role].spec.description.split("：")[0] in t


def test_role_templates_describe_channels_and_when_to_decline():
    assert "bi" in load_prompt(f"roles/{BI}") and "婉拒" in load_prompt(f"roles/{BI}")
    assert "血缘" in load_prompt(f"roles/{LN}") and "口径" in load_prompt(f"roles/{CAL}")
    import re
    assert all(re.search(rf"\b{ch}\b", load_prompt(f"roles/{OMNI}")) for ch in ("bi", "graph", "knowledge"))


def test_spec_template_name_defaults_to_role_name_and_an_explicit_template_wins():
    from dataclasses import replace
    specs = {s.name: s for s in list_role_specs()}
    assert specs[BI].template_name == f"roles/{BI}" and replace(specs[BI], template="roles/custom").template_name == "roles/custom"
    assert replace(specs[BI], phase="select").template_name is None                          # 选择阶段的角色没有召回模板


# ── 改写检索 query：一个节点一次调用 ─────────────────────────────────────────────────
async def test_plan_rewrites_the_query_used_for_the_channel_but_keeps_the_need_key():
    bi, llm = SpyBI(), ScriptedLLM(plan=plan_reply(("DAU", "最近7天各渠道的日活跃用户数 DAU")))
    _, rec = await run_graph([(BI, ["DAU"])], Deps(bi=bi, llm=llm, settings=cfg()))
    assert bi.calls == ["最近7天各渠道的日活跃用户数 DAU"]
    assert {c.key for c in rec.cards} >= {"assets:DAU"} and rec.role_reports[0].status == "ok" and rec.candidates


async def test_rewritten_query_must_keep_the_metric_name():
    bi, llm = SpyBI(), ScriptedLLM(plan=plan_reply(("DAU", "各渠道日活")))
    await run_graph([(BI, ["DAU"])], Deps(bi=bi, llm=llm, settings=cfg()))
    assert bi.calls == ["DAU 各渠道日活"]


async def test_one_plan_call_covers_every_metric_of_the_node_and_unknown_metrics_are_ignored():
    bi = SpyBI()
    llm = ScriptedLLM(plan=plan_reply(("DAU", "DAU 日活"), ("GHOST", "GHOST x")))        # GMV 没有给计划 → 用指标名
    await run_graph([(BI, ["DAU", "GMV"])], Deps(bi=bi, llm=llm, settings=cfg()), query="DAU和GMV")
    assert len(llm.of("plan")) == 1 and "DAU" in llm.of("plan")[0] and "GMV" in llm.of("plan")[0]
    assert sorted(bi.calls) == ["DAU 日活", "GMV"]


async def test_plan_prompt_carries_role_template_contract_and_has_no_leftover_placeholders():
    llm = ScriptedLLM()
    await run_graph([(BI, ["DAU"])], Deps(bi=SpyBI(), llm=llm, settings=cfg()))
    p = llm.of("plan")[0]
    assert "黑板协作协议" in p and "BI 资产发现" in p and "last_7_days" in p and "{{" not in p


async def test_malformed_plan_entries_fall_back_to_the_metric_name():
    bi = SpyBI()
    llm = ScriptedLLM(plan={"plans": [{"metric": "DAU", "query": ["x"]}, "oops", {"metric": "DAU", "query": ""}]})
    _, rec = await run_graph([(BI, ["DAU"])], Deps(bi=bi, llm=llm, settings=cfg()))
    assert bi.calls == ["DAU"] and rec.role_reports[0].status == "ok"


# ── 回落：LLM 不可用 / 关闭 / 超预算 ──────────────────────────────────────────────
async def test_flag_off_means_zero_llm_calls_and_unchanged_behaviour():
    bi, llm = SpyBI(), ScriptedLLM(plan=plan_reply(("DAU", "DAU 改写")))
    _, rec = await run_graph([(BI, ["DAU"])], Deps(bi=bi, llm=llm, settings=cfg(RECALL_LLM_ENABLED=False)))
    assert llm.calls == [] and bi.calls == ["DAU"] and rec.collab.llm.enabled is False and rec.collab.llm.calls == 0


async def test_llm_failure_falls_back_to_the_rule_path_and_is_counted():
    bi = SpyBI()
    llm = ScriptedLLM(plan=UpstreamError("llm", "boom"), annotate=UpstreamError("llm", "boom"))
    _, rec = await run_graph([(BI, ["DAU"])], Deps(bi=bi, llm=llm, settings=cfg()))
    assert bi.calls == ["DAU"] and rec.role_reports[0].status == "ok" and rec.candidates
    assert rec.collab.llm.failures == 2 and rec.collab.llm.fallbacks == 2


async def test_budget_exhaustion_degrades_later_calls_but_the_node_still_finishes():
    bi, llm = SpyBI(), ScriptedLLM(plan=plan_reply(("DAU", "DAU 日活")))
    _, rec = await run_graph([(BI, ["DAU"])], Deps(bi=bi, llm=llm, settings=cfg(RECALL_LLM_MAX_CALLS=1)))
    assert len(llm.calls) == 1 and bi.calls == ["DAU 日活"] and rec.role_reports[0].status == "ok"
    assert rec.collab.llm.budget_exhausted and rec.collab.llm.calls == 1


async def test_each_node_uses_at_most_two_llm_calls_when_nothing_is_asked():
    llm = ScriptedLLM(plan=plan_reply(("DAU", "DAU 日活")))
    _, rec = await run_graph([(BI, ["DAU"])], Deps(bi=SpyBI(), llm=llm, settings=cfg()))
    assert [k for k, _ in llm.calls] == ["plan", "annotate"] and rec.collab.llm.by_purpose == {"plan": 1, "annotate": 1}


# ── 评估结果：标注而不删除，只能引用通道真实返回的资产 ────────────────────────────────
def judge(*items):
    return {"judgements": [{"card": c, "asset_id": a, "relevant": r, "reason": why, "confidence": conf} for c, a, r, why, conf in items]}


async def test_annotation_becomes_evidence_and_never_changes_the_candidate_pool():
    _, base = await run_graph([(BI, ["DAU"])], Deps(bi=SpyBI(), settings=cfg(RECALL_LLM_ENABLED=False)))
    llm = ScriptedLLM(annotate=judge(("assets:DAU", "dws.user_active_d", True, "日活汇总，口径一致", 0.9),
                                     ("assets:DAU", "ghost.table", True, "编造的资产", 0.9)))
    _, rec = await run_graph([(BI, ["DAU"])], Deps(bi=SpyBI(), llm=llm, settings=cfg()))
    assert {c.asset_id for c in rec.candidates} == {c.asset_id for c in base.candidates}      # 候选不增不减
    ann = [e for e in rec.evidences if e.source == "llm_filter"]
    assert [e.asset_id for e in ann] == ["dws.user_active_d"] and "日活汇总" in ann[0].statement and ann[0].confidence == 0.9
    assert not any(e.asset_id == "ghost.table" for e in rec.evidences)


async def test_irrelevant_verdict_is_kept_visibly_with_its_reason():
    llm = ScriptedLLM(annotate=judge(("assets:DAU", "dws.user_active_d", False, "这是 eDAU 的表，口径不同", 0.2)))
    _, rec = await run_graph([(BI, ["DAU"])], Deps(bi=SpyBI(), llm=llm, settings=cfg()))
    ev = next(e for e in rec.evidences if e.source == "llm_filter")
    assert "irrelevant" in ev.statement and "eDAU" in ev.statement and ev.value == "irrelevant"
    assert "dws.user_active_d" in {c.asset_id for c in rec.candidates}


async def test_annotate_prompt_lists_only_real_candidates_of_the_nodes_own_cards():
    llm = ScriptedLLM()
    await run_graph([(BI, ["DAU"])], Deps(bi=SpyBI(), llm=llm, settings=cfg()))
    p = llm.of("annotate")[0]
    assert "dws.user_active_d" in p and "assets:DAU" in p and "{{" not in p


# ── 换角度重试：空卡 / 全部不相关 → 走转交机制 ─────────────────────────────────────────
async def test_empty_card_retries_another_angle_via_handoff():
    bi = SpyBI(only="日活")                                                               # 只有换成「日活」的说法才有结果
    llm = ScriptedLLM(plan=plan_reply(("DAU", "DAU 活跃", ["日活"])))
    _, rec = await run_graph([(BI, ["DAU"])], Deps(bi=bi, llm=llm, settings=cfg()))
    assert bi.calls == ["DAU 活跃", "日活"]
    card = next(c for c in rec.cards if c.key == "assets:日活")
    assert card.asset_ids and rec.candidates
    ev = [e for e in rec.collab.events if e.via == "assets:DAU"]
    assert ev and all(e.hop == 1 for e in ev)


async def test_all_irrelevant_verdict_also_triggers_the_angle_retry():
    bi = SpyBI()
    llm = ScriptedLLM(plan=plan_reply(("DAU", "DAU 活跃", ["日活"])), annotate=lambda p: judge(
        *[("assets:DAU", a, False, "不是同一指标", 0.1) for a in ("dws.user_active_d", "dws.user_active_valid_d")]))
    _, rec = await run_graph([(BI, ["DAU"])], Deps(bi=bi, llm=llm, settings=cfg()))
    assert "日活" in bi.calls and any(c.key == "assets:日活" for c in rec.cards)


async def test_angle_retry_is_bounded_by_handoff_depth_and_switch():
    for kw in ({"COLLAB_HANDOFF_MAX_DEPTH": 0}, {"COLLAB_HANDOFF_ENABLED": False}, {"COLLAB_HANDOFF_MAX_TOTAL": 0}):
        bi = SpyBI(only="日活")
        llm = ScriptedLLM(plan=plan_reply(("DAU", "DAU 活跃", ["日活"])))
        await run_graph([(BI, ["DAU"])], Deps(bi=bi, llm=llm, settings=cfg(**kw)))
        assert bi.calls == ["DAU 活跃"], kw


async def test_angles_equal_to_the_metric_are_dropped_and_fanout_is_respected():
    bi = SpyBI(only="zzz")
    llm = ScriptedLLM(plan=plan_reply(("DAU", "DAU 活跃", ["DAU", "日活", "活跃用户", "DAU用户"])))
    await run_graph([(BI, ["DAU"])], Deps(bi=bi, llm=llm, settings=cfg(COLLAB_HANDOFF_FANOUT=2)))
    assert bi.calls[0] == "DAU 活跃" and len(bi.calls) == 3 and set(bi.calls[1:]) <= {"日活", "活跃用户", "DAU用户"}


async def test_no_angles_without_llm_so_empty_cards_do_not_retry():
    bi = SpyBI(only="zzz")
    await run_graph([(BI, ["DAU"])], Deps(bi=bi, settings=cfg(RECALL_LLM_ENABLED=False)))
    assert bi.calls == ["DAU"]


# ── 节点内并发 ─────────────────────────────────────────────────────────────────
async def test_a_node_with_several_metrics_probes_them_concurrently():
    bi = SpyBI(delay=0.05)
    _, rec = await run_graph([(BI, ["DAU", "GMV", "CTR"])], Deps(bi=bi, settings=cfg(RECALL_LLM_ENABLED=False)), query="DAU和GMV和CTR")
    assert bi.peak == 3 and sorted(bi.calls) == ["CTR", "DAU", "GMV"] and rec.role_reports[0].status == "ok"


async def test_omni_node_runs_metrics_and_channels_concurrently_but_lineage_follows_bi():
    class SpyLN(FakeLineageRetrieval):
        def __init__(self):
            super().__init__()
            self.seeds: list[list[str]] = []

        async def search(self, query, *, limit=10, cookie=None, params=None):
            self.seeds.append(list((params or {}).get("seed_assets", [])))
            return await super().search(query, limit=limit)

    bi, ln = SpyBI(delay=0.05), SpyLN()
    _, rec = await run_graph([(OMNI, ["DAU", "GMV"])], Deps(bi=bi, lineage=ln, settings=cfg(RECALL_LLM_ENABLED=False)), query="DAU和GMV")
    assert bi.peak == 2                                                                   # 两个口径的 BI 通道同时查，而不是嵌套循环
    assert {c.key for c in rec.cards} >= {f"{k}:{m}" for k in ("assets", "lineage", "caliber") for m in ("DAU", "GMV")}
    assert all(seeds for seeds in ln.seeds)                                               # 血缘在 BI 之后：种子来自 BI 的证据卡
    assert rec.role_reports[0].status == "ok"


async def test_a_failing_probe_still_fails_the_node_without_leaving_orphans():
    class BoomBI(SpyBI):
        async def search(self, query, **kw):
            if "GMV" in query:
                raise RuntimeError("down")
            return await super().search(query, **kw)

    _, rec = await run_graph([(BI, ["DAU", "GMV"])], Deps(bi=BoomBI(), settings=cfg(RECALL_LLM_ENABLED=False)), query="DAU和GMV")
    assert rec.role_reports[0].status == "failed" and "down" in rec.role_reports[0].error


# ── 轨迹 ───────────────────────────────────────────────────────────────────────
async def test_trace_reports_llm_usage_over_http():
    from fastapi.testclient import TestClient
    from app.main import create_app
    b = TestClient(create_app(Deps())).post("/api/stage4", json={"query": "最近7天DAU"}).json()
    assert b["recall"]["collab"]["llm"] == {"enabled": False, "calls": 0, "failures": 0, "fallbacks": 0, "max_calls": 40, "budget_exhausted": False, "by_purpose": {}}


async def test_llm_filter_evidence_records_which_metric_it_judged():
    llm = ScriptedLLM(annotate=judge(("assets:DAU", "dws.user_active_d", True, "ok", 0.9)))
    _, rec = await run_graph([(BI, ["DAU"])], Deps(bi=SpyBI(), llm=llm, settings=cfg()))
    assert [e.subject for e in rec.evidences if e.source == "llm_filter"] == ["DAU"]
