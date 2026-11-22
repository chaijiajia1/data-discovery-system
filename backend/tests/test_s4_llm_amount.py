"""④ 由 LLM 决定召回多少条；召回后评估发现不够，再继续召回（受 RECALL_MAX_SUPPLEMENTS / RECALL_MAX_LIMIT / 总预算约束）。"""
import json

import pytest

from app.deps import Deps
from app.errors import UpstreamError
from app.schemas.recall import AssetMeta, CandidateAsset
from app.services.fakes import FakeBIRetrieval
from tests.test_s4_llm_roles import BI, ScriptedLLM, cfg, run_graph


def assets(n):
    return [CandidateAsset(asset_id=f"dataset:{i}", name=f"DAU 数据集{i}", asset_type="dataset", retrieval_score=round(0.9 - i * 0.02, 3),
                           meta=AssetMeta(grain="day", partition_field="dt", partition_latest="2026-11-12", fields=["dt", "dau"])) for i in range(n)]


class LimitBI(FakeBIRetrieval):
    """记录每次检索的 limit，并真的按 limit 返回前 N 个（共 40 个可召回）。"""

    def __init__(self, total=40, fail_on_call=None):
        super().__init__()
        self.limits: list[int] = []
        self.queries: list[str] = []
        self.pool, self.fail_on_call = assets(total), fail_on_call

    async def search(self, query, *, limit=10, cookie=None, params=None):
        self.limits.append(limit)
        self.queries.append(query)
        if self.fail_on_call is not None and len(self.limits) == self.fail_on_call:
            raise RuntimeError("down")
        return [a.model_copy(deep=True) for a in self.pool[:limit]]

    async def get(self, ids):
        return [a.model_copy(deep=True) for a in self.pool if a.asset_id in ids]


def plan(limit=None, query="DAU 日活", angles=()):
    item = {"metric": "DAU", "query": query, "angles": list(angles)}
    if limit is not None:
        item["limit"] = limit
    return {"plans": [item]}


def coverage(key="assets:DAU", sufficient=True, more=None, reason="x"):
    c = {"card": key, "sufficient": sufficient, "reason": reason}
    if more is not None:
        c["more"] = more
    return c


def annotate_reply(prompt_or_none=None, *, cov=(), judge_all_relevant=True):
    def f(prompt):
        shown = [a for a in (f"dataset:{i}" for i in range(60)) if f'"asset_id": "{a}"' in prompt]
        return {"judgements": [{"card": "assets:DAU", "asset_id": a, "relevant": judge_all_relevant, "reason": "ok", "confidence": 0.8} for a in shown],
                "coverage": list(cov)}
    return f


async def go(llm, bi=None, **settings):
    bi = bi or LimitBI()
    _, rec = await run_graph([(BI, ["DAU"])], Deps(bi=bi, llm=llm, settings=cfg(**settings)))
    return bi, rec


# ── 召回多少条：由 LLM 在改写 query 时一并决定 ──────────────────────────────────────
async def test_the_llm_decides_how_many_to_recall_and_the_channel_gets_that_limit():
    bi, _ = await go(ScriptedLLM(plan=plan(limit=18), annotate=annotate_reply(cov=[coverage()])))
    assert bi.limits == [18]


async def test_without_a_limit_the_default_top_k_is_used():
    bi, _ = await go(ScriptedLLM(plan=plan(limit=None), annotate=annotate_reply(cov=[coverage()])))
    assert bi.limits == [10]


@pytest.mark.parametrize("asked,expected", [(500, 30), (1, 3), (0, 3), (-4, 3), (12.0, 12)])
async def test_the_limit_is_clamped_to_the_configured_range(asked, expected):
    bi, _ = await go(ScriptedLLM(plan=plan(limit=asked), annotate=annotate_reply(cov=[coverage()])))
    assert bi.limits == [expected]


@pytest.mark.parametrize("asked", ["many", None, True, [5], {"n": 5}])
async def test_a_non_numeric_limit_is_ignored(asked):
    p = plan(limit=None)
    p["plans"][0]["limit"] = asked
    bi, _ = await go(ScriptedLLM(plan=p, annotate=annotate_reply(cov=[coverage()])))
    assert bi.limits == [10]


async def test_the_range_is_configurable():
    bi, _ = await go(ScriptedLLM(plan=plan(limit=9), annotate=annotate_reply(cov=[coverage()])), RECALL_MIN_LIMIT=5, RECALL_MAX_LIMIT=7)
    assert bi.limits == [7]


async def test_plan_prompt_explains_the_limit_and_its_range():
    llm = ScriptedLLM(plan=plan(limit=10), annotate=annotate_reply(cov=[coverage()]))
    await go(llm)
    p = llm.of("plan")[0]
    assert '"limit"' in p and "3" in p and "30" in p and "{{" not in p


# ── 召回后发现不够：继续召回 ──────────────────────────────────────────────────────────
async def test_insufficient_results_trigger_a_bigger_recall_and_only_the_new_assets_are_judged():
    llm = ScriptedLLM(plan=plan(limit=5), annotate=lambda p: annotate_reply(cov=[coverage(sufficient=False, more={"limit": 12}, reason="只有 5 条，不足以覆盖不同口径")])(p)
                      if '"asset_id": "dataset:11"' not in p else annotate_reply(cov=[coverage()])(p))
    bi, rec = await go(llm)
    assert bi.limits == [5, 12]
    first, second = llm.of("annotate")
    assert '"asset_id": "dataset:0"' in first and '"asset_id": "dataset:5"' not in first
    assert '"asset_id": "dataset:0"' not in second and '"asset_id": "dataset:5"' in second and '"asset_id": "dataset:11"' in second     # 第二次只评估新增的
    assert len({c.asset_id for c in rec.candidates}) == 12
    assert rec.collab.supplements == ["assets:DAU"] and rec.collab.retrievals.count("assets:DAU") == 1 and rec.collab.stats["supplements"] == 1
    card = next(c for c in rec.cards if c.key == "assets:DAU")
    assert len(card.asset_ids) == 12                                                       # 证据卡更新为更大的结果集
    judged = [e for e in rec.evidences if e.source == "llm_filter"]
    assert len({e.asset_id for e in judged}) == 12 and len(judged) == 12                   # 每个资产只被评估一次


async def test_the_annotate_prompt_tells_the_llm_the_current_limit_and_the_cap():
    llm = ScriptedLLM(plan=plan(limit=5), annotate=annotate_reply(cov=[coverage()]))
    await go(llm)
    p = llm.of("annotate")[0]
    assert '"limit": 5' in p and "sufficient" in p and "more" in p and "30" in p


async def test_a_sufficient_verdict_stops_without_recalling_more():
    bi, rec = await go(ScriptedLLM(plan=plan(limit=5), annotate=annotate_reply(cov=[coverage(sufficient=True)])))
    assert bi.limits == [5] and rec.collab.supplements == []


async def test_continuation_is_bounded_by_the_max_supplements():
    llm = ScriptedLLM(plan=plan(limit=3), annotate=annotate_reply(cov=[coverage(sufficient=False)]))
    bi, rec = await go(llm, RECALL_MAX_SUPPLEMENTS=2)
    assert len(bi.limits) == 3 and rec.collab.supplements == ["assets:DAU", "assets:DAU"]          # 首次 + 2 次继续
    assert bi.limits == sorted(bi.limits) and bi.limits[-1] <= 30 and len(set(bi.limits)) == 3
    assert len(llm.of("annotate")) == 3


async def test_zero_supplements_disables_the_continuation():
    bi, _ = await go(ScriptedLLM(plan=plan(limit=3), annotate=annotate_reply(cov=[coverage(sufficient=False)])), RECALL_MAX_SUPPLEMENTS=0)
    assert bi.limits == [3]


async def test_without_a_suggested_limit_the_recall_doubles_and_is_capped():
    llm = ScriptedLLM(plan=plan(limit=20), annotate=annotate_reply(cov=[coverage(sufficient=False)]))
    bi, _ = await go(llm, RECALL_MAX_SUPPLEMENTS=3)
    assert bi.limits == [20, 30]                                                              # 20 → 40 被截到 30；已到上限，不再继续


@pytest.mark.parametrize("more", [{"limit": 4}, {"limit": "lots"}, {"limit": 0}, {}, "x", None])
async def test_an_unusable_suggested_limit_falls_back_to_doubling(more):
    llm = ScriptedLLM(plan=plan(limit=8), annotate=annotate_reply(cov=[coverage(sufficient=False, more=more)]))
    bi, _ = await go(llm, RECALL_MAX_SUPPLEMENTS=1)
    assert bi.limits == [8, 16]


async def test_the_continuation_stops_when_more_recall_finds_nothing_new():
    bi = LimitBI(total=5)                                                                      # 总共只有 5 个可召回
    llm = ScriptedLLM(plan=plan(limit=5), annotate=annotate_reply(cov=[coverage(sufficient=False)]))
    _, rec = await go(llm, bi=bi, RECALL_MAX_SUPPLEMENTS=3)
    assert bi.limits == [5, 10] and len(llm.of("annotate")) == 1                               # 多召回没有新增 → 不再评估、不再继续
    assert rec.role_reports[0].status == "ok"


async def test_the_continuation_may_use_a_rewritten_query():
    llm = ScriptedLLM(plan=plan(limit=5, query="DAU 日活"), annotate=lambda p: {"judgements": [], "coverage": [coverage(sufficient=False, more={"limit": 8, "query": "DAU 日活 各渠道明细"})]}
                      if '"asset_id": "dataset:7"' not in p else {"judgements": [], "coverage": [coverage()]})
    bi, _ = await go(llm)
    assert bi.queries == ["DAU 日活", "DAU 日活 各渠道明细"] and bi.limits == [5, 8]


# ── 降级 / 预算 ───────────────────────────────────────────────────────────────────
async def test_a_failing_supplement_keeps_the_first_card_and_the_node_succeeds():
    bi = LimitBI(fail_on_call=2)
    llm = ScriptedLLM(plan=plan(limit=5), annotate=annotate_reply(cov=[coverage(sufficient=False)]))
    _, rec = await go(llm, bi=bi)
    assert rec.role_reports[0].status == "ok" and len(next(c for c in rec.cards if c.key == "assets:DAU").asset_ids) == 5
    assert rec.collab.supplements == []


async def test_the_total_llm_budget_also_stops_the_continuation():
    llm = ScriptedLLM(plan=plan(limit=3), annotate=annotate_reply(cov=[coverage(sufficient=False)]))
    bi, rec = await go(llm, RECALL_LLM_MAX_CALLS=2, RECALL_MAX_SUPPLEMENTS=5)                  # plan + 一次 annotate 就用完了预算
    assert bi.limits == [3] and rec.collab.llm.calls == 2 and rec.role_reports[0].status == "ok"      # 预算已用完，补召回的结果没法评估，就不补


async def test_flag_off_or_failing_annotate_never_continues():
    bi, _ = await go(ScriptedLLM(plan=plan(limit=5), annotate=UpstreamError("llm", "x")))
    assert bi.limits == [5]
    bi2, _ = await go(ScriptedLLM(plan=plan(limit=5), annotate=annotate_reply(cov=[coverage(sufficient=False)])), RECALL_LLM_ENABLED=False)
    assert bi2.limits == [10]


async def test_coverage_entries_for_unknown_cards_are_ignored():
    bi, _ = await go(ScriptedLLM(plan=plan(limit=5), annotate=annotate_reply(cov=[coverage(key="assets:GHOST", sufficient=False), {"sufficient": False}, "x"])))
    assert bi.limits == [5]


# ── 与换角度重试的关系 ──────────────────────────────────────────────────────────────
async def test_all_irrelevant_results_retry_another_angle_instead_of_a_bigger_recall():
    llm = ScriptedLLM(plan=plan(limit=5, angles=["日活用户"]),
                      annotate=lambda p: annotate_reply(judge_all_relevant=False, cov=[coverage(sufficient=False)])(p))
    bi, rec = await go(llm, RECALL_MAX_SUPPLEMENTS=2)
    assert bi.queries[0] == "DAU 日活" and "日活用户" in bi.queries and rec.collab.supplements == []


async def test_the_continuation_is_scoped_per_card():
    llm = ScriptedLLM(plan={"plans": [{"metric": "DAU", "query": "DAU 日活", "limit": 4}, {"metric": "GMV", "query": "GMV 交易", "limit": 4}]},
                      annotate=lambda p: {"judgements": [], "coverage": [coverage("assets:DAU", sufficient='"asset_id": "dataset:4"' in p), coverage("assets:GMV", True)]})
    bi = LimitBI()
    _, rec = await run_graph([(BI, ["DAU", "GMV"])], Deps(bi=bi, llm=llm, settings=cfg()), query="DAU和GMV")
    assert sorted(bi.limits) == [4, 4, 8] and rec.collab.supplements == ["assets:DAU"]
