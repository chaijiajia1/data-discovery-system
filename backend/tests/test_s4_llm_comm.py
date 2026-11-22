"""④ 通信：画像路由仍是确定性的；被点名 / 揭榜时的「语义胜任」判断与求助问题的撰写由 LLM 完成，硬规则不动，LLM 不可用回落规则。"""
import asyncio

import pytest

from app.collab.hub import build_hub
from app.deps import Deps
from app.errors import UpstreamError
from app.roles import REGISTRY, build_agents
from app.schemas.collab import Ask, AskMode, Need, ReplyKind
from app.schemas.intent import Intent
from app.schemas.task import Slot, TaskContract, ValueSource
from app.stages.s3_planning.candidates import make_node
from tests.test_s4_llm_roles import BI, CAL, LN, OMNI, ScriptedLLM, SpyBI, cfg, run_graph


def nodes(*specs):
    return [make_node(REGISTRY[r].spec, list(t)) for r, t in specs]


def make_hub(specs, llm=None, bi=None, **settings):
    deps = Deps(llm=llm, settings=cfg(**settings), **({"bi": bi} if bi is not None else {})) if llm is not None else Deps(settings=cfg(RECALL_LLM_ENABLED=False, **settings))
    contract = TaskContract(query="最近7天DAU", metrics=Slot(value=["DAU"], source=ValueSource.USER), intent=Intent(query="q", domain="transaction"))
    return build_hub(build_agents(nodes(*specs)), deps, contract=contract, knowledge=[], cookie=None, recall_params={})


def assets_need(subject="DAU", **kw):
    return Need(kind="assets", subject=subject, domains=kw.pop("domains", ["transaction"]), asset_type="dataset", **kw)


def verdict(accept, reason="", fit=0.8, defer_to=None):
    d = {"accept": accept, "fit": fit, "reason": reason}
    if defer_to is not None:
        d["defer_to"] = defer_to
    return d


LN_DAU, BI_GMV = f"{LN}@DAU", f"{BI}@GMV"


# ── 定向询问：被点名的角色用 LLM 判断语义上是否胜任 ──────────────────────────────────
async def test_directed_ask_llm_decline_is_a_reasoned_decline_and_downgrades_to_broadcast():
    llm = ScriptedLLM(selfeval=verdict(False, "全通道角色同时覆盖 BI 与口径，更适合它统一处理这个请求", defer_to=f"{OMNI}@DAU"))
    hub = make_hub([(LN, ["DAU"]), (BI, ["GMV"]), (OMNI, ["DAU"])], llm)
    res = await hub.obtain(assets_need(), requester=LN_DAU, who=LN_DAU, produce=None)
    rec = hub.trace.asks[0]
    assert rec.route.mode == AskMode.DIRECTED and rec.target_agent == BI_GMV
    declines = [r for r in rec.replies if r.kind == ReplyKind.DECLINE]
    assert declines and declines[0].reason.startswith("llm_decline") and "全通道角色" in declines[0].reason and f"{OMNI}@DAU" in declines[0].reason
    assert any(r.kind == ReplyKind.DOWNGRADE for r in rec.replies)


async def test_directed_ask_llm_accept_serves_the_request():
    bi = SpyBI()
    llm = ScriptedLLM(selfeval=verdict(True, "BI 看板正好承载 DAU"))
    hub = make_hub([(LN, ["DAU"]), (BI, ["GMV"])], llm, bi=bi)
    res = await hub.obtain(assets_need(), requester=LN_DAU, who=LN_DAU, produce=None)
    rec = hub.trace.asks[0]
    assert rec.outcome == "adopted" and rec.served_by == BI_GMV and res.card is not None and bi.calls == ["DAU"]
    p = llm.of("selfeval")[0]
    assert "BI 资产发现" in p and "黑板协作协议" in p and "DAU" in p and BI_GMV in p and "{{" not in p


async def test_self_evaluation_failure_falls_back_to_the_rule_check():
    llm = ScriptedLLM(selfeval=UpstreamError("llm", "boom"))
    hub = make_hub([(LN, ["DAU"]), (BI, ["GMV"])], llm, bi=SpyBI())
    res = await hub.obtain(assets_need(), requester=LN_DAU, who=LN_DAU, produce=None)
    assert hub.trace.asks[0].outcome == "adopted" and res.card is not None
    assert hub.llm.usage().failures >= 1


async def test_without_llm_the_rule_path_is_unchanged():
    hub = make_hub([(LN, ["DAU"]), (BI, ["GMV"])], None, bi=SpyBI())
    await hub.obtain(assets_need(), requester=LN_DAU, who=LN_DAU, produce=None)
    assert hub.trace.asks[0].outcome == "adopted" and hub.llm.usage().calls == 0


async def test_rule_low_fit_still_applies_when_the_llm_cannot_answer():
    llm = ScriptedLLM(selfeval=UpstreamError("llm", "boom"))
    hub = make_hub([(LN, ["DAU"]), (BI, ["GMV"])], llm, COLLAB_ACCEPT_MIN=0.99)
    ask = Ask(id="a1", from_role=LN_DAU, to_role=BI_GMV, mode=AskMode.DIRECTED, question="q", need=assets_need())
    reply = await hub.agents[BI_GMV].respond(ask, hub)
    assert reply.kind == ReplyKind.DECLINE and reply.reason.startswith("low_fit")


# ── 硬规则不动，也不为它们花 LLM ─────────────────────────────────────────────────
async def test_hard_refusals_never_call_the_llm():
    llm = ScriptedLLM(selfeval=verdict(True))
    hub = make_hub([(BI, ["DAU"])], llm)
    bi = hub.agents[f"{BI}@DAU"]
    out_of_scope = Ask(id="a1", from_role="x", mode=AskMode.DIRECTED, question="q", need=Need(kind="caliber", subject="DAU", domains=["transaction"], asset_type="knowledge"))
    r1 = await bi.respond(out_of_scope, hub)
    bi._active = bi.spec.max_concurrency
    r2 = await bi.respond(Ask(id="a2", from_role="x", mode=AskMode.DIRECTED, question="q", need=assets_need()), hub)
    assert r1.reason.startswith("out_of_scope") and r2.reason.startswith("busy") and llm.of("selfeval") == []
    hub2 = make_hub([("dim_value_resolver", ["x"])], llm)
    r3 = await hub2.agents["dim_value_resolver@x"].respond(Ask(id="a3", from_role="x", mode=AskMode.DIRECTED, question="q", need=Need(kind="dimension", subject="自然堂", domains=["overseas"], asset_type="table")), hub2)
    assert r3.reason.startswith("domain_uncovered") and llm.of("selfeval") == []


# ── 广播：粗筛 → LLM 自评并发 → 自愿揭榜 ─────────────────────────────────────────────
def fit_by_agent(prompt):
    return verdict(True, "我能查", 0.9 if f"{BI}@A" in prompt else 0.5)


async def test_broadcast_bids_are_evaluated_concurrently_and_the_llm_fit_orders_volunteers():
    llm = ScriptedLLM(selfeval=fit_by_agent)
    llm.delay = 0.03
    hub = make_hub([(LN, ["DAU"]), (BI, ["A"]), (BI, ["B"]), (CAL, ["DAU"])], llm, bi=SpyBI(), COLLAB_HANDOFF_ENABLED=False)
    await hub.obtain(assets_need("DAU", domains=["transaction", "community"]), requester=LN_DAU, who=LN_DAU, produce=None)
    rec = hub.trace.asks[0]
    assert rec.route.mode == AskMode.BROADCAST and llm.peak == 2                      # 两个 BI 副本的自评同时进行；口径角色超出能力，硬规则挡掉，不花 LLM
    assert len(llm.of("selfeval")) == 2 and rec.served_by == f"{BI}@A" and rec.outcome == "adopted"
    assert [b.volunteered for b in rec.bids if b.role.startswith(BI)] == [True, True]
    assert next(b for b in rec.bids if b.role == f"{BI}@A").fit == 0.9


async def test_llm_bid_decline_is_a_reasoned_non_volunteer():
    llm = ScriptedLLM(selfeval=verdict(False, "跨域问题，全通道角色覆盖面更广", 0.2, defer_to=f"{OMNI}@X"))
    hub = make_hub([(LN, ["DAU"]), (BI, ["A"]), (OMNI, ["X"])], llm, bi=SpyBI(), COLLAB_HANDOFF_ENABLED=False)
    res = await hub.obtain(assets_need("Z", domains=["transaction", "community"]), requester=LN_DAU, who=LN_DAU, produce=None)
    bid = next(b for b in hub.trace.asks[0].bids if b.role == f"{BI}@A")
    assert not bid.volunteered and bid.reason.startswith("llm_decline") and "覆盖面更广" in bid.reason


async def test_broadcast_coarse_filter_skips_the_llm_for_hopeless_fits():
    llm = ScriptedLLM(selfeval=verdict(True))
    hub = make_hub([(LN, ["DAU"]), (BI, ["A"])], llm, bi=SpyBI(), COLLAB_ACCEPT_MIN=0.99)
    await hub.obtain(assets_need("Z", domains=["transaction", "community"]), requester=LN_DAU, who=LN_DAU, produce=None)
    bid = hub.trace.asks[0].bids[0]
    assert not bid.volunteered and bid.reason.startswith("low_fit") and llm.of("selfeval") == []


async def test_broadcast_with_failing_llm_uses_the_rule_bid():
    llm = ScriptedLLM(selfeval=UpstreamError("llm", "boom"))
    hub = make_hub([(LN, ["DAU"]), (BI, ["A"])], llm, bi=SpyBI())
    res = await hub.obtain(assets_need("Z", domains=["transaction", "community"]), requester=LN_DAU, who=LN_DAU, produce=None)
    assert res.card is not None or hub.trace.asks[0].bids                              # 与无 LLM 时同样的规则自评
    nollm = make_hub([(LN, ["DAU"]), (BI, ["A"])], None, bi=SpyBI())
    await nollm.obtain(assets_need("Z", domains=["transaction", "community"]), requester=LN_DAU, who=LN_DAU, produce=None)
    assert [(b.role, b.volunteered) for b in hub.trace.asks[0].bids] == [(b.role, b.volunteered) for b in nollm.trace.asks[0].bids]


# ── 求助问题由 LLM 撰写；路由不受影响 ─────────────────────────────────────────────────
async def test_ask_question_is_written_by_the_llm_and_routing_is_unchanged():
    q = "最近 7 天交易域里，哪些已认证的看板或数据集承载「DAU」？"
    llm = ScriptedLLM(selfeval=verdict(True), ask={"question": q})
    hub = make_hub([(LN, ["DAU"]), (BI, ["GMV"])], llm, bi=SpyBI())
    await hub.obtain(assets_need(), requester=LN_DAU, who=LN_DAU, produce=None)
    base = make_hub([(LN, ["DAU"]), (BI, ["GMV"])], None, bi=SpyBI())
    await base.obtain(assets_need(), requester=LN_DAU, who=LN_DAU, produce=None)
    rec, brec = hub.trace.asks[0], base.trace.asks[0]
    assert rec.ask.question == q and rec.ask.need.question == q and rec.ask.need.key == "assets:DAU" and rec.need_key == "assets:DAU"
    assert (rec.route.mode, rec.route.target, rec.route.scores) == (brec.route.mode, brec.route.target, brec.route.scores)
    assert q in llm.of("selfeval")[0]                                                  # 被问的角色看到的是自足的问题


async def test_unusable_question_falls_back_to_the_template_text():
    for reply in ({"question": "太短"}, {"question": "这是一个没有提到主题名称的很长的问题而已"}, {"question": 5}, UpstreamError("llm", "x")):
        llm = ScriptedLLM(selfeval=verdict(True), ask=reply)
        hub = make_hub([(LN, ["DAU"]), (BI, ["GMV"])], llm, bi=SpyBI())
        n = assets_need()
        await hub.obtain(n, requester=LN_DAU, who=LN_DAU, produce=None)
        assert hub.trace.asks[0].ask.question == n.text


async def test_handoff_needs_are_already_structured_so_no_llm_is_spent_on_rewriting_them():
    llm = ScriptedLLM(selfeval=verdict(True), ask={"question": "不应该被调用 DAU 的问题"})
    hub = make_hub([(LN, ["DAU"]), (BI, ["GMV"])], llm, bi=SpyBI())
    n = assets_need(via="caliber:DAU", hop=1, hint_assets=["dws.user_active_d"])
    await hub.obtain(n, requester=LN_DAU, who=LN_DAU, produce=None)
    assert llm.of("ask") == [] and hub.trace.asks[0].ask.question == n.text and llm.of("selfeval") == []      # 转交来的需求：不改写问题，也不做 LLM 自评（规则判断）


async def test_an_already_written_question_is_not_rewritten():
    llm = ScriptedLLM(selfeval=verdict(True), ask={"question": "不应该被调用 DAU 的问题"})
    hub = make_hub([(LN, ["DAU"]), (BI, ["GMV"])], llm, bi=SpyBI())
    await hub.obtain(assets_need(question="已经写好的、关于 DAU 的问题"), requester=LN_DAU, who=LN_DAU, produce=None)
    assert llm.of("ask") == [] and hub.trace.asks[0].ask.question == "已经写好的、关于 DAU 的问题"


# ── 预算 ──────────────────────────────────────────────────────────────────────
async def test_budget_exhaustion_midway_falls_back_to_rules_without_breaking_the_ask():
    llm = ScriptedLLM(selfeval=verdict(False, "拒绝", defer_to=f"{OMNI}@X"), ask={"question": "关于 DAU 的自足问题，请问哪些资产承载它？"})
    hub = make_hub([(LN, ["DAU"]), (BI, ["GMV"])], llm, bi=SpyBI(), RECALL_LLM_MAX_CALLS=1)
    await hub.obtain(assets_need(), requester=LN_DAU, who=LN_DAU, produce=None)       # 唯一一次调用被用来写问题，自评回落规则 → 接活
    rec = hub.trace.asks[0]
    assert [k for k, _ in llm.calls] == ["ask"] and rec.outcome == "adopted" and hub.llm.usage().budget_exhausted


# ── 端到端 ─────────────────────────────────────────────────────────────────────
async def test_end_to_end_ask_goes_through_llm_question_and_self_evaluation():
    llm = ScriptedLLM(selfeval=verdict(True, "可以"), ask={"question": "请问哪些资产承载「DAU」这个指标？"})
    _, rec = await run_graph([(LN, ["DAU"]), (BI, ["GMV"])], Deps(bi=SpyBI(), llm=llm, settings=cfg()), query="最近7天DAU和GMV")
    assert [r.status for r in rec.role_reports] == ["ok", "ok"] and rec.collab.asks[0].outcome == "adopted"
    u = rec.collab.llm
    assert u.by_purpose.get("ask") == 1 and u.by_purpose.get("selfeval") == 1 and u.calls <= u.max_calls


# ── 自评：知道在场的其他角色；婉拒必须点名一个在场且能服务该需求的角色 ──────────────────────────────
async def test_selfeval_prompt_lists_present_peers_and_states_the_deference_rule():
    llm = ScriptedLLM(selfeval=verdict(True, "可以"))
    hub = make_hub([(LN, ["DAU"]), (BI, ["GMV"]), (OMNI, ["DAU"])], llm, bi=SpyBI())
    await hub.obtain(assets_need(), requester=LN_DAU, who=LN_DAU, produce=None)
    p = llm.of("selfeval")[0]
    assert "在场的其他 agent" in p and f"{OMNI}@DAU" in p and "defer_to" in p and "没有更合适的" in p
    assert f"{LN_DAU}" in p                                                              # 请求方也是在场者（它不能接自己的求助，但模型应知道谁在问）
    assert "{{" not in p


async def test_prompt_marks_which_peers_can_serve_this_kind_of_need():
    llm = ScriptedLLM(selfeval=verdict(True, "可以"))
    hub = make_hub([(LN, ["DAU"]), (BI, ["GMV"]), (CAL, ["DAU"])], llm, bi=SpyBI())
    await hub.obtain(assets_need(), requester=LN_DAU, who=LN_DAU, produce=None)
    import json as _j, re
    block = re.search(r"```json\n(.*?)\n```", llm.of("selfeval")[0], re.S)
    peers = {x["agent_id"]: x for x in _j.loads(block.group(1))["peers"]} if block else {}
    assert peers[f"{CAL}@DAU"]["can_serve_this_need"] is False and BI_GMV not in peers          # 自己不在同伴里；口径角色不服务 assets
    assert peers[LN_DAU]["can_serve_this_need"] is False


async def test_decline_without_a_qualified_peer_is_invalid_and_the_rule_decides():
    """没有任何在场的同伴能服务这类需求时，「更适合别人」不成立：婉拒无效，回落规则（画像匹配度够 → 接活）。"""
    llm = ScriptedLLM(selfeval=verdict(False, "这个问题更适合专门的口径角色", defer_to=CAL + "@DAU"))
    hub = make_hub([(LN, ["DAU"]), (BI, ["GMV"]), (CAL, ["DAU"])], llm, bi=SpyBI())
    res = await hub.obtain(assets_need(), requester=LN_DAU, who=LN_DAU, produce=None)
    assert hub.trace.asks[0].outcome == "adopted" and res.card is not None


@pytest.mark.parametrize("defer_to", [None, "", "ghost@X", "bi_asset_discovery@GMV", "lineage_discovery@DAU", 5])
async def test_decline_must_name_a_present_peer_that_serves_the_need(defer_to):
    """defer_to 缺失 / 不在场 / 是自己 / 是请求方 / 不服务该类需求 → 婉拒无效。"""
    llm = ScriptedLLM(selfeval=verdict(False, "更适合别的角色", defer_to=defer_to))
    hub = make_hub([(LN, ["DAU"]), (BI, ["GMV"]), (OMNI, ["DAU"])], llm, bi=SpyBI(), COLLAB_HANDOFF_ENABLED=False)
    res = await hub.obtain(assets_need(), requester=LN_DAU, who=LN_DAU, produce=None)
    assert hub.trace.asks[0].outcome == "adopted" and res.card is not None and not any(r.kind == ReplyKind.DECLINE for r in hub.trace.asks[0].replies)


async def test_a_concrete_deference_is_honoured_and_the_named_peer_can_pick_it_up_on_broadcast():
    llm = ScriptedLLM(selfeval=lambda p: verdict(False, "全通道角色已持有该口径的相关证据", 0.2, defer_to=f"{OMNI}@DAU") if f"{BI}@GMV" in p.split("你当前的状态")[1][:200] else verdict(True, "可以", 0.9))
    hub = make_hub([(LN, ["DAU"]), (BI, ["GMV"]), (OMNI, ["DAU"])], llm, bi=SpyBI(), COLLAB_HANDOFF_ENABLED=False)
    res = await hub.obtain(assets_need(), requester=LN_DAU, who=LN_DAU, produce=None)
    rec = hub.trace.asks[0]
    assert any(r.kind == ReplyKind.DECLINE and f"{OMNI}@DAU" in r.reason for r in rec.replies)
    assert rec.served_by == f"{OMNI}@DAU" and res.card is not None


async def test_accepting_needs_no_defer_to_and_a_vague_accept_reason_is_fine():
    llm = ScriptedLLM(selfeval=verdict(True, ""))
    hub = make_hub([(LN, ["DAU"]), (BI, ["GMV"])], llm, bi=SpyBI())
    res = await hub.obtain(assets_need(), requester=LN_DAU, who=LN_DAU, produce=None)
    assert res.card is not None


# ── 控制成本：转交来的需求不花 LLM 自评；能自己做的转交不再询问同类角色；已解析的资产不再转交 ─────────────────────
async def test_handoff_needs_are_judged_by_rules_only_even_for_directed_asks_and_broadcast_bids():
    llm = ScriptedLLM(selfeval=verdict(False, "拒绝", defer_to=f"{OMNI}@DAU"))
    hub = make_hub([(LN, ["DAU"]), (BI, ["GMV"]), (OMNI, ["DAU"])], llm, bi=SpyBI(), COLLAB_HANDOFF_ENABLED=False)
    res = await hub.obtain(assets_need(via="caliber:DAU", hop=1), requester=LN_DAU, who=LN_DAU, produce=None)
    assert llm.of("selfeval") == [] and res.card is not None and hub.trace.asks[0].outcome == "adopted"       # 规则判断：接
    hub2 = make_hub([(LN, ["DAU"]), (BI, ["A"]), (BI, ["B"])], llm, bi=SpyBI(), COLLAB_HANDOFF_ENABLED=False)
    await hub2.obtain(assets_need("Z", domains=["transaction", "community"], via="caliber:DAU", hop=1), requester=LN_DAU, who=LN_DAU, produce=None)
    assert llm.of("selfeval") == [] and all("llm" not in b.reason for b in hub2.trace.asks[0].bids)             # 广播揭榜也只用规则


async def test_rule_decline_still_applies_to_handoff_needs():
    llm = ScriptedLLM(selfeval=verdict(True))
    hub = make_hub([(LN, ["DAU"]), (BI, ["GMV"])], llm, bi=SpyBI(), COLLAB_ACCEPT_MIN=0.99, COLLAB_HANDOFF_ENABLED=False)
    ask = Ask(id="a", from_role=LN_DAU, to_role=BI_GMV, mode=AskMode.DIRECTED, question="q", need=assets_need(via="caliber:DAU", hop=1))
    r = await hub.agents[BI_GMV].respond(ask, hub)
    assert r.kind == ReplyKind.DECLINE and r.reason.startswith("low_fit") and llm.of("selfeval") == []


async def test_own_needs_still_get_the_llm_self_evaluation():
    llm = ScriptedLLM(selfeval=verdict(True, "可以"))
    hub = make_hub([(LN, ["DAU"]), (BI, ["GMV"])], llm, bi=SpyBI(), COLLAB_HANDOFF_ENABLED=False)
    await hub.obtain(assets_need(), requester=LN_DAU, who=LN_DAU, produce=None)
    assert len(llm.of("selfeval")) == 1


async def test_an_agent_that_can_serve_the_follow_up_itself_does_not_ask_a_peer_of_the_same_kind():
    llm = ScriptedLLM(selfeval=verdict(True))
    _, rec = await run_graph([(OMNI, ["DAU"]), (OMNI, ["GMV"])], Deps(bi=SpyBI(), llm=llm, settings=cfg()), query="最近7天DAU和GMV")
    assert [a for a in rec.collab.asks if a.ask.need.via] == []                          # 全通道 agent 的转交自己顺着查，不再逐条询问另一个全通道 agent
    assert any(c.key.startswith(("lineage:", "caliber:", "assets:")) and c.key.split(":")[1] not in ("DAU", "GMV") for c in rec.cards)        # 但线索仍被追下去了
    assert len(llm.of("selfeval")) == 0


async def test_specialists_still_ask_the_role_that_can_do_what_they_cannot():
    hub = make_hub([(LN, ["DAU"]), (BI, ["GMV"])], None, bi=SpyBI())
    await hub.delegate(LN_DAU, [Need(kind="assets", subject="X", domains=["transaction"], asset_type="dataset", via="lineage:DAU", hop=1)])
    assert hub.trace.asks and hub.trace.asks[0].served_role == BI and hub.info[LN_DAU].handoffs == 1


async def test_handoff_count_only_counts_real_hand_offs_to_other_agents():
    hub = make_hub([(OMNI, ["DAU"]), (OMNI, ["GMV"])], None, bi=SpyBI())
    await hub.delegate(f"{OMNI}@DAU", [Need(kind="assets", subject="X", domains=["transaction"], asset_type="dataset", via="caliber:DAU", hop=1)])
    assert hub.trace.asks == [] and hub.info[f"{OMNI}@DAU"].handoffs == 0 and "assets:X" in hub.board.cards
