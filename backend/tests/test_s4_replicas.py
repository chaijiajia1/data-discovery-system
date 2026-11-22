"""④ 同一角色的多个副本：每个节点一个独立 agent，并行执行，并通过黑板 / 定向询问 / 广播沟通。"""
import asyncio
import time

import pytest

from app.collab.hub import build_hub
from app.deps import Deps
from app.roles import REGISTRY, build_agents, build_roles
from app.schemas.collab import AskMode, Need, ReplyKind
from app.schemas.intent import Intent
from app.schemas.task import Slot, TaskContract, ValueSource
from app.schemas.workflow import EdgeKind, WorkflowEdge, WorkflowGraph
from app.services.fakes import FakeBIRetrieval, FakeLineageRetrieval
from app.stages import RunContext
from app.stages.s2_coordination import CoordinationStage
from app.stages.s3_planning import PlanningStage
from app.stages.s3_planning.candidates import make_node
from app.stages.s4_recall import RecallStage

BI, LN, CAL = "bi_asset_discovery", "lineage_discovery", "caliber_lookup"


def _contract(*metrics):
    return TaskContract(query="q", metrics=Slot(value=list(metrics), source=ValueSource.USER), intent=Intent(query="q", domain="transaction"))


def nodes(*specs):
    return [make_node(REGISTRY[r].spec, list(t)) for r, t in specs]


def test_build_agents_gives_each_node_its_own_replica_with_identity_and_load():
    ns = nodes((BI, ["DAU"]), (BI, ["GMV"]), (LN, ["DAU"]))
    ags = build_agents(ns)
    a, b, c = (ags[n.id] for n in ns)
    assert a is not b and a.agent_id == "bi_asset_discovery@DAU" and b.agent_id == "bi_asset_discovery@GMV" and c.agent_id == "lineage_discovery@DAU"
    assert (a.replica, a.replicas, b.replica, b.replicas, c.replica, c.replicas) == (1, 2, 2, 2, 1, 1)
    assert a.targets == ["DAU"] and a.spec.name == BI
    a._active = 3
    assert b._active == 0                                                                  # 负载互不影响


def test_build_roles_keeps_name_keyed_single_instances_for_compat():
    r = build_roles([BI, LN])
    assert set(r) == {BI, LN} and r[BI].agent_id == BI and (r[BI].replica, r[BI].replicas) == (1, 1)


def hub_with(ns, **kw):
    deps = kw.pop("deps", None) or Deps()
    if kw:
        deps.settings = deps.settings.model_copy(update=kw)
    return build_hub(build_agents(ns), deps, contract=_contract("DAU", "GMV"), knowledge=[], cookie=None, recall_params={})


def need(kind, s):
    return Need(kind=kind, subject=s, domains=["transaction"], asset_type={"assets": "dataset", "lineage": "table"}[kind])


async def test_directed_ask_goes_to_the_replica_that_owns_the_need_not_just_any_replica_of_the_role():
    ns = nodes((BI, ["DAU"]), (BI, ["GMV"]), (LN, ["GMV"]))
    hub = hub_with(ns, COLLAB_WAIT_SEC=0.05)
    await hub.board.try_claim("bi_asset_discovery@GMV", "assets:GMV")                      # GMV 副本正在探查 assets:GMV
    res = await hub.obtain(need("assets", "GMV"), requester="lineage_discovery@GMV", who="lineage_discovery@GMV", produce=None)
    rec = hub.trace.asks[0]
    assert rec.route.mode == AskMode.DIRECTED and rec.route.target == BI                  # 路由选的是角色（画像是角色级的）
    assert rec.target_agent == "bi_asset_discovery@GMV" and rec.requester_role == LN        # 角色内再选副本：持有该需求的那个
    assert res.how in ("ask", "reuse_wait") or res.card is not None


async def test_directed_ask_prefers_the_idle_replica_when_nobody_owns_the_need():
    ns = nodes((BI, ["DAU"]), (BI, ["GMV"]), (LN, ["CTR"]))
    hub = hub_with(ns)
    hub.agents["bi_asset_discovery@DAU"]._active = 2                                         # DAU 副本在忙别的询问
    await hub.obtain(need("assets", "CTR"), requester="lineage_discovery@CTR", who="lineage_discovery@CTR", produce=None)
    assert hub.trace.asks[0].target_agent == "bi_asset_discovery@GMV" and hub.trace.asks[0].served_by == "bi_asset_discovery@GMV"
    assert hub.trace.asks[0].served_role == BI


async def test_busy_replica_declines_and_a_peer_replica_of_the_same_role_picks_it_up():
    ns = nodes((BI, ["DAU"]), (BI, ["GMV"]), (LN, ["CTR"]))
    hub = hub_with(ns, COLLAB_WAIT_SEC=0.05)
    owner = "bi_asset_discovery@DAU"
    hub.agents[owner]._active = hub.agents[owner].spec.max_concurrency                       # 把「首选副本」占满
    hub.agents["bi_asset_discovery@GMV"]._active = 0
    # 让首选副本确定为 DAU：DAU 副本持有该需求
    await hub.board.try_claim(owner, "assets:CTR")
    res = await hub.obtain(need("assets", "CTR"), requester="lineage_discovery@CTR", who="lineage_discovery@CTR", produce=None)
    rec = hub.trace.asks[0]
    kinds = [r.kind for r in rec.replies]
    assert rec.target_agent == owner and kinds[:2] == [ReplyKind.DECLINE, ReplyKind.DOWNGRADE] and rec.replies[0].reason.startswith("busy")
    assert rec.served_by == "bi_asset_discovery@GMV" and rec.served_role == BI              # 同角色的另一个副本揭榜并回答
    assert owner not in [b.role for b in rec.bids] and any(b.role == "bi_asset_discovery@GMV" and b.volunteered for b in rec.bids)
    assert res.card is not None and hub.registry.get(BI).stats.declined == 1               # 画像是角色级的：婉拒记在角色上


async def test_broadcast_ties_are_broken_by_ownership_then_load_then_id():
    ns = nodes((BI, ["A"]), (BI, ["B"]), (BI, ["C"]), (LN, ["Z"]))
    hub = hub_with(ns)
    hub.agents["bi_asset_discovery@A"]._active = 1
    n = Need(kind="assets", subject="Z", domains=["transaction", "community"], asset_type="dataset")           # 跨域 → 广播
    await hub.obtain(n, requester="lineage_discovery@Z", who="lineage_discovery@Z", produce=None)
    rec = hub.trace.asks[0]
    assert rec.route.mode == AskMode.BROADCAST and rec.served_by == "bi_asset_discovery@B"        # A 有负载，B/C 并列 → 按 id


async def test_requester_never_asks_itself_but_may_ask_a_same_role_peer():
    ns = nodes((BI, ["DAU"]), (BI, ["GMV"]))
    hub = hub_with(ns)
    n = Need(kind="assets", subject="X", domains=["transaction"], asset_type="dataset")
    await hub.obtain(n, requester="bi_asset_discovery@DAU", who="bi_asset_discovery@DAU", produce=None)
    rec = hub.trace.asks[0]
    assert rec.target_agent in (None, "bi_asset_discovery@GMV") and rec.served_by in (None, "bi_asset_discovery@GMV")
    assert all(r.from_role != "bi_asset_discovery@DAU" for r in rec.replies)


# ── 端到端：副本并行、重复探查被协调、轨迹可见 ───────────────────────────────────
class SpyBI(FakeBIRetrieval):
    def __init__(self, delay=0.08):
        super().__init__()
        self.calls, self.delay = [], delay

    async def search(self, query, **kw):
        i = len(self.calls)
        t0 = time.perf_counter()
        self.calls.append((query, t0, None))
        await asyncio.sleep(self.delay)
        self.calls[i] = (query, t0, time.perf_counter())
        return await super().search(query, limit=5)


async def run_graph(specs, deps, query="DAU和GMV", edges=(), a_max=8):
    ctx = RunContext(query=query, deps=deps)
    ctx.contract = await CoordinationStage().run(ctx)
    ctx.plan = await PlanningStage().run(ctx)
    g = WorkflowGraph(nodes=nodes(*specs), edges=[WorkflowEdge(src=a, dst=b, kind=EdgeKind.DEP) for a, b in edges])
    ctx.plan = ctx.plan.model_copy(update={"chosen": ctx.plan.chosen.model_copy(update={"graph": g}), "max_parallel": a_max, "known_assets": []})
    return ctx, await RecallStage().run(ctx)


async def test_same_role_replicas_run_in_parallel():
    bi = SpyBI()
    t0 = time.perf_counter()
    _, rec = await run_graph([(BI, ["DAU"]), (BI, ["GMV"]), (BI, ["CTR"])], Deps(bi=bi), query="DAU和GMV和CTR")
    wall = time.perf_counter() - t0
    spans = {q: (s, e) for q, s, e in bi.calls}
    assert set(spans) == {"DAU", "GMV", "CTR"}
    assert all(spans["DAU"][0] < spans[o][1] and spans[o][0] < spans["DAU"][1] for o in ("GMV", "CTR"))      # 三个副本的探查在时间上重叠
    assert wall < 0.08 * 3 * 0.8                                                                         # 并行：总时长远小于串行之和
    assert [r.status for r in rec.role_reports] == ["ok"] * 3 and len({r.node_id for r in rec.role_reports}) == 3


async def test_replicas_exchange_work_through_the_blackboard_without_duplicate_probes():
    bi = SpyBI()
    ctx, rec = await run_graph([(BI, ["DAU", "GMV"]), (BI, ["GMV"]), (BI, ["DAU"])], Deps(bi=bi))
    assert sorted(q for q, _, _ in bi.calls) == ["DAU", "GMV"]                                              # 三个副本、五个口径任务，只探查两次
    assert rec.collab.retrievals.count("assets:DAU") == 1 and rec.collab.retrievals.count("assets:GMV") == 1
    st = rec.collab.summarize()
    assert st["reuse_hits"] + st["reuse_waited"] >= 3 and st["reuse_waited"] >= 1                           # 有副本等到了同伴的证据卡
    authors = {e.author for e in rec.collab.reuse}
    assert authors <= {"bi_asset_discovery@DAU+GMV", "bi_asset_discovery@GMV", "bi_asset_discovery@DAU"}


async def test_batched_replica_works_on_unclaimed_needs_first_instead_of_blocking_on_a_peer():
    bi = SpyBI(delay=0.1)
    t0 = time.perf_counter()
    _, rec = await run_graph([(BI, ["GMV"]), (BI, ["GMV", "DAU"])], Deps(bi=bi))
    order = [q for q, _, _ in sorted(bi.calls, key=lambda x: x[1])]
    assert sorted(order) == ["DAU", "GMV"] and time.perf_counter() - t0 < 0.1 * 2 * 0.9                 # 批量副本先做没人认领的 DAU，同时同伴在做 GMV
    assert [r.status for r in rec.role_reports] == ["ok", "ok"]


async def test_lineage_replicas_ask_the_matching_bi_replica_and_trace_names_agents_and_replicas():
    bi = SpyBI(delay=0.05)
    _, rec = await run_graph([(BI, ["DAU"]), (BI, ["GMV"]), (LN, ["DAU"]), (LN, ["GMV"])], Deps(bi=bi), edges=[(f"{BI}@DAU", f"{LN}@DAU"), (f"{BI}@GMV", f"{LN}@GMV")])
    infos = {a.agent_id: a for a in rec.collab.agents}
    assert set(infos) == {"bi_asset_discovery@DAU", "bi_asset_discovery@GMV", "lineage_discovery@DAU", "lineage_discovery@GMV"}
    assert (infos["bi_asset_discovery@DAU"].replica, infos["bi_asset_discovery@DAU"].replicas) == (1, 2) and infos[f"{LN}@GMV"].replicas == 2
    assert infos["bi_asset_discovery@DAU"].targets == ["DAU"] and infos["bi_asset_discovery@GMV"].produced == ["assets:GMV"]
    assert {e.by for e in rec.collab.reuse} >= {f"{LN}@DAU", f"{LN}@GMV"}                                  # 血缘副本复用各自对应 BI 副本的证据卡
    assert bi.calls and len(bi.calls) == 2


async def test_busy_preferred_replica_fails_over_end_to_end_and_all_nodes_still_finish():
    bi = SpyBI(delay=0.03)
    deps = Deps(bi=bi)
    deps.settings = deps.settings.model_copy(update={"COLLAB_WAIT_SEC": 2.0})
    _, rec = await run_graph([(LN, ["DAU"]), (BI, ["GMV"]), (BI, ["DAU"])], deps, edges=[(f"{LN}@DAU", f"{BI}@DAU")])   # 血缘先于 BI@DAU 运行，必须求助
    assert [r.status for r in rec.role_reports] == ["ok"] * 3
    assert rec.collab.asks and rec.collab.asks[0].outcome == "adopted" and rec.collab.asks[0].served_role == BI
    assert rec.collab.retrievals.count("assets:DAU") == 1


async def test_serial_a_max_with_replicas_does_not_deadlock():
    _, rec = await asyncio.wait_for(run_graph([(LN, ["DAU"]), (BI, ["GMV"]), (BI, ["DAU"])], Deps(bi=SpyBI(0.01)), a_max=1,
                                              edges=[(f"{LN}@DAU", f"{BI}@DAU")]), timeout=10)
    assert [r.status for r in rec.role_reports] == ["ok"] * 3


# ── 协作事件流：供前端泳道图按时间顺序绘制 ───────────────────────────────────────
async def test_events_are_ordered_and_reference_only_agents_in_the_graph():
    bi = SpyBI(delay=0.02)
    ctx, rec = await run_graph([(LN, ["DAU"]), (BI, ["DAU"])], Deps(bi=bi), edges=[])
    ev = rec.collab.events
    assert [e.seq for e in ev] == sorted(e.seq for e in ev) and len({e.seq for e in ev}) == len(ev)
    agents = {n.id for n in ctx.plan.chosen.graph.nodes}
    assert all(e.a in agents and (e.b is None or e.b in agents) for e in ev)
    kinds = [e.kind for e in ev]
    assert "probe" in kinds and ("reuse" in kinds or "ask" in kinds)


async def test_reuse_event_points_from_the_card_author_to_the_agent_that_reused_it():
    bi = SpyBI(delay=0.02)
    _, rec = await run_graph([(BI, ["DAU"]), (LN, ["DAU"])], Deps(bi=bi), edges=[(f"{BI}@DAU", f"{LN}@DAU")])
    r = next(e for e in rec.collab.events if e.kind == "reuse")
    assert (r.a, r.b, r.need_key) == (f"{BI}@DAU", f"{LN}@DAU", "assets:DAU")


async def test_directed_ask_emits_ask_then_reply_with_outcome():
    ns = nodes((BI, ["GMV"]), (LN, ["CTR"]))
    hub = hub_with(ns)
    await hub.obtain(need("assets", "CTR"), requester="lineage_discovery@CTR", who="lineage_discovery@CTR", produce=None)
    ev = [(e.kind, e.a, e.b, e.sub) for e in hub.trace.events if e.kind != "probe"]
    assert [x[0] for x in ev] == ["ask", "reply"]
    assert ev[0][1:3] == ("lineage_discovery@CTR", "bi_asset_discovery@GMV") and ev[1][1:3] == ("bi_asset_discovery@GMV", "lineage_discovery@CTR") and ev[1][3] == "adopted"


async def test_decline_downgrade_broadcast_and_volunteer_events_in_order():
    ns = nodes((BI, ["DAU"]), (BI, ["GMV"]), (LN, ["CTR"]))
    hub = hub_with(ns, COLLAB_WAIT_SEC=0.05)
    owner = "bi_asset_discovery@DAU"
    hub.agents[owner]._active = hub.agents[owner].spec.max_concurrency
    await hub.board.try_claim(owner, "assets:CTR")
    await hub.obtain(need("assets", "CTR"), requester="lineage_discovery@CTR", who="lineage_discovery@CTR", produce=None)
    ev = [e for e in hub.trace.events if e.kind != "probe"]
    assert [e.kind for e in ev] == ["ask", "decline", "broadcast", "volunteer"]
    assert (ev[1].a, ev[1].b) == (owner, "lineage_discovery@CTR") and ev[1].note.startswith("busy")            # 婉拒带理由
    assert ev[2].a == "lineage_discovery@CTR" and ev[2].b is None                                                # 广播：请求方 → 全体
    assert (ev[3].a, ev[3].b, ev[3].sub) == ("bi_asset_discovery@GMV", "lineage_discovery@CTR", "adopted")


async def test_unanswered_ask_emits_an_unanswered_event():
    hub = hub_with(nodes((CAL, ["DAU"]), ("dim_value_resolver", [])))
    await hub.obtain(need("lineage", "DAU"), requester="caliber_lookup@DAU", who="caliber_lookup@DAU", produce=None)
    ev = [e for e in hub.trace.events if e.kind != "probe"]
    assert [e.kind for e in ev][-1] == "unanswered" and ev[-1].a == "caliber_lookup@DAU"


async def test_an_agent_reusing_its_own_card_is_not_a_communication_event():
    _, rec = await run_graph([("omni_discovery", ["DAU"])], Deps(bi=SpyBI(0.01)))
    assert all(not (e.kind == "reuse" and e.a == e.b) for e in rec.collab.events)
    assert [e.kind for e in rec.collab.events if e.kind != "check"] == ["probe", "probe", "probe"]        # 三个通道各探查一次；没有跨 agent 的交互
    assert sum(e.kind == "check" for e in rec.collab.events) == 1                                         # 探查完做一次内联核对
