"""④ 发现驱动的转交（handoff）：口径查找找到相关口径 → 交给 BI 与血缘；血缘找到的资产 → 再交给 BI / 口径查找。"""
import asyncio

import pytest
from fastapi.testclient import TestClient

from app.deps import Deps
from app.main import create_app
from app.roles import REGISTRY
from app.schemas.knowledge import KnowledgeItem, KnowledgeKind, KnowledgeOrigin
from app.schemas.recall import AssetMeta, CandidateAsset
from app.schemas.workflow import EdgeKind, WorkflowEdge, WorkflowGraph
from app.services.fakes import FakeBIRetrieval, FakeKnowledgeRetrieval, FakeLineageRetrieval
from app.stages import RunContext
from app.stages.s2_coordination import CoordinationStage
from app.stages.s3_planning import PlanningStage
from app.stages.s3_planning.candidates import make_node
from app.stages.s4_recall import RecallStage

BI, LN, CAL, OMNI = "bi_asset_discovery", "lineage_discovery", "caliber_lookup", "omni_discovery"


@pytest.fixture(autouse=True)
def _handoff_on(monkeypatch):
    from app.settings import settings as _s
    monkeypatch.setattr(_s, "COLLAB_HANDOFF_ENABLED", True)


def _asset(i, name, typ="table", score=0.8):
    return CandidateAsset(asset_id=i, name=name, asset_type=typ, retrieval_score=score, meta=AssetMeta(grain="day", partition_field="dt", fields=["dt", "dau"]))


def _metric(i, title, assets=(), terms=()):
    return KnowledgeItem(id=i, kind=KnowledgeKind.METRIC_DEF, title=title, content=f"{title} 的口径", origin=KnowledgeOrigin.RETRIEVAL,
                         asset_ids=list(assets), terms=list(terms), relevance=0.8)


class Spy:
    """记录各通道被调用时的 (query, params/ids)。"""

    def __init__(self):
        self.bi, self.bi_get, self.ln, self.kb = [], [], [], []


def make_deps(spy: Spy, *, caliber_items=None, bi_assets=None, lineage_assets=None, **settings):
    bi_assets = bi_assets if bi_assets is not None else [_asset("dataset:1", "DAU 数据集", "dataset")]
    lineage_assets = lineage_assets if lineage_assets is not None else [_asset("table:dws_dau", "dws_dau"), _asset("table:dws_edau", "dws_edau")]
    caliber_items = caliber_items if caliber_items is not None else [_metric("k_dau", "DAU", ["dataset:1"]), _metric("k_edau", "eDAU", ["table:dws_edau"], terms=["有效日活"])]

    class BI(FakeBIRetrieval):
        async def search(self, query, **kw):
            spy.bi.append(query)
            await asyncio.sleep(0.01)
            return [a.model_copy(deep=True) for a in bi_assets] if query in ("DAU", "eDAU") or "dws" in query else []

        async def get(self, ids):
            spy.bi_get.append(list(ids))
            return [CandidateAsset(asset_id=i, name=f"resolved {i}", asset_type="table" if i.startswith("table") else "dataset", retrieval_score=0.7,
                                   meta=AssetMeta(grain="day", partition_field="dt", fields=["dt", "dau"])) for i in ids]

    class LN(FakeLineageRetrieval):
        async def search(self, query, *, limit=10, cookie=None, params=None):
            spy.ln.append((query, list((params or {}).get("seed_assets", []))))
            if query == "eDAU":                                       # 不同口径沿血缘找到不同的表，转交链路才会真正延伸
                return [_asset("table:edau_detail", "edau_detail", score=0.9)]
            return [a.model_copy(deep=True) for a in lineage_assets]

    class KB(FakeKnowledgeRetrieval):
        async def search(self, query, **kw):
            spy.kb.append(query)
            return list(caliber_items) if query == "DAU" else []

    deps = Deps(bi=BI(), lineage=LN(), caliber_kb=KB(), kh=KB())
    if settings:
        deps.settings = deps.settings.model_copy(update=settings)
    return deps


async def run(specs, deps, query="最近7天DAU", edges=(), a_max=8):
    ctx = RunContext(query=query, deps=deps)
    ctx.contract = await CoordinationStage().run(ctx)
    ctx.plan = await PlanningStage().run(ctx)
    g = WorkflowGraph(nodes=[make_node(REGISTRY[r].spec, t) for r, t in specs], edges=[WorkflowEdge(src=a, dst=b, kind=EdgeKind.DEP) for a, b in edges])
    ctx.plan = ctx.plan.model_copy(update={"chosen": ctx.plan.chosen.model_copy(update={"graph": g}), "max_parallel": a_max, "known_assets": []})
    return await RecallStage().run(ctx)


def handoffs(rec):
    return [(e.a, e.b, e.need_key) for e in rec.collab.events if e.kind == "ask" and e.via]


# ── 口径查找 → BI + 血缘 ────────────────────────────────────────────────────────
async def test_caliber_lookup_hands_a_related_caliber_to_bi_and_lineage():
    spy = Spy()
    rec = await run([(CAL, ["DAU"]), (BI, ["DAU"]), (LN, ["DAU"])], make_deps(spy))
    hs = handoffs(rec)
    assert (f"{CAL}@DAU", f"{BI}@DAU", "assets:eDAU") in hs and (f"{CAL}@DAU", f"{LN}@DAU", "lineage:eDAU") in hs
    keys = {c.key for c in rec.cards}
    assert {"caliber:DAU", "assets:eDAU", "lineage:eDAU"} <= keys
    assert "eDAU" in spy.bi and any(q == "eDAU" for q, _ in spy.ln)                          # BI 与血缘真的去查了这个新口径
    assert next(seeds for q, seeds in spy.ln if q == "eDAU")[0] == "table:dws_edau"          # 血缘以口径文档关联的资产为种子（转交带来的排第一）
    assert {"table:dws_edau"} <= set(spy.bi_get[0]) or any("table:dws_edau" in g for g in spy.bi_get)   # BI 解析了口径关联的资产


async def test_handoff_results_become_candidates_with_the_producing_role_in_found_by():
    spy = Spy()
    rec = await run([(CAL, ["DAU"]), (BI, ["DAU"]), (LN, ["DAU"])], make_deps(spy, bi_assets=[_asset("dataset:1", "DAU 数据集", "dataset")]))
    by = {c.asset_id: c for c in rec.candidates}
    assert "table:dws_edau" in by and by["table:dws_edau"].channels                          # 只经转交链路找到的资产也进了候选池
    assert BI in by["table:dws_edau"].found_by or LN in by["table:dws_edau"].found_by or CAL in by["table:dws_edau"].found_by


# ── 血缘 → 其他角色 ───────────────────────────────────────────────────────────────
async def test_lineage_hands_what_it_found_to_bi_and_caliber_lookup():
    spy = Spy()
    rec = await run([(BI, ["DAU"]), (LN, ["DAU"]), (CAL, ["DAU"])], make_deps(spy, caliber_items=[_metric("k_dau", "DAU", ["dataset:1"])]), edges=[(f"{BI}@DAU", f"{LN}@DAU")])
    hs = handoffs(rec)
    froms = {(a, b.split("@")[0]) for a, b, k in hs if a == f"{LN}@DAU"}
    assert (f"{LN}@DAU", BI) in froms and (f"{LN}@DAU", CAL) in froms                           # 血缘找到的资产：交给 BI 和口径查找
    assert any(k.startswith("assets:dws") for _, _, k in hs) and any(k.startswith("caliber:dws") for _, _, k in hs)
    assert any(set(ids) & {"table:dws_dau", "table:dws_edau"} for ids in spy.bi_get)         # BI 按血缘资产的 id 解析元信息


async def test_chains_continue_but_stop_at_the_depth_limit():
    deep = await run([(CAL, ["DAU"]), (BI, ["DAU"]), (LN, ["DAU"])], make_deps(Spy(), COLLAB_HANDOFF_MAX_DEPTH=3))
    shallow = await run([(CAL, ["DAU"]), (BI, ["DAU"]), (LN, ["DAU"])], make_deps(Spy(), COLLAB_HANDOFF_MAX_DEPTH=1))
    hop = lambda r: max((e.hop for e in r.collab.events if e.via), default=0)
    assert hop(shallow) == 1 and 2 <= hop(deep) <= 3                                            # 转交的结果会再触发下一跳，但不超过深度上限
    assert len(handoffs(deep)) >= len(handoffs(shallow)) >= 2


async def test_total_budget_caps_the_number_of_handoffs():
    spy = Spy()
    rec = await run([(CAL, ["DAU"]), (BI, ["DAU"]), (LN, ["DAU"])], make_deps(spy, COLLAB_HANDOFF_MAX_TOTAL=2))
    assert len(handoffs(rec)) <= 2 and rec.collab.stats["handoffs"] <= 2


async def test_handoff_can_be_switched_off():
    spy = Spy()
    rec = await run([(CAL, ["DAU"]), (BI, ["DAU"]), (LN, ["DAU"])], make_deps(spy, COLLAB_HANDOFF_ENABLED=False))
    assert handoffs(rec) == [] and rec.collab.stats["handoffs"] == 0


async def test_every_follow_up_need_is_handled_once():
    spy = Spy()
    rec = await run([(CAL, ["DAU"]), (BI, ["DAU"]), (LN, ["DAU"])], make_deps(spy))
    hs = handoffs(rec)
    assert len({k for _, _, k in hs}) == len(hs) or len(set(hs)) == len(hs)
    assert len(rec.collab.retrievals) == len(set(rec.collab.retrievals))                       # 同一需求不会被重复探查（含转交链路）


# ── 谁来接 ────────────────────────────────────────────────────────────────────────
async def test_the_finder_hands_off_to_other_agents_not_to_itself():
    spy = Spy()
    rec = await run([(CAL, ["DAU"]), (BI, ["DAU"]), (LN, ["DAU"])], make_deps(spy))
    assert all(a != b for a, b, _ in handoffs(rec))


async def test_when_no_other_agent_serves_the_kind_the_finder_does_it_itself_if_it_can():
    spy = Spy()
    rec = await run([(OMNI, ["DAU"])], make_deps(spy))                                         # 只有一个全通道 agent：转交给自己
    assert handoffs(rec) == []                                                                  # 没有「别人」可转交
    assert {"assets:eDAU", "lineage:eDAU"} <= {c.key for c in rec.cards}                       # 但它自己顺着相关口径继续查了
    assert all(e.a == f"{OMNI}@DAU" for e in rec.collab.events if e.kind == "probe")


async def test_no_pointless_handoff_when_nobody_in_the_workflow_serves_the_kind():
    spy = Spy()
    rec = await run([(CAL, ["DAU"])], make_deps(spy))                                          # 工作流里只有口径查找：BI / 血缘无人可交
    assert handoffs(rec) == [] and not any(e.kind == "unanswered" for e in rec.collab.events)


async def test_replica_holding_the_subject_gets_the_handoff():
    spy = Spy()
    items = [_metric("k_dau", "DAU", ["dataset:1"]), _metric("k_edau", "eDAU", ["table:dws_edau"])]
    rec = await run([(CAL, ["DAU"]), (BI, ["DAU"]), (BI, ["eDAU"])], make_deps(spy, caliber_items=items))
    h = [(a, b) for a, b, k in handoffs(rec) if k == "assets:eDAU"]
    assert h and h[0][1] == f"{BI}@eDAU"                                                        # 目标口径就是 eDAU 的 BI 副本接手


# ── 不惩罚转交里的空结果 ───────────────────────────────────────────────────────────
async def test_an_empty_handoff_result_does_not_hurt_the_responders_profile_but_an_empty_normal_answer_does():
    from app.collab.hub import build_hub
    from app.roles import build_agents
    from app.schemas.collab import Need
    from app.schemas.intent import Intent
    from app.schemas.task import Slot, TaskContract, ValueSource
    from app.stages.s3_planning.candidates import make_node

    class NoBI(FakeBIRetrieval):
        async def search(self, query, **kw):
            return []

    def hub():
        ns = [make_node(REGISTRY[BI].spec, ["X"]), make_node(REGISTRY[LN].spec, ["X"])]
        c = TaskContract(query="q", metrics=Slot(value=["X"], source=ValueSource.USER), intent=Intent(query="q", domain="transaction"))
        return build_hub(build_agents(ns), Deps(bi=NoBI()), contract=c, knowledge=[], cookie=None, recall_params={})

    mk = lambda via: Need(kind="assets", subject="Y", domains=["transaction"], asset_type="dataset", via=via, hop=1 if via else 0)
    h = hub()
    await h._ask(mk("caliber:X"), f"{LN}@X", "t")
    assert h.trace.asks[0].outcome == "ignored" and not h.trace.profile_updates                  # 转交里的空结果：不惩罚
    h = hub()
    await h._ask(mk(""), f"{LN}@X", "t")
    assert h.trace.asks[0].outcome == "ignored" and [u.outcome for u in h.trace.profile_updates] == ["ignored"]     # 普通求助的空回应：照旧计入响应质量


# ── 并发 / 死锁 ───────────────────────────────────────────────────────────────────
async def test_handoff_chains_do_not_deadlock_under_serial_a_max_and_cross_asks():
    spy = Spy()
    rec = await asyncio.wait_for(run([(CAL, ["DAU"]), (BI, ["DAU"]), (LN, ["DAU"])], make_deps(spy, COLLAB_WAIT_SEC=2), a_max=1,
                                     edges=[(f"{CAL}@DAU", f"{BI}@DAU"), (f"{BI}@DAU", f"{LN}@DAU")]), timeout=15)
    assert [r.status for r in rec.role_reports] == ["ok"] * 3


# ── 轨迹 / HTTP ───────────────────────────────────────────────────────────────────
async def test_trace_counts_handoffs_per_agent_and_overall():
    spy = Spy()
    rec = await run([(CAL, ["DAU"]), (BI, ["DAU"]), (LN, ["DAU"])], make_deps(spy))
    info = {a.agent_id: a for a in rec.collab.agents}
    assert rec.collab.stats["handoffs"] == len(handoffs(rec)) >= 2
    assert info[f"{CAL}@DAU"].handoffs >= 2 and sum(a.handoffs for a in info.values()) == rec.collab.stats["handoffs"]
    ev = next(e for e in rec.collab.events if e.via)
    assert ev.via.startswith(("caliber:", "lineage:", "assets:")) and ev.hop >= 1


def test_http_stage4_exposes_handoff_events_with_the_demo_fakes():
    b = TestClient(create_app(Deps())).post("/api/stage4", json={"query": "最近7天DAU", "max_parallel": 8, "graph": {
        "nodes": [{"id": "caliber_lookup@DAU", "role": "caliber_lookup", "targets": ["DAU"]}, {"id": "bi_asset_discovery@DAU", "role": "bi_asset_discovery", "targets": ["DAU"]},
                  {"id": "lineage_discovery@DAU", "role": "lineage_discovery", "targets": ["DAU"]}], "edges": []}}).json()
    t = b["recall"]["collab"]
    assert t["stats"]["handoffs"] >= 1 and any(e["via"] for e in t["events"])


# ── 已经解析过元信息的资产，不再转交给 BI 重新解析 ──────────────────────────────────────────────
async def test_assets_whose_metadata_is_already_complete_are_not_handed_to_bi_again():
    spy = Spy()
    rec = await run([(BI, ["DAU"]), (LN, ["DAU"])], make_deps(spy), edges=[(f"{BI}@DAU", f"{LN}@DAU")])
    in_pool = {c.asset_id: c for c in rec.candidates}
    redundant = [k for a, b, k in handoffs(rec) if k.startswith("assets:") and any(
        (c := in_pool.get(x)) is not None and c.meta.grain and c.meta.partition_latest for x in [i for i in in_pool if in_pool[i].name == k.split(":", 1)[1]])]
    assert redundant == []


async def test_a_found_asset_with_thin_metadata_is_still_handed_to_bi_to_resolve():
    from app.collab.handoff import follow_ups
    from app.collab.hub import build_hub
    from app.roles import build_agents
    from app.schemas.collab import EvidenceCard
    ns = [make_node(REGISTRY[r].spec, ["DAU"]) for r in (LN, BI)]
    hub = build_hub(build_agents(ns), Deps(), contract=(await _contract("DAU")), knowledge=[], cookie=None, recall_params={})
    full = CandidateAsset(asset_id="table:full", name="full", asset_type="table", retrieval_score=0.9, meta=AssetMeta(grain="day", partition_latest="2026-11-12", fields=["dt"]))
    thin = CandidateAsset(asset_id="table:thin", name="thin", asset_type="table", retrieval_score=0.8, meta=AssetMeta(fields=["dt"]))
    await hub.board.publish(EvidenceCard(id="c1", key="lineage:DAU", kind="lineage", subject="DAU", author="x", asset_ids=["table:full", "table:thin"], confidence=0.9), assets=[full, thin])
    card = hub.board.cards["lineage:DAU"]
    from app.schemas.collab import Need
    needs = follow_ups(hub, card, Need(kind="lineage", subject="DAU", domains=["transaction"], asset_type="table"))
    keys = {n.key for n in needs}
    assert "assets:thin" in keys and "assets:full" not in keys and {"caliber:thin", "caliber:full"} <= keys     # 口径查找不受影响


async def _contract(m):
    ctx = RunContext(query=f"最近7天{m}", deps=Deps())
    return await CoordinationStage().run(ctx)
