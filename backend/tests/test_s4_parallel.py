"""④ 多 agent 并行检索：按 ③ 的图并行执行，重复探查被协调；BI + 知识双路归一化进同一候选池；画像在线更新。"""
import asyncio

import pytest
from fastapi.testclient import TestClient

from app.collab.registry import JsonProfileStore, ProfileRegistry
from app.deps import Deps
from app.main import create_app
from app.roles import REGISTRY
from app.schemas.knowledge import KnowledgeItem, KnowledgeKind, KnowledgeOrigin, PreInjectionResult
from app.schemas.recall import AssetMeta, CandidateAsset
from app.schemas.workflow import EdgeKind, WorkflowEdge, WorkflowGraph
from app.services.fakes import FakeBIRetrieval, FakeKnowledgeRetrieval, FakeLineageRetrieval
from app.stages import RunContext
from app.stages.s2_coordination import CoordinationStage
from app.stages.s3_planning import PlanningStage
from app.stages.s3_planning.candidates import make_node
from app.stages.s4_recall import RecallStage


async def run_graph(specs, deps, query="最近7天DAU", edges=()):
    """specs: [(role, [targets])]；跳过 ③ 的择优，直接按给定图跑 ④。"""
    ctx = RunContext(query=query, deps=deps, cookie="common-internal-access-token-prod=t")
    ctx.contract = await CoordinationStage().run(ctx)
    ctx.plan = await PlanningStage().run(ctx)
    nodes = [make_node(REGISTRY[r].spec, t) for r, t in specs]
    g = WorkflowGraph(nodes=nodes, edges=[WorkflowEdge(src=a, dst=b, kind=EdgeKind.DEP) for a, b in edges])
    ctx.plan = ctx.plan.model_copy(update={"chosen": ctx.plan.chosen.model_copy(update={"graph": g}), "max_parallel": 8, "known_assets": []})
    return ctx, await RecallStage().run(ctx)


class SpyBI(FakeBIRetrieval):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls: list[str] = []

    async def search(self, query, **kw):
        self.calls.append(query)
        await asyncio.sleep(0.02)                         # 让并发节点真的交错
        return await super().search(query, limit=5)


class SpyLineage(FakeLineageRetrieval):
    def __init__(self):
        super().__init__()
        self.calls: list[tuple[str, list[str]]] = []

    async def search(self, query, *, limit=10, cookie=None, params=None):
        self.calls.append((query, list((params or {}).get("seed_assets", []))))
        return await super().search(query, limit=limit)


async def test_parallel_bi_and_lineage_nodes_probe_the_same_assets_only_once():
    bi, ln = SpyBI(), SpyLineage()
    _, rec = await run_graph([("bi_asset_discovery", ["DAU"]), ("lineage_discovery", ["DAU"]), ("caliber_lookup", ["DAU"])], Deps(bi=bi, lineage=ln))
    assert bi.calls == ["DAU"]                                                   # 血缘节点没有自己再查一遍 BI
    assert ln.calls and ln.calls[0][1], "血缘节点以 BI 资产为种子沿血缘扩展"
    assert all(r.status == "ok" for r in rec.role_reports) and rec.candidates
    st = rec.collab.summarize()
    assert st["reuse_hits"] + st["asks"] >= 1 and rec.collab.retrievals.count("assets:DAU") == 1


async def test_omni_agent_and_bi_agent_do_not_duplicate_the_bi_probe():
    bi = SpyBI()
    _, rec = await run_graph([("bi_asset_discovery", ["DAU"]), ("omni_discovery", ["DAU"])], Deps(bi=bi))
    assert bi.calls == ["DAU"] and rec.collab.retrievals.count("assets:DAU") == 1
    assert all(r.status == "ok" for r in rec.role_reports)
    assert {c.key for c in rec.cards} == {"assets:DAU", "lineage:DAU", "caliber:DAU"}                  # omni 补齐另两个通道，BI 的卡被复用


async def test_two_metric_batched_nodes_query_each_caliber_once():
    bi = SpyBI()
    _, rec = await run_graph([("bi_asset_discovery", ["DAU", "GMV"]), ("omni_discovery", ["GMV"])], Deps(bi=bi), query="DAU和GMV")
    assert sorted(bi.calls) == ["DAU", "GMV"] and rec.collab.retrievals.count("assets:GMV") == 1


async def test_dependent_nodes_check_the_candidates_their_upstream_found():
    _, rec = await run_graph([("bi_asset_discovery", ["DAU"]), ("lineage_discovery", ["DAU"])], Deps(), query="最近7天各渠道的DAU", edges=[("bi_asset_discovery@DAU", "lineage_discovery@DAU")])
    ln = next(r for r in rec.role_reports if r.role == "lineage_discovery")
    assert ln.candidates and any(e.role == "lineage_discovery" and e.source == "bi_asset" for e in ln.evidences)          # 血缘节点核对了上游 BI 找到的候选的维度覆盖


async def test_caliber_without_evidence_asks_for_carrier_assets_instead_of_searching_bi_itself():
    class EmptyKB(FakeKnowledgeRetrieval):
        async def search(self, query, **kw):
            return []

    bi = SpyBI()
    _, rec = await run_graph([("caliber_lookup", ["DAU"]), ("bi_asset_discovery", ["DAU"])], Deps(bi=bi, caliber_kb=EmptyKB()))
    assert bi.calls == ["DAU"]
    assert rec.collab.retrievals.count("assets:DAU") == 1
    card = next(c for c in rec.cards if c.key == "caliber:DAU")
    assert card.asset_ids, "口径卡通过复用 / 询问关联到了承载资产"


# ── 混合检索：BI + 知识（向量 + 图谱）归一化进同一候选池 ───────────────────────────
async def test_hybrid_pool_merges_bi_and_knowledge_channels():
    base = CandidateAsset(asset_id="dataset:1", name="DAU 数据集", asset_type="dataset", retrieval_score=0.6,
                          meta=AssetMeta(grain="day", fields=["dau"], partition_field="dt"))

    class BI(FakeBIRetrieval):
        async def search(self, query, **kw):
            return [base.model_copy(deep=True)]

        async def get(self, ids):
            return [base.model_copy(deep=True) if i == "dataset:1" else CandidateAsset(asset_id=i, name=i, retrieval_score=0.5) for i in ids]

    class LN(FakeLineageRetrieval):
        async def search(self, query, **kw):
            return [CandidateAsset(asset_id="dataset:1", name="DAU 数据集", asset_type="dataset", retrieval_score=0.9, meta=AssetMeta(grain="month", fields=["dau", "channel"])),
                    CandidateAsset(asset_id="table:dws_dau", name="dws_dau", asset_type="table", retrieval_score=0.8)]

    class KB(FakeKnowledgeRetrieval):
        async def search(self, query, **kw):
            return [KnowledgeItem(id="k1", kind=KnowledgeKind.METRIC_DEF, title="DAU", content="日活", origin=KnowledgeOrigin.RETRIEVAL,
                                  asset_ids=["dataset:1", "dws.legacy"], relevance=0.8),
                    KnowledgeItem(id="k2", kind=KnowledgeKind.DIMENSION, title="渠道", content="渠道", origin=KnowledgeOrigin.RETRIEVAL, asset_ids=["dws.legacy"], relevance=0.7)]

    _, rec = await run_graph([("bi_asset_discovery", ["DAU"]), ("lineage_discovery", ["DAU"]), ("caliber_lookup", ["DAU"])], Deps(bi=BI(), lineage=LN(), caliber_kb=KB()))
    by = {c.asset_id: c for c in rec.candidates}
    both = by["dataset:1"]
    assert both.channels == ["bi", "knowledge:graph", "knowledge:vector"]                  # 三路都命中同一个资产，合并为一条
    assert both.meta.grain == "day" and both.meta.partition_field == "dt" and both.meta.fields == ["dau", "channel"]   # BI 元信息为准，知识通道补缺口
    assert both.retrieval_score == pytest.approx(0.91)                                      # max(0.9×0.9, 0.6, 0.8×0.9) + 0.1 多通道加成
    assert by["table:dws_dau"].channels == ["knowledge:graph"] and by["dws.legacy"].channels == ["knowledge:vector"]
    assert set(both.found_by) == {"bi_asset_discovery", "lineage_discovery", "caliber_lookup"}
    assert rec.by_channel == {"bi+knowledge:graph+knowledge:vector": 1, "knowledge:graph": 1, "knowledge:vector": 1}
    assert {(e.kind, e.name) for e in rec.related_entities} == {("metric", "DAU"), ("dimension", "渠道")}


# ── 画像在线更新与持久化 ─────────────────────────────────────────────────────────
async def test_profiles_are_updated_by_the_run_and_persisted(tmp_path):
    path = tmp_path / "profiles.json"
    deps = Deps(profiles=ProfileRegistry(store=JsonProfileStore(path)), bi=SpyBI())
    _, rec = await run_graph([("lineage_discovery", ["DAU"])], deps)                       # 没有 BI 节点：血缘节点必须求助
    ask = rec.collab.asks[0]
    assert ask.route.mode.value == "directed" and ask.route.target == "bi_asset_discovery" or ask.outcome == "unanswered"
    assert path.exists()
    again = ProfileRegistry(store=JsonProfileStore(path))
    assert again.get("lineage_discovery") is not None
    assert {u.role for u in rec.collab.profile_updates} <= {"bi_asset_discovery", "omni_discovery", "lineage_discovery"}


async def test_ask_to_a_role_that_is_not_in_the_workflow_is_not_possible_so_it_probes_itself():
    bi = SpyBI()
    _, rec = await run_graph([("lineage_discovery", ["DAU"])], Deps(bi=bi))               # 工作流里只有血缘角色
    card = next(c for c in rec.cards if c.key == "lineage:DAU")
    assert card.author == "lineage_discovery@DAU" and bi.calls == []                          # 没人可问 → 无种子，直接按口径名沿图谱检索
    assert rec.collab.asks and rec.collab.asks[0].outcome == "unanswered"
    assert rec.candidates


# ── 契约 / HTTP ─────────────────────────────────────────────────────────────────
def test_http_stage4_returns_candidates_cards_and_collaboration_trace():
    c = TestClient(create_app(Deps()))
    b = c.post("/api/stage4", json={"query": "最近7天各渠道的DAU", "session_id": "ui", "max_parallel": 4}).json()
    assert b["status"] == "ready" and b["plan"]["chosen"]["graph"]["nodes"]
    rec = b["recall"]
    assert rec["candidates"] and rec["cards"] and set(rec["collab"]["stats"]) >= {"retrievals", "reuse_hits", "asks", "directed", "broadcast", "declined", "downgraded", "volunteered", "adopted"}
    assert all(ch in ("bi", "knowledge:vector", "knowledge:graph", "known") for cand in rec["candidates"] for ch in cand["channels"])
    assert rec["by_channel"]


def test_http_stage4_runs_even_when_metric_missing():
    b = TestClient(create_app(Deps())).post("/api/stage4", json={"query": "帮我看看数据"}).json()
    assert b["status"] == "ready" and b["recall"] is not None


def test_http_profiles_endpoint_lists_role_profiles_with_quality():
    d = Deps()
    c = TestClient(create_app(d))
    c.post("/api/stage4", json={"query": "最近7天DAU"})
    ps = {p["role"]: p for p in c.get("/api/profiles").json()}
    assert {"bi_asset_discovery", "lineage_discovery", "caliber_lookup"} <= set(ps)
    assert {"domains", "serves", "asset_types", "keywords", "quality", "stats"} <= set(ps["bi_asset_discovery"])


def test_sse_stage4_event_carries_collaboration_trace():
    ev = TestClient(create_app(Deps())).post("/api/discover/sync", json={"query": "最近7天DAU"}).json()["events"]
    s4 = next(e for e in ev if e["event"] == "stage4")
    assert "collab" in s4["result"] and "by_channel" in s4["result"]


async def test_serial_a_max_does_not_deadlock_when_the_dependent_node_runs_first():
    """A_max=1：血缘节点先拿到唯一并发名额并向 BI 角色求助——被求助的角色在询问内联完成，不需要名额，所以不会死锁。"""
    bi = SpyBI()
    ctx = RunContext(query="最近7天DAU", deps=Deps(bi=bi))
    ctx.contract = await CoordinationStage().run(ctx)
    ctx.plan = await PlanningStage().run(ctx)
    g = WorkflowGraph(nodes=[make_node(REGISTRY[r].spec, ["DAU"]) for r in ("lineage_discovery", "bi_asset_discovery", "caliber_lookup")],
                      edges=[WorkflowEdge(src="lineage_discovery@DAU", dst="bi_asset_discovery@DAU", kind=EdgeKind.THEN)])      # 让血缘节点先于 BI 节点运行
    ctx.plan = ctx.plan.model_copy(update={"chosen": ctx.plan.chosen.model_copy(update={"graph": g}), "max_parallel": 1, "known_assets": []})
    rec = await asyncio.wait_for(RecallStage().run(ctx), timeout=10)
    assert [r.status for r in rec.role_reports] == ["ok", "ok", "ok"] and bi.calls == ["DAU"]
    assert rec.collab.asks[0].requester == "lineage_discovery@DAU" and rec.collab.asks[0].requester_role == "lineage_discovery" and rec.collab.stats["adopted"] == 1
    assert any(e.by == "bi_asset_discovery@DAU" for e in rec.collab.reuse)                       # BI 节点随后复用了询问产出的证据卡


# ── 血缘以现有召回内容为种子，沿图谱通道找 lineage ────────────────────────────────
class SeedLineage(FakeLineageRetrieval):
    def __init__(self):
        super().__init__()
        self.params: list[dict] = []

    async def search(self, query, *, limit=10, cookie=None, params=None):
        self.params.append(dict(params or {}))
        return await super().search(query, limit=limit)


def _known(asset_ids):
    return [KnowledgeItem(id="kd", kind=KnowledgeKind.METRIC_DEF, title="DAU", content="日活", origin=KnowledgeOrigin.RETRIEVAL, asset_ids=list(asset_ids))]


async def test_lineage_seeds_come_from_preinjected_assets_and_the_bi_recall_and_are_passed_to_the_graph_channel():
    ln = SeedLineage()
    ctx = RunContext(query="最近7天DAU", deps=Deps(lineage=ln, bi=SpyBI()))
    ctx.contract = await CoordinationStage().run(ctx)
    ctx.plan = await PlanningStage().run(ctx)
    g = WorkflowGraph(nodes=nodes_of([("bi_asset_discovery", ["DAU"]), ("lineage_discovery", ["DAU"])]), edges=[])
    ctx.plan = ctx.plan.model_copy(update={"chosen": ctx.plan.chosen.model_copy(update={"graph": g}), "known_assets": []})
    ctx.preinjection = PreInjectionResult(injected=_known(["dws.legacy_dau"]))
    rec = await RecallStage().run(ctx)
    seeds = ln.params[0]["seed_assets"]
    assert seeds[0] == "dws.legacy_dau" and "dws.user_active_d" in seeds              # 预注入知识指明的资产在前，BI 召回的资产（经黑板复用）在后
    assert ln.params[0]["seed_names"]                                                  # 种子名也给图谱通道做语义匹配
    card = next(c for c in rec.cards if c.key == "lineage:DAU")
    assert card.seeds == seeds and "种子" in card.summary
    assert all(c.meta.extra.get("lineage_seeds") == seeds for c in rec.candidates if "knowledge:graph" in c.channels)


async def test_lineage_still_seeds_from_preinjected_knowledge_when_there_is_no_bi_node():
    ln = SeedLineage()
    ctx = RunContext(query="最近7天DAU", deps=Deps(lineage=ln))
    ctx.contract = await CoordinationStage().run(ctx)
    ctx.plan = await PlanningStage().run(ctx)
    g = WorkflowGraph(nodes=nodes_of([("lineage_discovery", ["DAU"])]), edges=[])
    ctx.plan = ctx.plan.model_copy(update={"chosen": ctx.plan.chosen.model_copy(update={"graph": g}), "known_assets": []})
    ctx.preinjection = PreInjectionResult(injected=_known(["dws.legacy_dau"]))
    await RecallStage().run(ctx)
    assert ln.params[0]["seed_assets"] == ["dws.legacy_dau"]


async def test_session_confirmed_assets_are_seeds_too():
    ln = SeedLineage()
    ctx = RunContext(query="最近7天DAU", deps=Deps(lineage=ln), confirmed_assets={"DAU": ["dataset:77"]})
    ctx.contract = await CoordinationStage().run(ctx)
    ctx.plan = await PlanningStage().run(ctx)
    g = WorkflowGraph(nodes=nodes_of([("lineage_discovery", ["DAU"])]), edges=[])
    ctx.plan = ctx.plan.model_copy(update={"chosen": ctx.plan.chosen.model_copy(update={"graph": g}), "known_assets": ["dataset:77"]})
    await RecallStage().run(ctx)
    assert "dataset:77" in ln.params[0]["seed_assets"]


async def test_no_seed_at_all_is_recorded_not_hidden():
    ln = SeedLineage()
    _, rec = await run_graph([("lineage_discovery", ["DAU"])], Deps(lineage=ln))
    assert ln.params[0]["seed_assets"] == []
    card = next(c for c in rec.cards if c.key == "lineage:DAU")
    assert card.seeds == [] and "无种子" in card.summary


def nodes_of(specs):
    return [make_node(REGISTRY[r].spec, t) for r, t in specs]
