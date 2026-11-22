import json
from unittest import mock

import pytest

from app.deps import Deps
from app.errors import UpstreamError
from app.schemas.knowledge import KnowledgeItem, KnowledgeKind, KnowledgeOrigin, KnowledgeSnapshot, RetrievalPack, VerifiedPair
from app.services.context_query import ContextQueryClient, chunk_to_item, infer_kind, parse_pack
from app.services.fakes import FakeKnowledgeRetrieval
from app.stages.s1_preinjection import PreInjectionStage
from app.stages.s1_preinjection.merge import merge_routes
from app.stages.s1_preinjection.pair_source import pair_to_items, reliability
from app.stages.s1_preinjection.selector import SelectorParams, greedy_select, interaction
from app.stores import JsonlPairStore

P = SelectorParams()


def _k(i, title="t", content="c", rel=0.9, cost=10, kind=KnowledgeKind.METRIC_DEF, assets=(), routes=("retrieval",)):
    return KnowledgeItem(id=i, kind=kind, title=title, content=content, origin=KnowledgeOrigin.RETRIEVAL, routes=list(routes),
                         relevance=rel, token_cost=cost, asset_ids=list(assets))


# ── 选择器：u_k / c_k / w_ij / Δ/c / τ / L ───────────────────────────────
def test_greedy_orders_by_gain_per_cost():
    # 单位成本收益：a=0.5/10=0.05, b=0.9/90=0.01, c=0.6/20=0.03
    items = [_k("a", "A", "aaaa", 0.5, 10), _k("b", "B", "bbbb", 0.9, 90), _k("c", "C", "cccc", 0.6, 20)]
    out = greedy_select(items, capacity=1000, p=P)
    assert [i.id for i in out.selected] == ["a", "c", "b"]
    assert [round(s.ratio, 3) for s in out.trace] == [0.05, 0.03, 0.01]


def test_stops_when_best_ratio_below_tau_not_when_budget_left():
    items = [_k("a", "A", "aaaa", 0.9, 10), _k("b", "B", "bbbb", 0.05, 100)]      # b: 0.05/100 = 0.0005 < τ
    out = greedy_select(items, capacity=10_000, p=P)
    assert [i.id for i in out.selected] == ["a"]
    assert out.dropped[0].item_id == "b" and out.dropped[0].reason == "low_gain"


def test_capacity_is_hard_cap_only():
    items = [_k("a", "A", "aaaa", 0.9, 60), _k("b", "B", "bbbb", 0.8, 60)]
    out = greedy_select(items, capacity=80, p=P)
    assert out.used_tokens <= 80 and [i.id for i in out.selected] == ["a"]
    assert [(d.item_id, d.reason) for d in out.dropped] == [("b", "over_budget")]


def test_oversized_block_does_not_stop_the_loop():
    items = [_k("big", "BIG", "xxxx", 0.99, 5000), _k("a", "A", "aaaa", 0.5, 10)]
    out = greedy_select(items, capacity=100, p=P)
    assert [i.id for i in out.selected] == ["a"]


def test_marginal_gain_includes_negative_interaction():
    a = _k("a", "DAU", "日活跃用户数按user_id去重", 0.9, 10)
    b = _k("b", "DAU说明", "日活跃用户数按user_id去重", 0.8, 10, kind=KnowledgeKind.BUSINESS_TERM)
    b.title = "DAU"    # 同标题同内容 → 完全重复
    w, label = interaction(a, b, P)
    assert label == "redundant" and w == pytest.approx(-0.8)           # -ρ·sim·min(u) = -1·1·0.8
    out = greedy_select([a, b], capacity=100, p=P)
    assert [i.id for i in out.selected] == ["a"]                       # Δ(b|a) = 0.8 - 0.8 = 0 < τ
    assert out.dropped[0].reason == "redundant"


def test_partial_overlap_only_discounts():
    a = _k("a", "GMV 口径", "成交总额含已支付订单不剔除退款", 0.9, 10)
    b = _k("b", "GMV 说明", "成交总额含已支付订单不剔除退款，按天汇总", 0.8, 10, kind=KnowledgeKind.BUSINESS_TERM)
    w, label = interaction(a, b, P)
    assert label == "redundant" and -0.8 < w < 0
    out = greedy_select([a, b], capacity=100, p=P)
    assert [i.id for i in out.selected] == ["a", "b"] and out.trace[1].delta < 0.8


def test_conflict_same_topic_different_content():
    a = _k("a", "DAU", "按 user_id 去重，当日有任意启动行为", 0.9, 10)
    b = _k("b", "DAU", "仅统计登录用户，按设备数计算", 0.7, 10)
    assert interaction(a, b, P)[1] == "conflict"
    out = greedy_select([a, b], capacity=100, p=P)
    assert [i.id for i in out.selected] == ["a"] and out.dropped[0].reason == "conflict"


def test_synergy_raises_delta_of_second_block():
    a = _k("a", "GMV", "成交总额含已支付订单", 0.6, 10, assets=["t1"])
    b = _k("b", "T1 元数据", "按天分区粒度为日", 0.5, 10, kind=KnowledgeKind.ASSET_META, assets=["t1"])
    out = greedy_select([a, b], capacity=100, p=P)
    assert out.trace[1].delta == pytest.approx(0.5 + P.synergy * 0.5, abs=1e-3)
    assert out.trace[1].interactions == {"a": pytest.approx(0.1, abs=1e-3)}


def test_value_matches_formula():
    """V(S) = Σu + Σw：逐步 Δ 之和 == V(S)。"""
    a = _k("a", "GMV", "成交总额含已支付订单", 0.6, 10, assets=["t1"])
    b = _k("b", "T1 元数据", "按天分区粒度为日", 0.5, 10, kind=KnowledgeKind.ASSET_META, assets=["t1"])
    c = _k("c", "CTR", "点击率等于点击除以曝光", 0.4, 10, assets=["t2"])
    out = greedy_select([a, b, c], capacity=100, p=P)
    sel = out.selected
    v = sum(i.relevance for i in sel) + sum(interaction(sel[i], sel[j], P)[0] for i in range(len(sel)) for j in range(i + 1, len(sel)))
    assert sum(s.delta for s in out.trace) == pytest.approx(v, abs=1e-3)


# ── 双路合并 ────────────────────────────────────────────────────────────
def test_merge_dual_route_bonus_and_dedup():
    r = _k("k1", rel=0.6, routes=("retrieval",))
    p = _k("k1", rel=0.5, routes=("pair",), assets=["t1"])
    merged = merge_routes([p], [r])
    assert len(merged) == 1 and merged[0].routes == ["pair", "retrieval"]
    assert merged[0].relevance == pytest.approx(0.7) and merged[0].asset_ids == ["t1"]


# ── 路 1：context-query ─────────────────────────────────────────────────
_VO = {"sourceType": "knowledge", "chunkId": "c#0", "fileName": "dau", "title": "DAU 口径定义", "chunkSummary": "s",
       "content": "按 user_id 去重", "sourceUri": "kb://dau", "score": 0.87, "consistencyStatus": "possible_inconsistent"}


def test_chunk_mapping_and_kind_inference():
    it = chunk_to_item(_VO)
    assert it.id == "c#0" and it.kind == KnowledgeKind.METRIC_DEF and it.relevance == 0.87
    assert it.consistency == "possible_inconsistent" and it.token_cost > 0 and it.routes == ["retrieval"]
    assert infer_kind("订单表", "字段说明") == KnowledgeKind.ASSET_META
    assert infer_kind("闲聊", "x") == KnowledgeKind.BUSINESS_TERM


def test_parse_pack_bare_wrapped_and_error():
    bare = {"contextPackId": "p", "items": [_VO], "partial": True, "failedSources": ["memory"], "failedReasons": [{"source": "memory"}]}
    assert parse_pack(bare).partial and parse_pack(bare).failed_sources == ["memory"]
    assert len(parse_pack({"success": True, "data": bare}).items) == 1
    with pytest.raises(UpstreamError, match="50003"):
        parse_pack({"success": False, "code": 50003, "msg": "无法确定 user_id", "data": None})


def test_client_request_shape_and_cookie():
    c = ContextQueryClient(base_url="https://x/", source_types=["knowledge"], default_cookie="def=1", default_owner="svc", max_tokens=2000)
    body = c.build_body("DAU", "u1", None)
    assert body["query"] == "DAU" and body["owner"] == "u1"
    assert body["retrievalOptions"] == {"maxTokens": 2000, "sourceTypes": ["knowledge"]}
    assert c.url == "https://x/v1/context/retrieve"
    fake = mock.MagicMock()
    fake.__enter__.return_value.read.return_value = json.dumps({"items": [_VO]}).encode()
    with mock.patch("urllib.request.urlopen", return_value=fake) as uo:
        c._post(body, "sso=abc")
    assert uo.call_args[0][0].get_header("Cookie") == "sso=abc"


async def test_client_http_error_becomes_upstream_error():
    import urllib.error
    c = ContextQueryClient(base_url="https://x", source_types=[])
    err = urllib.error.HTTPError("u", 401, "unauth", {}, mock.MagicMock(read=lambda: b"sso"))
    with mock.patch("urllib.request.urlopen", side_effect=err), pytest.raises(UpstreamError, match="401"):
        await c.retrieve("q")


async def test_inconsistent_chunk_gets_discounted_utility(make_ctx, deps):
    class KH(FakeKnowledgeRetrieval):
        async def retrieve(self, query, **kw):
            return RetrievalPack(items=[chunk_to_item(_VO)])
    deps.kh = KH()
    res = await PreInjectionStage().run(make_ctx("DAU"))
    assert res.injected[0].relevance == pytest.approx(0.87 * 0.7, abs=1e-3)


# ── 路 2：历史 query 文件 ───────────────────────────────────────────────
def test_pair_expands_to_summary_and_knowledge_snapshots():
    pair = VerifiedPair(pair_id="p", query="最近7天DAU", terms=["DAU"], asset_ids=["t1"],
                        knowledge=[KnowledgeSnapshot(id="k1", kind=KnowledgeKind.METRIC_DEF, title="DAU", content="去重")],
                        hit_count=3, success_count=1)
    items = pair_to_items(pair, 0.8)
    assert [i.kind for i in items] == [KnowledgeKind.PAIR, KnowledgeKind.METRIC_DEF]
    assert items[1].id == "k1" and "t1" in items[0].content
    assert reliability(pair) == 0.5 and items[0].relevance == pytest.approx(0.4)


def test_history_file_roundtrip(tmp_path):
    f = tmp_path / "h.jsonl"
    s = JsonlPairStore(f)
    s.add(VerifiedPair(pair_id="p", query="q", terms=["DAU"], asset_ids=["t1"],
                       knowledge=[KnowledgeSnapshot(id="k", kind=KnowledgeKind.PAIR, title="a", content="b")]))
    s2 = JsonlPairStore(f)
    assert s2.get("p").asset_ids == ["t1"] and s2.get("p").knowledge[0].id == "k"


def test_seed_history_file_is_valid():
    from app.settings import settings
    store = JsonlPairStore(settings.PAIR_STORE_PATH)
    assert len(store.all()) >= 3
    assert all(p.query and p.asset_ids and p.knowledge for p in store.all())


# ── 阶段 ────────────────────────────────────────────────────────────────
async def test_stage_both_routes_hit_same_chunk(make_ctx, deps):
    deps.pair_store.add(VerifiedPair(pair_id="p1", query="最近7天DAU", terms=["DAU", "天各"], asset_ids=["dws.user_active_d"],
                                     knowledge=[KnowledgeSnapshot(id="k_dau", kind=KnowledgeKind.METRIC_DEF, title="DAU", content="日活跃用户数，按 user_id 去重，口径=当日有任意启动行为")]))
    res = await PreInjectionStage().run(make_ctx("昨天DAU是多少"))
    dau = next(i for i in res.injected if i.id == "k_dau")
    assert dau.routes == ["pair", "retrieval"]                      # 双路命中，只注入一份
    assert [i.id for i in res.injected].count("k_dau") == 1
    assert res.pair_hits >= 1 and res.retrieval_hits >= 1
    assert "DAU" in res.covered_terms and res.used_tokens <= res.token_budget
    assert res.route_status == {"pair": "ok", "retrieval": "ok"} and res.trace


async def test_stage_degrades_when_retrieval_fails(make_ctx, deps):
    class Boom(FakeKnowledgeRetrieval):
        async def retrieve(self, query, **kw):
            raise UpstreamError("s1", "401")
    deps.kh = Boom()
    deps.pair_store.add(VerifiedPair(pair_id="p1", query="最近7天DAU", terms=["DAU"], asset_ids=["t1"]))
    res = await PreInjectionStage().run(make_ctx("DAU"))
    assert res.route_status["retrieval"].startswith("failed") and res.route_status["pair"] == "ok"
    assert res.pair_hits >= 1


async def test_stage_both_routes_fail_still_returns_empty(make_ctx, deps):
    class Boom(FakeKnowledgeRetrieval):
        async def retrieve(self, query, **kw):
            raise UpstreamError("s1", "down")
    deps.kh = Boom()
    res = await PreInjectionStage().run(make_ctx("DAU"))
    assert res.injected == [] and res.uncovered_terms == ["DAU"]


async def test_stage_partial_pack_recorded(make_ctx, deps):
    class KH(FakeKnowledgeRetrieval):
        async def retrieve(self, query, **kw):
            return RetrievalPack(items=[], partial=True, failed_sources=["memory"])
    deps.kh = KH()
    res = await PreInjectionStage().run(make_ctx("DAU"))
    assert res.route_status["retrieval"] == "partial: ['memory']"


# ── 前端消费的明细 ──────────────────────────────────────────────────────
async def test_result_exposes_route_details_for_ui(make_ctx, deps):
    deps.pair_store.add(VerifiedPair(pair_id="p1", query="Last 30 days DAU by channel", terms=["DAU", "CHANNEL"], asset_ids=["dws.user_active_d"],
                                     knowledge=[KnowledgeSnapshot(id="k_dau", kind=KnowledgeKind.METRIC_DEF, title="DAU", content="日活跃用户数，按 user_id 去重，口径=当日有任意启动行为")]))
    res = await PreInjectionStage().run(make_ctx("Last 7 days DAU and GMV by channel"))
    assert [m.pair_id for m in res.pair_matches] == ["p1"] and res.pair_matches[0].similarity >= 0.3
    assert res.pair_matches[0].asset_ids == ["dws.user_active_d"] and res.pair_matches[0].knowledge_titles == ["DAU"]
    assert res.route_stats["pair"].count == 2 and res.route_stats["retrieval"].count >= 1
    assert res.total_value == pytest.approx(sum(t.delta for t in res.trace), abs=1e-3)


def test_english_stopwords_removed():
    from app.text_utils import extract_terms
    assert extract_terms("Last 7 days DAU and GMV by channel") == ["DAU", "GMV", "CHANNEL"]


def test_stage1_http_endpoint_and_cors():
    from fastapi.testclient import TestClient
    from app.main import create_app
    c = TestClient(create_app(Deps()))
    r = c.post("/api/stage1", json={"query": "最近7天各渠道DAU"}, headers={"Origin": "null"})
    body = r.json()
    assert r.status_code == 200 and body["result"]["injected"] and body["run_id"]
    assert r.headers["access-control-allow-origin"] == "*"
    assert c.options("/api/stage1", headers={"Origin": "null", "Access-Control-Request-Method": "POST"}).status_code == 200


# ── partial 的原因要看得见：界面 / 日志不能只有来源名 ───────────────────────────────────────────
def test_failed_reasons_are_summarised_for_the_ui():
    from app.stages.s1_preinjection import describe_pack_status
    pack = RetrievalPack(items=[], partial=True, failed_sources=["knowledge"],
                         failed_reasons=[{"source": "knowledge", "reason": "owner has no access to the knowledge base"}])
    assert describe_pack_status(pack) == "partial: ['knowledge'] · knowledge: owner has no access to the knowledge base"
    assert describe_pack_status(RetrievalPack(items=[], partial=True, failed_sources=["memory"])) == "partial: ['memory']"      # 没有原因时保持原样
    assert describe_pack_status(RetrievalPack(items=[])) == "ok"


@pytest.mark.parametrize("reason,expected", [
    ({"source": "knowledge", "message": "timeout"}, "knowledge: timeout"),
    ({"source": "knowledge", "errorMsg": "boom"}, "knowledge: boom"),
    ({"reason": "only reason"}, "only reason"),
    ({"source": "knowledge", "code": 500}, "knowledge: " + '{"source": "knowledge", "code": 500}'),
])
def test_failed_reason_shapes(reason, expected):
    from app.stages.s1_preinjection import describe_pack_status
    assert describe_pack_status(RetrievalPack(items=[], partial=True, failed_sources=["knowledge"], failed_reasons=[reason])).endswith(" · " + expected)


def test_long_reasons_are_truncated_and_capped_in_number():
    from app.stages.s1_preinjection import describe_pack_status
    st = describe_pack_status(RetrievalPack(items=[], partial=True, failed_sources=["knowledge"], failed_reasons=[{"source": f"s{i}", "reason": "x" * 500} for i in range(6)]))
    assert len(st) < 700 and st.count("s") >= 1 and "…" in st


async def test_stage_records_the_reason_in_route_status_and_logs_it(make_ctx, deps, monkeypatch):
    events = []
    import app.stages.s1_preinjection as s1
    monkeypatch.setattr(s1, "log_event", lambda ev, level=None, **f: events.append((ev, f)))

    class KH(FakeKnowledgeRetrieval):
        async def retrieve(self, query, **kw):
            return RetrievalPack(items=[], partial=True, failed_sources=["knowledge"], failed_reasons=[{"source": "knowledge", "reason": "no access"}])

    deps.kh = KH()
    res = await PreInjectionStage().run(make_ctx("DAU", deps=deps))
    assert res.route_status["retrieval"] == "partial: ['knowledge'] · knowledge: no access"
    assert any(ev == "s1.route_degraded" and "no access" in f["status"] for ev, f in events)
