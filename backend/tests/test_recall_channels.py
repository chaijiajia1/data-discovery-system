"""检索通道（对齐 old-version recall_bridge / recall_multi 的 bi · graph · knowledge 三路）。"""
import gzip
import json
from unittest import mock

import pytest

from app.errors import UpstreamError
from app.schemas.intent import Intent
from app.services.recall_channels import (
    BiChannel, FrameworkKnowledgeChannel, GraphChannel, intent_to_params, resolve_kb_intents, resolve_kb_top_k,
)


def _intent(**kw):
    base = dict(query="q", intent_type="data_fetch", domain="transaction", sub_domain=["交易驾驶舱"], keywords=["dgmv"])
    return Intent(**{**base, **kw})


# ── intent → 检索参数 ─────────────────────────────────────────────────────
def test_params_data_fetch_prefers_chart_and_maps_domain():
    p = intent_to_params(_intent())
    assert p["asset_types"] == ["chart", "dataset", "hive_table", "knowledge"] and p["top_k"] == 10
    assert p["knowledge_areas"] == ["交易"] and p["graph_domain"] == "交易"
    assert p["graph_sub_domain"] == ["交易驾驶舱"] and p["keywords"] == ["dgmv"]
    assert p["knowledge_context_only"] is True and p["knowledge_two_stage"] is False and p["graph_comm"] is False
    assert p["kb_intents"] == [] and p["kb_top_k"] == 0


def test_params_definition_makes_knowledge_a_primary_asset_and_drops_chart():
    p = intent_to_params(_intent(intent_type="definition"))
    assert p["asset_types"] == ["knowledge", "dataset", "hive_table"] and p["knowledge_context_only"] is False


def test_params_commercialization_two_stage_and_kb_topk():
    p = intent_to_params(_intent(domain="commercialization", sub_domain=["产品平台/聚光"], kb_intents=["definition", "diagnose"]))
    assert p["knowledge_two_stage"] and p["graph_comm"] and p["graph_domain"] == "商业化"
    assert p["kb_intents"] == ["definition", "diagnose"] and p["kb_top_k"] == 15      # 多意图取最大


def test_params_overseas_area_and_missing_domain():
    assert intent_to_params(_intent(domain="overseas", sub_domain=["翻译"]))["knowledge_areas"] == ["overseas"]
    assert intent_to_params(_intent(domain="overseas", sub_domain=["翻译"]))["graph_domain"] == "国际化"
    p = intent_to_params(_intent(domain=None, sub_domain=[]))
    assert p["knowledge_areas"] == [] and p["graph_domain"] == ""


def test_kb_intents_fallback_from_intent_type():
    assert resolve_kb_intents({"kb_intents": [], "intent_type": "analysis"}) == ["attribute"]
    assert resolve_kb_intents({"kb_intents": ["bogus ", "definition"]}) == ["bogus", "definition"]
    assert resolve_kb_top_k([]) == 10 and resolve_kb_top_k(["discovery"]) == 15


# ── HTTP 假实现 ───────────────────────────────────────────────────────────
class _Resp:
    def __init__(self, payload, gz=False):
        raw = json.dumps(payload, ensure_ascii=False).encode()
        self._raw, self.headers = (gzip.compress(raw), {"Content-Encoding": "gzip"}) if gz else (raw, {})

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _router(routes, seen=None):
    def fake(req, timeout=None):
        url = req.full_url
        body = json.loads(req.data) if req.data else None
        if seen is not None:
            seen.append((url, body, req.get_header("Cookie")))
        for frag, payload in routes.items():
            if frag in url:
                if isinstance(payload, Exception):
                    raise payload
                return _Resp(payload, gz=True)
        raise AssertionError(f"unexpected url {url}")
    return fake


BI_OK = {"success": True, "data": [
    {"assetType": "DATASET", "assetList": [{"id": 2479, "名称": "店播DGMV数据集", "链接": "https://x/ds/2479", "负责人": "a@x.com", "热度": 9, "has_perm": True, "score": 0.8}]},
    {"assetType": "CHART", "assetList": [{"chart_key": "p1/d1/c9", "名称": "店播趋势", "核心指标": ["店播DGMV"], "核心维度": ["日期"], "dataset_id": 2479}]},
    {"assetType": "KNOWLEDGE", "assetList": [{"id": "k"}]},
]}


async def test_bi_channel_normalizes_dataset_and_chart_and_forwards_cookie():
    seen = []
    with mock.patch("urllib.request.urlopen", side_effect=_router({"dataAsset/search": BI_OK}, seen)):
        out = await BiChannel().search("店播DGMV", limit=5, cookie="common-internal-access-token-prod=t; junk=1",
                                       params={"asset_types": ["chart", "dataset", "knowledge"], "keywords": ["店播"]})
    url, body, cookie = seen[0]
    assert url.endswith("/mark/openapi/dataAsset/search") and body["keyword"] == "店播DGMV" and body["topK"] == 5
    assert body["assetTypes"] == ["CHART", "DATASET"] and body["keywords"] == ["店播"] and body["searchMethod"] == "HYBRID"
    assert cookie == "common-internal-access-token-prod=t"                 # 只透传 SSO token
    ids = [a.asset_id for a in out]
    assert ids == ["dataset:2479", "chart:c9"]                             # knowledge 组被过滤
    assert out[0].asset_type == "dataset" and out[0].name == "店播DGMV数据集" and out[0].retrieval_score == 0.8
    assert out[1].asset_type == "chart" and out[1].meta.extra["metrics"] == ["店播DGMV"] and out[1].retrieval_score > 0


async def test_bi_channel_business_error_and_http_error():
    with mock.patch("urllib.request.urlopen", side_effect=_router({"dataAsset": {"success": False, "errorMsg": "boom"}})):
        with pytest.raises(UpstreamError, match="boom"):
            await BiChannel().search("q")
    import urllib.error
    err = urllib.error.HTTPError("u", 403, "Forbidden", {}, None)
    with mock.patch("urllib.request.urlopen", side_effect=_router({"dataAsset": err})):
        with pytest.raises(UpstreamError, match="HTTP 403"):
            await BiChannel().search("q")


GRAPH_OK = {"datasets": [{"dataset_id": "2479", "name": "DGMV数据集", "score": 0.9, "tags": ["核心看板"]}],
            "tables": [{"node_id": "table:dws.gmv_d", "name": "dws.gmv_d", "score": 0.7}],
            "charts": [{"node_id": "chart:c1", "dataset_id": "2479", "name": "图", "metric_ids": ["m"], "rule_text": "r", "score": 0.5}]}


async def test_graph_channel_generic_domain_uses_dwassetgraph():
    seen = []
    with mock.patch("urllib.request.urlopen", side_effect=_router({"dwassetgraph": GRAPH_OK}, seen)):
        out = await GraphChannel().search("DGMV", limit=8, params={"graph_domain": "交易", "graph_sub_domain": ["交易驾驶舱"], "keywords": ["dgmv"], "graph_comm": False})
    url, body, _ = seen[0]
    assert url.endswith("/api/recall") and body == {"q": "DGMV", "top_k": 8, "verbose": False, "domain": "交易", "sub_domain": ["交易驾驶舱"], "keywords": ["dgmv"]}
    assert {a.asset_id: a.asset_type for a in out} == {"dataset:2479": "dataset", "table:dws.gmv_d": "table", "chart:c1": "chart"}
    assert next(a for a in out if a.asset_id == "chart:c1").meta.extra["strict"] is True


async def test_graph_channel_commercialization_uses_kb_asset_finder_and_unwraps():
    seen = []
    wrapped = {"success": True, "code": 0, "data": {"datasets": [], "tables": [{"node_id": "table:t", "name": "t", "heat": 3}], "charts": [{"chart_id": "9", "dataset_id": "1", "name": "c"}]}}
    with mock.patch("urllib.request.urlopen", side_effect=_router({"kb-asset-finder": wrapped}, seen)):
        out = await GraphChannel().search("聚光消耗", params={"graph_comm": True, "graph_domain": "商业化", "asset_types": ["chart", "hive_table"]})
    url, body, _ = seen[0]
    assert url.endswith("/graph/recall") and body["query"] == "聚光消耗" and body["include_tables"] is True
    assert {a.asset_id for a in out} == {"table:t", "chart:9"}


async def test_graph_channel_error():
    with mock.patch("urllib.request.urlopen", side_effect=_router({"dwassetgraph": ["not a dict"]})):
        with pytest.raises(UpstreamError):
            await GraphChannel().search("q")


CHUNKS = [{"chunkId": "c1", "chunkTitle": "DGMV 口径", "content": "DGMV 为成交金额", "positioning": "指标口径", "businessArea": "交易", "metricsMentioned": "[\"DGMV\"]", "score": 0.8, "tables": "dws.gmv_d"}]


async def test_knowledge_channel_direct_and_chunk_dual_dedup():
    seen = []
    routes = {"chunk-retrieve": {"success": True, "data": CHUNKS + [{"chunkId": "c2", "chunkTitle": "别的", "content": "x", "score": 0.3}]},
              "framework/retrieve": {"success": True, "data": CHUNKS}}
    with mock.patch("urllib.request.urlopen", side_effect=_router(routes, seen)):
        out = await FrameworkKnowledgeChannel().search("DGMV 口径", limit=10, params={
            "knowledge_areas": ["交易"], "knowledge_subdomains": ["交易驾驶舱"], "keywords": ["dgmv"],
            "knowledge_two_stage": False, "knowledge_chunk_retrieve": True})
    urls = sorted(u.rsplit("/", 1)[-1] for u, _, _ in seen)
    assert urls == ["chunk-retrieve", "retrieve"]
    direct = next(b for u, b, _ in seen if u.endswith("/retrieve"))
    assert direct["businessAreas"] == ["交易"] and direct["subdomains"] == ["交易驾驶舱"] and "frameworkNames" not in direct
    assert direct["fieldTypes"] == ["positioning", "data_sources", "potential_questions", "content"]
    chunk = next(b for u, b, _ in seen if u.endswith("chunk-retrieve"))
    assert chunk["source"] == "dibp"
    assert [i.id for i in out] == ["c1", "c2"]                              # c1 双通道去重
    assert out[0].kind.value == "metric_def" and out[0].relevance == 0.8 and "DGMV" in out[0].content


async def test_knowledge_channel_commercialization_two_stage_passes_framework_names():
    seen = []
    routes = {"kb-graph-recall": {"success": True, "code": 0, "data": {"file_list": ["产品平台/Insight/勘误.md", "产品平台/Insight/勘误.md", "common/口径.md"]}},
              "framework/retrieve": {"success": True, "data": CHUNKS}}
    with mock.patch("urllib.request.urlopen", side_effect=_router(routes, seen)):
        await FrameworkKnowledgeChannel().search("聚光口径", params={
            "knowledge_areas": ["商业化"], "knowledge_subdomains": ["产品平台/Insight"], "keywords": [],
            "knowledge_two_stage": True, "kb_intents": ["definition"], "kb_top_k": 10, "knowledge_chunk_retrieve": False})
    coarse = next(b for u, b, _ in seen if "kb-graph-recall" in u)
    assert coarse["intents"] == ["definition"] and coarse["sub_domain"] == ["产品平台/Insight"] and coarse["top_k"] == 10
    fine = next(b for u, b, _ in seen if "framework/retrieve" in u)
    assert fine["frameworkNames"] == ["产品平台/Insight/勘误", "common/口径"]      # strip .md + 去重
    assert "subdomains" not in fine                                          # 精召不再叠 sub_domain


async def test_knowledge_channel_chunk_failure_degrades_to_direct_only():
    import urllib.error
    routes = {"chunk-retrieve": urllib.error.HTTPError("u", 500, "x", {}, None), "framework/retrieve": {"success": True, "data": CHUNKS}}
    with mock.patch("urllib.request.urlopen", side_effect=_router(routes)):
        out = await FrameworkKnowledgeChannel().search("q", params={"knowledge_areas": ["交易"], "knowledge_chunk_retrieve": True})
    assert [i.id for i in out] == ["c1"]


async def test_knowledge_channel_all_failed_raises():
    import urllib.error
    e = urllib.error.HTTPError("u", 500, "x", {}, None)
    with mock.patch("urllib.request.urlopen", side_effect=_router({"chunk-retrieve": e, "framework/retrieve": e})):
        with pytest.raises(UpstreamError):
            await FrameworkKnowledgeChannel().search("q", params={"knowledge_areas": ["交易"], "knowledge_chunk_retrieve": True})


async def test_graph_channel_uses_seed_assets_names_as_extra_keywords_for_lineage():
    seen = []
    with mock.patch("urllib.request.urlopen", side_effect=_router({"dwassetgraph": GRAPH_OK}, seen)):
        await GraphChannel().search("DAU", params={"graph_domain": "交易", "keywords": ["dau"], "seed_assets": ["dataset:1", "table:t"],
                                                     "seed_names": ["DAU 数据集", "dws_dau", "dau"]})
    body = seen[0][1]
    assert body["keywords"] == ["dau", "DAU 数据集", "dws_dau"]                         # 种子名并入关键词（去重、保序），图谱接口没有 seed 参数


async def test_graph_channel_without_seeds_is_unchanged():
    seen = []
    with mock.patch("urllib.request.urlopen", side_effect=_router({"dwassetgraph": GRAPH_OK}, seen)):
        await GraphChannel().search("DAU", params={"keywords": ["dau"]})
    assert seen[0][1]["keywords"] == ["dau"]


# ── 通道适配器要把真实接口的元信息写进 ⑤ 读的类型化字段（原来只塞进 meta.extra，⑤ 看不到）────────────────────────
async def test_bi_channel_fills_the_typed_meta_selection_reads():
    with mock.patch("urllib.request.urlopen", side_effect=_router({"dataAsset/search": BI_OK})):
        out = await BiChannel().search("店播DGMV", limit=5, params={"asset_types": ["chart", "dataset"]})
    ds, chart = out
    assert ds.meta.heat_30d == 9
    assert chart.meta.metric_ids == ["店播DGMV"] and chart.meta.supported_dims == ["日期"]
    assert chart.meta.is_strict is None                                       # 接口没给「是否严格」：如实留空，不臆测


async def test_graph_channel_fills_tags_heat_and_chart_metadata():
    data = {"datasets": [{"dataset_id": "2479", "name": "DGMV数据集", "score": 0.9, "tags": ["核心看板", "GMV"], "heat_30d": 7}],
            "tables": [{"node_id": "table:dws.gmv_d", "name": "dws.gmv_d", "score": 0.7, "heat_30d": 3}],
            "charts": [{"node_id": "chart:c1", "dataset_id": "2479", "name": "图", "metric_ids": ["m"], "rule_text": "r", "score": 0.5,
                        "supported_dimensions": [{"name": "渠道"}, {"field": "platform"}]},
                       {"node_id": "chart:c2", "dataset_id": "2479", "name": "图2", "metric_ids": ["m"], "score": 0.4}]}
    with mock.patch("urllib.request.urlopen", side_effect=_router({"dwassetgraph": data})):
        out = {a.asset_id: a for a in await GraphChannel().search("DGMV", limit=8, params={"graph_comm": False})}
    assert out["dataset:2479"].meta.tags == ["核心看板", "GMV"] and out["dataset:2479"].meta.heat_30d == 7
    assert out["table:dws.gmv_d"].meta.heat_30d == 3
    c1, c2 = out["chart:c1"].meta, out["chart:c2"].meta
    assert c1.metric_ids == ["m"] and c1.supported_dims == ["渠道", "platform"] and c1.is_strict is True
    assert c2.is_strict is False and c2.supported_dims == []                  # 有指标但没有取数规则：不严格；没给维度 = 无限制
