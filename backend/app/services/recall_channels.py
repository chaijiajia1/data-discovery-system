"""④ 检索通道：BI 向量检索 / 图谱血缘 / 知识框架（对齐 old-version `recall_bridge.py` + `recall_multi.py`）。

- `intent_to_params`：② 的 Intent → 各通道检索参数（与 old-version 同一套映射表）；
- `BiChannel` / `GraphChannel` / `FrameworkKnowledgeChannel`：stdlib urllib 实现（asyncio.to_thread），
  鉴权同源 SSO cookie，只透传 SSO token（避免请求头过大）；
- 三者分别实现 `BIAssetRetrieval` / `LineageRetrieval` / `CaliberRetrieval` 协议，经 Deps 注入，默认关闭回退 fakes。

通道差异不是规划的独立维度，而是体现在角色能力画像里（见 roles/builtin.py）。
"""
from __future__ import annotations

import asyncio
import gzip
import json
import logging
import urllib.error
import urllib.request
from typing import Any

from ..auth import slim_for_upstream
from ..errors import UpstreamError
from ..schemas.intent import Intent
from ..schemas.knowledge import KnowledgeItem, KnowledgeOrigin
from ..schemas.recall import AssetMeta, CandidateAsset
from ..settings import Settings, settings as _settings
from ..structured_log import log_event
from ..text_utils import estimate_tokens
from .context_query import infer_kind

logger = logging.getLogger(__name__)

# ── intent → 参数（design §4.2）──────────────────────────────────────────────
_PARAMS_BY_INTENT: dict[str, dict[str, Any]] = {
    "definition": {"asset_types": ["knowledge", "dataset", "hive_table"], "top_k": 10},
    "data_fetch": {"asset_types": ["chart", "dataset", "hive_table", "knowledge"], "top_k": 10},
    "discovery": {"asset_types": ["chart", "dataset", "hive_table", "knowledge"], "top_k": 15},
    "analysis": {"asset_types": ["chart", "dataset", "hive_table", "knowledge"], "top_k": 10},
    "other": {"asset_types": ["dataset", "hive_table", "knowledge"], "top_k": 10},
}
DOMAIN_TO_AREAS = {"commercialization": ["商业化"], "community": ["社区"], "transaction": ["交易"],
                   "resource_cost": ["资源成本"], "overseas": ["overseas"]}     # 知识库索引区域名：overseas 用英文
DOMAIN_EN_TO_ZH = {"commercialization": "商业化", "community": "社区", "transaction": "交易",
                   "resource_cost": "资源成本", "overseas": "国际化"}
_INTENT_TYPE_TO_KB_INTENTS = {"definition": ["definition"], "data_fetch": ["data_fetch_bi"], "discovery": ["discovery"],
                              "analysis": ["attribute"], "other": ["write_sql"]}
_KB_INTENT_TOPK = {"discovery": 15, "definition": 10, "data_fetch_bi": 15, "write_sql": 10, "diagnose": 15, "attribute": 15, "interpret": 10}
_KB_TOPK_DEFAULT = 10


def resolve_kb_intents(intent: dict[str, Any]) -> list[str]:
    kb = [x.strip() for x in (intent.get("kb_intents") or []) if isinstance(x, str) and x.strip()]
    return kb or _INTENT_TYPE_TO_KB_INTENTS.get((intent.get("intent_type") or "other").lower(), ["write_sql"])


def resolve_kb_top_k(kb_intents: list[str]) -> int:
    tops = [_KB_INTENT_TOPK[i] for i in kb_intents if i in _KB_INTENT_TOPK]
    return max(tops) if tops else _KB_TOPK_DEFAULT


def intent_to_params(intent: Intent) -> dict[str, Any]:
    d = intent.model_dump()
    itype = (intent.intent_type or "other").lower()
    spec = _PARAMS_BY_INTENT.get(itype, _PARAMS_BY_INTENT["other"])
    domain = (intent.domain or "").strip()
    is_comm = domain == "commercialization"
    kb_intents = resolve_kb_intents(d) if is_comm else []
    return {
        "asset_types": list(spec["asset_types"]), "top_k": spec["top_k"], "enrich": "perm,attrs",
        "knowledge_areas": list(DOMAIN_TO_AREAS.get(domain, [])), "knowledge_subdomains": list(intent.sub_domain),
        # 仅 definition 把 knowledge 视为可作 primary 的资产，其余只作 selection 的术语桥接 context
        "knowledge_context_only": itype != "definition",
        "knowledge_two_stage": is_comm, "knowledge_chunk_retrieve": True,
        "kb_intents": kb_intents, "kb_top_k": resolve_kb_top_k(kb_intents) if is_comm else 0,
        "graph_comm": is_comm, "keywords": list(intent.keywords),
        "graph_domain": DOMAIN_EN_TO_ZH.get(domain, domain), "graph_sub_domain": list(intent.sub_domain),
    }


# ── HTTP ────────────────────────────────────────────────────────────────────
def _post_sync(url: str, body: dict[str, Any], *, cookie: str | None, timeout: float, source: str, headers: dict[str, str] | None = None) -> Any:
    h = {"accept": "*/*", "accept-encoding": "gzip", "content-type": "application/json", **(headers or {})}
    slim = slim_for_upstream(cookie)
    if slim:
        h["Cookie"] = slim
    req = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode(), headers=h, method="POST")
    log_event("upstream.request", upstream=source, url=url, body=body)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
            raw = r.read()
            if (r.headers.get("Content-Encoding") or "").lower() == "gzip":
                raw = gzip.decompress(raw)
        return json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise UpstreamError(f"recall.{source}", f"HTTP {e.code} {e.reason}") from e
    except urllib.error.URLError as e:
        raise UpstreamError(f"recall.{source}", f"URL error: {e.reason}") from e
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise UpstreamError(f"recall.{source}", f"json parse failed: {e}") from e
    except OSError as e:  # socket.timeout 等
        raise UpstreamError(f"recall.{source}", f"{type(e).__name__}: {e}") from e


def _biz_ok(data: Any, source: str, ok_code: int) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise UpstreamError(f"recall.{source}", f"unexpected response type: {type(data).__name__}")
    if not data.get("success", data.get("code") == ok_code):
        raise UpstreamError(f"recall.{source}", f"business error: {data.get('errorMsg') or data.get('message')}")
    return data


def _rank_score(score: Any, rank: int) -> float:
    try:
        if score is not None:
            return max(0.0, min(1.0, float(score)))
    except (TypeError, ValueError):
        pass
    return round(max(0.1, 1.0 - 0.08 * (rank - 1)), 4)


def _int(v: Any) -> int | None:
    try:
        return int(v) if v is not None and not isinstance(v, bool) else None
    except (TypeError, ValueError):
        return None


def _strs(v: Any) -> list[str]:
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            return [v] if v.strip() else []
    return [str(x) for x in v] if isinstance(v, list) else []


class _Base:
    def __init__(self, s: Settings | None = None):
        self.s = s or _settings

    async def _post(self, url: str, body: dict[str, Any], cookie: str | None, source: str, **kw: Any) -> Any:
        return await asyncio.to_thread(_post_sync, url, body, cookie=cookie, timeout=self.s.RECALL_TIMEOUT_SEC, source=source, **kw)


# ── BI 向量检索 ───────────────────────────────────────────────────────────────
def _norm_bi(item: dict[str, Any], rank: int) -> CandidateAsset | None:
    kind = item.get("_asset_type")
    extra = {"url": item.get("链接"), "heat": item.get("热度"), "visibility": item.get("公共/私有"), "has_perm": item.get("has_perm"), "source": "bi"}
    owner = item.get("负责人")
    if kind == "dataset":
        if item.get("id") is None:
            return None
        return CandidateAsset(asset_id=f"dataset:{item['id']}", name=item.get("名称") or "", asset_type="dataset",
                              meta=AssetMeta(owner=owner, heat_30d=_int(item.get("热度")), extra=extra), retrieval_score=_rank_score(item.get("score"), rank))
    if kind == "chart":
        key = str(item.get("chart_key") or item.get("id") or "")
        cid = str(item.get("chart_id") or (key.split("/")[-1] if key else "")).strip()
        if not cid:
            return None
        extra |= {"metrics": _strs(item.get("核心指标")), "dims": _strs(item.get("核心维度")), "dataset_id": str(item.get("dataset_id") or "") or None,
                  "dashboard": item.get("看板")}
        return CandidateAsset(asset_id=f"chart:{cid}", name=item.get("名称") or "", asset_type="chart",
                              meta=AssetMeta(owner=owner, fields=extra["dims"], metric_ids=list(extra["metrics"]), supported_dims=list(extra["dims"]),   # ⑤ 读的是类型化字段；接口没给 is_strict，如实留空
                                             heat_30d=_int(item.get("热度")), extra=extra),
                              retrieval_score=_rank_score(item.get("score"), rank))
    return None


class BiChannel(_Base):
    """/mark/openapi/dataAsset/search（HYBRID）。BI 路只负责 dataset / chart（hive_table 归图谱路）。"""

    async def search(self, query: str, *, limit: int = 10, cookie: str | None = None, params: dict[str, Any] | None = None) -> list[CandidateAsset]:
        p = params or {}
        types = [t.upper() for t in (p.get("asset_types") or ["dataset"]) if t in ("dataset", "chart")] or ["DATASET"]
        body = {"keyword": query, "assetTypes": types, "searchMethod": "HYBRID", "topK": limit, "keywords": list(p.get("keywords") or [])}
        data = _biz_ok(await self._post(f"{self.s.RECALL_BI_BASE_URL}/mark/openapi/dataAsset/search", body, cookie, "bi",
                                        headers={"traceId": f"dd-bi-{abs(hash(query)) % 10**8}"}), "bi", 200)
        wanted = {t.lower() for t in types}
        out: list[CandidateAsset] = []
        for grp in data.get("data") or []:
            kind = (grp.get("assetType") or "").lower()
            if kind not in wanted:
                continue
            for item in grp.get("assetList") or []:
                if isinstance(item, dict):
                    a = _norm_bi({**item, "_asset_type": kind}, len(out) + 1)
                    if a:
                        out.append(a)
        return out[:limit]

    async def get(self, asset_ids: list[str]) -> list[CandidateAsset]:
        """按 id 物化已知资产（真实接口无按 id 查询；元信息缺失，由 ⑤ 的硬过滤按缺失处理）。"""
        return [CandidateAsset(asset_id=a, name=a, asset_type=_kind_of_id(a), meta=AssetMeta(extra={"source": "known"}), retrieval_score=0.5) for a in asset_ids]


def _kind_of_id(aid: str) -> str:
    return {"dataset": "dataset", "chart": "chart", "dashboard": "dashboard", "metric": "metric"}.get(aid.split(":", 1)[0], "table")


# ── 图谱（血缘）────────────────────────────────────────────────────────────────
def _norm_graph(data: dict[str, Any]) -> list[CandidateAsset]:
    out: list[CandidateAsset] = []
    for i, it in enumerate(data.get("datasets") or [], 1):
        did = it.get("dataset_id") or str(it.get("node_id", "")).removeprefix("dataset:")
        if did:
            out.append(CandidateAsset(asset_id=f"dataset:{did}", name=it.get("name") or "", asset_type="dataset",
                                      meta=AssetMeta(owner=it.get("owner_email"), tags=_strs(it.get("tags")), heat_30d=_int(it.get("heat_30d")),
                                                     extra={"tags": it.get("tags") or [], "heat": it.get("heat_30d"), "source": "graph"}),
                                      retrieval_score=_rank_score(it.get("score"), i)))
    for i, it in enumerate(data.get("tables") or [], 1):
        bare = str(it.get("node_id", "")).removeprefix("table:") or it.get("name") or ""
        if bare:
            out.append(CandidateAsset(asset_id=f"table:{bare}", name=it.get("name") or bare, asset_type="table",
                                      meta=AssetMeta(owner=it.get("owner_email"), heat_30d=_int(it.get("heat_30d", it.get("heat"))),
                                                     extra={"heat": it.get("heat_30d", it.get("heat")), "source": "graph"}),
                                      retrieval_score=_rank_score(it.get("score"), i)))
    for i, it in enumerate(data.get("charts") or [], 1):
        if it.get("is_onedash"):
            continue
        cid = str(it.get("chart_id") or str(it.get("node_id", "")).removeprefix("chart:")).strip()
        if not cid:
            continue
        dims = [d.get("name") or d.get("field") or "" for d in (it.get("supported_dimensions") or []) if isinstance(d, dict)]
        metrics = list(it.get("metric_ids") or [])
        out.append(CandidateAsset(asset_id=f"chart:{cid}", name=it.get("name") or "", asset_type="chart",
                                  meta=AssetMeta(fields=[d for d in dims if d], metric_ids=[str(m) for m in metrics], supported_dims=[d for d in dims if d],
                                                 is_strict=bool(metrics and it.get("rule_text")), heat_30d=_int(it.get("heat_30d", it.get("heat"))),
                                                 extra={"dataset_id": str(it.get("dataset_id") or "") or None, "metrics": metrics,
                                                                                         "strict": bool(metrics and it.get("rule_text")), "source": "graph"}),
                                  retrieval_score=_rank_score(it.get("score"), i)))
    return out


class GraphChannel(_Base):
    """通用图谱 /api/recall；商业化域整路替换为 kb-asset-finder /graph/recall。"""

    async def search(self, query: str, *, limit: int = 10, cookie: str | None = None, params: dict[str, Any] | None = None) -> list[CandidateAsset]:
        p = params or {}
        # 图谱接口没有「种子」参数：血缘种子（已有召回的资产名）并入 keywords，让图谱沿这些资产做语义 / 血缘匹配
        kw = list(dict.fromkeys([*(p.get("keywords") or []), *(p.get("seed_names") or [])]))[:12]
        p = {**p, "keywords": kw}
        if p.get("graph_comm"):
            body = {"query": query, "top_k": limit, "verbose": False, "domain": p.get("graph_domain") or "", "sub_domain": list(p.get("graph_sub_domain") or []),
                    "keywords": list(p.get("keywords") or []), "include_tables": "hive_table" in (p.get("asset_types") or ["hive_table"])}
            raw = _biz_ok(await self._post(f"{self.s.RECALL_COMM_GRAPH_BASE_URL}/graph/recall", body, cookie, "graph"), "graph", 0)
            data = raw.get("data")
            if not isinstance(data, dict):
                raise UpstreamError("recall.graph", f"missing data field: {type(data).__name__}")
            data = {**data, "charts": [{**c, "node_id": c.get("node_id") or f"chart:{c.get('chart_id')}"} for c in data.get("charts") or [] if isinstance(c, dict)]}
        else:
            body = {"q": query, "top_k": limit, "verbose": False, "domain": p.get("graph_domain") or "", "sub_domain": list(p.get("graph_sub_domain") or []),
                    "keywords": list(p.get("keywords") or [])}
            data = await self._post(f"{self.s.RECALL_GRAPH_BASE_URL}/api/recall", body, cookie, "graph")
            if not isinstance(data, dict):
                raise UpstreamError("recall.graph", f"unexpected response type: {type(data).__name__}")
        return sorted(_norm_graph(data), key=lambda a: a.retrieval_score, reverse=True)[: max(limit, 1) * 3]


# ── 知识框架（口径）────────────────────────────────────────────────────────────
_FIELD_TYPES = ["positioning", "data_sources", "potential_questions", "content"]     # 固定不可覆盖（覆盖成合成类型会导致召回为 0）


def _chunk_to_item(c: dict[str, Any]) -> KnowledgeItem | None:
    cid = (c.get("chunkId") or "").strip()
    if not cid:
        return None
    title = c.get("chunkTitle") or c.get("fileTitle") or cid
    content = c.get("content") or c.get("positioning") or ""
    kind = infer_kind(title, f"{c.get('positioning') or ''} {content}")
    tables = [t.strip() for t in str(c.get("tables") or "").replace("，", ",").split(",") if t.strip()]
    return KnowledgeItem(id=cid, kind=kind, title=title, content=content, origin=KnowledgeOrigin.RETRIEVAL, routes=["retrieval"],
                         terms=_strs(c.get("metricsMentioned")), asset_ids=tables,
                         relevance=_rank_score(c.get("score"), 1) if c.get("score") is not None else 0.3,
                         token_cost=estimate_tokens(f"{title}\n{content}"), source_uri=c.get("frameworkName") or "")


class FrameworkKnowledgeChannel(_Base):
    """framework/retrieve（老路，商业化走两段式：粗召筛文件 → 精召传 frameworkNames）+ chunk-retrieve（dibp 新路）并行取并集。"""

    async def _coarse(self, query: str, p: dict[str, Any], cookie: str | None) -> list[str]:
        body = {"query": query, "top_k": p.get("kb_top_k") or 10, "intents": list(p.get("kb_intents") or ["write_sql"]),
                "sub_domain": list(p.get("knowledge_subdomains") or []), "keywords": list(p.get("keywords") or [])}
        raw = _biz_ok(await self._post(f"{self.s.RECALL_KB_GRAPH_BASE_URL}/recall", body, cookie, "kb_graph"), "kb_graph", 0)
        out: list[str] = []
        for path in ((raw.get("data") or {}).get("file_list")) or []:
            fw = path.strip().removesuffix(".md") if isinstance(path, str) else ""
            if fw and fw not in out:
                out.append(fw)
        return out

    def _body(self, query: str, limit: int, p: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {"query": query, "businessAreas": list(p.get("knowledge_areas") or []), "fieldTypes": list(_FIELD_TYPES),
                                "topK": limit, "dedupByChunkId": True}
        if p.get("keywords"):
            body["keywords"] = list(p["keywords"])
        return body

    async def _direct(self, query: str, limit: int, p: dict[str, Any], cookie: str | None) -> list[dict[str, Any]]:
        body = self._body(query, limit, p)
        if p.get("knowledge_two_stage"):
            names = await self._coarse(query, p, cookie)       # 粗召失败向上传播，不静默掩盖
            if names:
                body["frameworkNames"] = names               # 精召由 frameworkNames 限定范围，不再叠 sub_domain
            elif p.get("knowledge_subdomains"):
                pass                                          # 粗召为空：退化为按 businessAreas 的宽召回
        elif p.get("knowledge_subdomains"):
            body["subdomains"] = list(p["knowledge_subdomains"])
        raw = _biz_ok(await self._post(f"{self.s.RECALL_KNOWLEDGE_BASE_URL}/mark/openapi/framework/retrieve", body, cookie, "knowledge"), "knowledge", 200)
        return list(raw.get("data") or [])

    async def _chunk(self, query: str, limit: int, p: dict[str, Any], cookie: str | None) -> list[dict[str, Any]]:
        body = {**self._body(query, limit, p), "source": "dibp"}
        if p.get("knowledge_subdomains"):
            body["subdomains"] = list(p["knowledge_subdomains"])
        raw = _biz_ok(await self._post(f"{self.s.RECALL_KNOWLEDGE_BASE_URL}/mark/openapi/framework/chunk-retrieve", body, cookie, "knowledge_chunk"), "knowledge_chunk", 200)
        return list(raw.get("data") or [])

    async def search(self, query: str, *, limit: int = 10, cookie: str | None = None, params: dict[str, Any] | None = None) -> list[KnowledgeItem]:
        p = params or {}
        jobs = [self._direct(query, limit, p, cookie)]
        if p.get("knowledge_chunk_retrieve"):
            jobs.append(self._chunk(query, limit, p, cookie))
        results = await asyncio.gather(*jobs, return_exceptions=True)
        ok = [r for r in results if not isinstance(r, BaseException)]
        if not ok:
            raise next(r for r in results if isinstance(r, BaseException))     # 全部失败才报错；单通道失败仅降级
        for r in results:
            if isinstance(r, BaseException):
                logger.warning("knowledge channel degraded: %s", r)
        out: dict[str, KnowledgeItem] = {}
        for rows in ok:
            for c in rows:
                it = _chunk_to_item(c) if isinstance(c, dict) else None
                if it and it.id not in out:
                    out[it.id] = it
        return list(out.values())[:limit]
