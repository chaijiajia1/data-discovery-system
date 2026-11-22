"""context-query 客户端：POST {base}/v1/context/retrieve（路 1）。

- 鉴权：SSO。优先使用调用方透传的 Cookie，其次 settings.CONTEXT_QUERY_COOKIE；
  服务间调用可传 owner/userId 覆盖（owner 优先于 userId）。
- 响应兼容两种形态：裸 ContextPackVo，或 `{success, code, msg, data}` 包装。
- partial=true 时仍返回已有 items，并带上 failed_sources 供上层记录降级。
"""
from __future__ import annotations

import asyncio
import gzip
import json
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

from ..auth import slim_for_upstream
from ..errors import UpstreamError
from ..structured_log import log_event
from ..schemas.knowledge import KnowledgeItem, KnowledgeKind, KnowledgeOrigin, RetrievalPack
from ..text_utils import estimate_tokens

STAGE = "s1_preinjection.retrieval"

_KIND_KEYWORDS: list[tuple[KnowledgeKind, tuple[str, ...]]] = [
    (KnowledgeKind.METRIC_DEF, ("口径", "定义", "公式", "计算方式")),
    (KnowledgeKind.ASSET_META, ("字段", "表结构", "分区", "粒度", "元数据", "schema")),
    (KnowledgeKind.DIMENSION, ("维度",)),
]


def infer_kind(title: str, summary: str) -> KnowledgeKind:
    text = f"{title} {summary}".lower()
    for kind, kws in _KIND_KEYWORDS:
        if any(k in text for k in kws):
            return kind
    return KnowledgeKind.BUSINESS_TERM


def chunk_to_item(vo: dict[str, Any], *, inconsistent_discount: float = 1.0) -> KnowledgeItem:
    """ContextItemVo → KnowledgeItem。score 视为 [0,1] 相关度，越界则截断。"""
    title = vo.get("title") or vo.get("fileName") or ""
    content = vo.get("content") or vo.get("chunkSummary") or ""
    score = max(0.0, min(1.0, float(vo.get("score") or 0.0)))
    return KnowledgeItem(
        id=str(vo.get("chunkId") or f"{vo.get('sourceType', 'x')}:{title}"),
        kind=infer_kind(title, vo.get("chunkSummary") or ""),
        title=title, content=content, origin=KnowledgeOrigin.RETRIEVAL, routes=["retrieval"],
        relevance=score, token_cost=estimate_tokens(f"{title}\n{content}"),
        source_uri=vo.get("sourceUri") or "", consistency=vo.get("consistencyStatus") or "none",
    )


def _business_error(body: dict[str, Any]) -> str:
    """兼容 `{code,msg}`（文档）与 `{errorCode,errorMsg}`（实测）两种错误体。"""
    code, msg = body.get("errorCode", body.get("code")), body.get("errorMsg") or body.get("msg")
    if code == 50003:                                  # the upstream message is Chinese; say the same thing in English
        return f"code={code} msg=caller identity missing — sync the cookie first, or pass user_id / owner"
    return f"code={code} msg={msg}"


def parse_pack(body: dict[str, Any]) -> RetrievalPack:
    if body.get("success") is False:
        raise UpstreamError(STAGE, _business_error(body))
    data = body.get("data") if isinstance(body.get("data"), dict) else body
    reasons = data.get("failedReasons") or []
    return RetrievalPack(
        items=[chunk_to_item(i) for i in data.get("items") or []],
        context_pack_id=data.get("contextPackId"),
        partial=bool(data.get("partial")) and bool(data.get("failedSources") or data.get("failedReasons")),
        failed_sources=list(data.get("failedSources") or []),
        failed_reasons=[r if isinstance(r, dict) else {"reason": str(r)} for r in reasons],
    )


class ContextQueryClient:
    def __init__(
        self, *, base_url: str, source_types: list[str], timeout_sec: float = 10.0, default_cookie: str = "",
        default_owner: str = "", agent_id: str = "", max_tokens: int = 2000,
    ):
        self.url = base_url.rstrip("/") + "/v1/context/retrieve"
        self.source_types, self.timeout_sec = source_types, timeout_sec
        self.default_cookie, self.default_owner, self.agent_id, self.max_tokens = default_cookie, default_owner, agent_id, max_tokens

    def build_body(self, query: str, user_id: str | None, max_tokens: int | None) -> dict[str, Any]:
        body: dict[str, Any] = {"requestId": uuid.uuid4().hex, "query": query,
                                "retrievalOptions": {"maxTokens": max_tokens or self.max_tokens}}
        if self.source_types:
            body["retrievalOptions"]["sourceTypes"] = self.source_types
        if owner := (user_id or self.default_owner):
            body["owner"] = owner
        if self.agent_id:
            body["agentId"] = self.agent_id
        return body

    def _post(self, body: dict[str, Any], cookie: str | None) -> dict[str, Any]:
        headers = {"Content-Type": "application/json", "Accept-Encoding": "gzip"}
        if c := slim_for_upstream(cookie or self.default_cookie):
            headers["Cookie"] = c
        # 不打印 cookie 全文，仅记长度
        log_event("upstream.request", upstream="context_query", url=self.url, body=body, cookie_len=len(headers.get("Cookie", "")))
        t0 = time.monotonic()
        req = urllib.request.Request(self.url, data=json.dumps(body, ensure_ascii=False).encode(), headers=headers, method="POST")

        def _fail(kind: str, msg: str, status: int | None = None) -> UpstreamError:
            log_event("upstream.response", level=40, upstream="context_query", status=status, error_type=kind, error_msg=msg,
                      elapsed_ms=int((time.monotonic() - t0) * 1000))
            return UpstreamError(STAGE, msg)

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:  # noqa: S310
                raw = resp.read()
                if (resp.headers.get("Content-Encoding") or "").lower() == "gzip":
                    raw = gzip.decompress(raw)
            data = json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = ""
            try:  # 服务端业务错误常以 400 + JSON 返回，原因在响应体里
                raw_err = e.read()
                if (e.headers.get("Content-Encoding") or "").lower() == "gzip":
                    raw_err = gzip.decompress(raw_err)
                detail = _business_error(json.loads(raw_err.decode("utf-8")))
            except Exception:  # noqa: BLE001
                hint = " (cookie missing or expired)" if e.code in (401, 403) else ""
                detail = f"{e.reason}{hint}"
            raise _fail("HTTPError", f"HTTP {e.code}: {detail}", e.code) from e
        except urllib.error.URLError as e:
            raise _fail("URLError", f"URL error: {e.reason}") from e
        except (TimeoutError, OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
            raise _fail(type(e).__name__, f"{type(e).__name__}: {e}") from e
        log_event("upstream.response", upstream="context_query", status=200, elapsed_ms=int((time.monotonic() - t0) * 1000),
                  count=len((data.get("data") if isinstance(data.get("data"), dict) else data).get("items") or []))
        return data

    async def retrieve(self, query, *, user_id=None, cookie=None, max_tokens=None) -> RetrievalPack:
        body = self.build_body(query, user_id, max_tokens)
        return parse_pack(await asyncio.to_thread(self._post, body, cookie))

    async def search(self, query: str, *, limit: int = 10) -> list[KnowledgeItem]:
        return (await self.retrieve(query)).items[:limit]
