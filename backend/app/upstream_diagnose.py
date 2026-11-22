"""上游连通性诊断：用最小请求 ping context-query，返回 status + 耗时 + 提示（对齐 old-version）。"""
from __future__ import annotations

import gzip
import json
import time
import urllib.error
import urllib.request
from typing import Any

from .auth import cookie_names, slim_for_upstream
from .settings import Settings, settings as _s

PING_QUERY = "ping"


def evaluate(status: int, parsed: Any) -> tuple[bool, str]:
    if status in (401, 403):
        return False, f"HTTP {status}：cookie 缺失或失效"
    if status == 404:
        return False, "HTTP 404：端点路径变更？"
    if 500 <= status < 600:
        return False, f"HTTP {status}：服务端错误"
    if status != 200:
        return False, f"HTTP {status}"
    if not isinstance(parsed, dict):
        return False, "响应为空或非 JSON"
    if parsed.get("success") is False:
        return False, f"业务错误 code={parsed.get('code')} msg={parsed.get('msg')}"
    return True, "ok"


def ping_context_query(*, cookie: str | None, cfg: Settings = _s, timeout: float = 8.0) -> dict[str, Any]:
    url = cfg.CONTEXT_QUERY_BASE_URL.rstrip("/") + "/v1/context/retrieve"
    body: dict[str, Any] = {"query": PING_QUERY, "retrievalOptions": {"maxTokens": 100, "sourceTypes": cfg.context_source_types()}}
    if cfg.CONTEXT_QUERY_OWNER:
        body["owner"] = cfg.CONTEXT_QUERY_OWNER
    headers = {"Content-Type": "application/json", "Accept-Encoding": "gzip"}
    raw_cookie = cookie or cfg.CONTEXT_QUERY_COOKIE
    if c := slim_for_upstream(raw_cookie):
        headers["Cookie"] = c
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    t0 = time.monotonic()
    out: dict[str, Any] = {"source": "context_query", "url": url, "cookie_names_sent": cookie_names(headers.get("Cookie"))}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
            raw, enc, status = r.read(), (r.headers.get("Content-Encoding") or "").lower(), r.status
    except urllib.error.HTTPError as e:
        raw, enc, status = e.read() if hasattr(e, "read") else b"", "", e.code
    except Exception as e:  # noqa: BLE001
        return out | {"ok": False, "status": None, "hint": f"{type(e).__name__}: {e}", "elapsed_ms": int((time.monotonic() - t0) * 1000)}
    try:
        parsed = json.loads((gzip.decompress(raw) if enc == "gzip" else raw).decode("utf-8", "replace"))
    except Exception:  # noqa: BLE001
        parsed = None
    ok, hint = evaluate(status, parsed)
    return out | {"ok": ok, "status": status, "hint": hint, "elapsed_ms": int((time.monotonic() - t0) * 1000)}


def diagnose_all(*, cookie: str | None) -> dict[str, Any]:
    r = ping_context_query(cookie=cookie)
    return {"context_query": r, "summary": {"ok_count": int(r["ok"]), "failed": [] if r["ok"] else [r["source"]]}}
