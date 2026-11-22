"""LLM 客户端抽象。

各阶段只依赖 `LLMClient` 协议：
- `NullLLM`：未配置 key 时使用，直接抛 LLMUnavailable，调用方走规则/启发式兜底；
- `OpenAICompatClient`：OpenAI chat/completions 兼容协议（stdlib 实现，无额外依赖）。调用方式对齐旧版 drr-service-debug 的
  runway 网关客户端：非流式、无重试、`api-key` 请求头（可切成 `Authorization: Bearer`）、请求体只有 model / max_tokens / messages
  （不强制 json mode，JSON 靠 prompt 约束 + `parse_llm_json` 兜底）、结构化日志只打元信息（不打 prompt / 响应正文 / key）、错误带状态码。
"""
from __future__ import annotations

import asyncio
import json
import socket
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

from .errors import UpstreamError
from .structured_log import log_event

DEFAULT_ENDPOINT = "https://runway.devops.rednote.life/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MAX_TOKENS = 8192            # 中文长输出（如 selection 的 reason schema）实测需要 8192


class LLMUnavailable(UpstreamError):
    def __init__(self, message: str = "LLM not configured"):
        super().__init__("llm", message)


class LLMClient(Protocol):
    async def complete(self, *, prompt: str, system: str | None = None, json_mode: bool = False) -> str: ...


class NullLLM:
    async def complete(self, *, prompt: str, system: str | None = None, json_mode: bool = False) -> str:
        raise LLMUnavailable()


class OpenAICompatClient:
    def __init__(self, *, endpoint: str, api_key: str, model: str, timeout_sec: float = 60.0, max_tokens: int = DEFAULT_MAX_TOKENS,
                 max_workers: int = 16, auth_header: str = "api-key", json_mode: bool = False):
        self.endpoint, self.api_key, self.model = endpoint, api_key, model
        self.timeout_sec, self.max_tokens = timeout_sec, max_tokens
        self.auth_header, self.json_mode = auth_header.strip().lower() or "api-key", json_mode
        # 独立线程池：LLM 调用的并发度由它决定，不与 asyncio 默认线程池里的检索通道 / 其他 to_thread 调用互相挤占
        self._pool = ThreadPoolExecutor(max_workers=max(1, max_workers), thread_name_prefix="llm")

    def _headers(self) -> dict[str, str]:
        h = {"content-type": "application/json"}
        if self.auth_header == "authorization":
            h["Authorization"] = f"Bearer {self.api_key}"
        else:
            h["api-key"] = self.api_key                      # runway 网关：不是 Authorization: Bearer
        return h

    def _call_sync(self, body: dict[str, Any]) -> str:
        t0 = time.monotonic()
        ms = lambda: int((time.monotonic() - t0) * 1000)       # noqa: E731
        msgs = body["messages"]
        log_event("llm.request", logger="app.llm", endpoint=self.endpoint, model=body["model"], max_tokens=body["max_tokens"],
                  prompt_chars=len(msgs[-1]["content"]), has_system=msgs[0]["role"] == "system")     # 只打元信息：不打 prompt / key
        req = urllib.request.Request(self.endpoint, data=json.dumps(body, ensure_ascii=False).encode("utf-8"), headers=self._headers(), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            log_event("llm.response", level=40, logger="app.llm", status=e.code, elapsed_ms=ms(), error_type="HTTPError", error_msg=f"{e.code} {e.reason}")
            raise UpstreamError("llm", f"HTTP {e.code} {e.reason}") from e
        except urllib.error.URLError as e:
            log_event("llm.response", level=40, logger="app.llm", elapsed_ms=ms(), error_type="URLError", error_msg=str(e.reason))
            raise UpstreamError("llm", f"URL error: {e.reason}") from e
        except (socket.timeout, TimeoutError) as e:
            log_event("llm.response", level=40, logger="app.llm", elapsed_ms=ms(), error_type="timeout", error_msg=str(e))
            raise UpstreamError("llm", f"timeout: {e}") from e
        try:
            data = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            log_event("llm.response", level=40, logger="app.llm", elapsed_ms=ms(), error_type="BadResponse", error_msg=str(e))
            raise UpstreamError("llm", f"response not json: {e}") from e
        choices = (data.get("choices") if isinstance(data, dict) else None) or []
        first = (choices[0] if choices else None) or {}
        text = ((first.get("message") or {}).get("content")) or ""
        log_event("llm.response", logger="app.llm", status=200, stop_reason=first.get("finish_reason"),    # 只发字符数，不发内容
                  usage=(data.get("usage") if isinstance(data, dict) else None) or {}, text_chars=len(text), elapsed_ms=ms())
        return text

    async def complete(self, *, prompt: str, system: str | None = None, json_mode: bool = False) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body: dict[str, Any] = {"model": self.model, "max_tokens": self.max_tokens, "messages": messages}
        if json_mode and self.json_mode:                       # 默认不发 response_format（与旧版一致）；确认网关支持后用 LLM_JSON_MODE 打开
            body["response_format"] = {"type": "json_object"}
        try:
            return await asyncio.get_running_loop().run_in_executor(self._pool, self._call_sync, body)
        except UpstreamError:
            raise
        except Exception as e:  # noqa: BLE001
            raise UpstreamError("llm", str(e)) from e


def build_llm(*, endpoint: str = "", api_key: str = "", model: str = "", timeout_sec: float = 60.0, max_workers: int = 16,
              max_tokens: int = DEFAULT_MAX_TOKENS, auth_header: str = "api-key", json_mode: bool = False) -> LLMClient:
    """只要有 key 就启用；endpoint / model 缺省用内部网关 runway 与 gpt-4o-mini（与旧版一致）。没有 key → NullLLM（各阶段走规则兜底）。"""
    if not api_key:
        return NullLLM()
    return OpenAICompatClient(endpoint=endpoint or DEFAULT_ENDPOINT, api_key=api_key, model=model or DEFAULT_MODEL, timeout_sec=timeout_sec,
                              max_tokens=max_tokens, max_workers=max_workers, auth_header=auth_header, json_mode=json_mode)
