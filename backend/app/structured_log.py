"""结构化 JSON 日志：每行一个 JSON；contextvars 注入 session_id / run_id。

用法：
    setup_logging(level="info")
    set_run("run-123")
    log_event("s4_recall.role_done", role="metric_scout", n=5, elapsed_ms=120)
"""
from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
import time
from typing import Any

_session_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("dd_session_id", default=None)
_run_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("dd_run_id", default=None)


def set_session(session_id: str | None) -> None:
    _session_ctx.set(session_id)


def set_run(run_id: str | None) -> None:
    _run_ctx.set(run_id)


_STD_KEYS = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module",
    "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs",
    "relativeCreated", "thread", "threadName", "processName", "process", "message",
    "asctime", "taskName",
})


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if (s := _session_ctx.get()) is not None:
            payload["session_id"] = s
        if (r := _run_ctx.get()) is not None:
            payload["run_id"] = r
        for k, v in record.__dict__.items():
            if k not in _STD_KEYS and not k.startswith("_"):
                payload[k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(level: str = "info") -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    if os.environ.get("DD_LOG_FORMAT") == "text":
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    else:
        handler.setFormatter(JsonFormatter())
    root.addHandler(handler)


def log_event(event: str, level: int = logging.INFO, **fields: Any) -> None:
    logging.getLogger("dd.event").log(level, event, extra=fields)
