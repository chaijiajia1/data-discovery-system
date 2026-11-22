"""④ LLM 基础设施：开关 / 整次召回的调用总预算 / 并发上限 / 超时与失败降级 / 调用统计；LLM 客户端独立线程池。"""
import asyncio
import json
import threading
import time

import pytest

from app.collab.llm import RecallLLM
from app.errors import UpstreamError
from app.llm_client import NullLLM, OpenAICompatClient
from app.settings import settings as _settings


class Scripted:
    """记录调用与并发峰值的 LLM 替身。"""

    def __init__(self, reply=None, exc=None, delay=0.0):
        self.reply = {"ok": 1} if reply is None else reply
        self.exc, self.delay = exc, delay
        self.prompts: list[str] = []
        self.active = self.peak = 0

    async def complete(self, *, prompt, system=None, json_mode=False):
        self.prompts.append(prompt)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.exc:
                raise self.exc
            return self.reply if isinstance(self.reply, str) else json.dumps(self.reply, ensure_ascii=False)
        finally:
            self.active -= 1


def cfg(**kw):
    return _settings.model_copy(update={"RECALL_LLM_ENABLED": True, **kw})


async def test_disabled_by_flag_never_calls_llm():
    llm = Scripted()
    g = RecallLLM(llm, _settings.model_copy(update={"RECALL_LLM_ENABLED": False}))
    assert g.enabled is False and await g.json("plan", "p") is None
    assert llm.prompts == [] and g.usage().calls == 0


async def test_null_llm_counts_as_disabled_even_when_flag_on():
    g = RecallLLM(NullLLM(), cfg())
    assert g.enabled is False and await g.json("plan", "p") is None and g.usage().calls == 0


async def test_returns_parsed_json_and_counts_by_purpose():
    g = RecallLLM(Scripted({"a": 1}), cfg())
    assert await g.json("plan", "p") == {"a": 1}
    assert await g.json("plan", "p") == {"a": 1} and await g.json("annotate", "p") == {"a": 1}
    u = g.usage()
    assert u.enabled and u.calls == 3 and u.failures == 0 and u.by_purpose == {"plan": 2, "annotate": 1}


async def test_total_call_budget_is_enforced_and_overflow_falls_back():
    llm = Scripted()
    g = RecallLLM(llm, cfg(RECALL_LLM_MAX_CALLS=2))
    assert [await g.json("plan", "p") for _ in range(4)] == [{"ok": 1}, {"ok": 1}, None, None]
    u = g.usage()
    assert len(llm.prompts) == 2 and u.calls == 2 and u.max_calls == 2 and u.budget_exhausted and u.fallbacks == 2


async def test_budget_holds_under_concurrency():
    llm = Scripted(delay=0.01)
    g = RecallLLM(llm, cfg(RECALL_LLM_MAX_CALLS=5, RECALL_LLM_CONCURRENCY=8))
    res = await asyncio.gather(*(g.json("plan", "p") for _ in range(12)))
    assert sum(r is not None for r in res) == 5 and len(llm.prompts) == 5


async def test_concurrency_cap():
    llm = Scripted(delay=0.03)
    g = RecallLLM(llm, cfg(RECALL_LLM_CONCURRENCY=2, RECALL_LLM_MAX_CALLS=100))
    res = await asyncio.gather(*(g.json("plan", "p") for _ in range(6)))
    assert all(r is not None for r in res) and llm.peak == 2


@pytest.mark.parametrize("llm", [Scripted(exc=UpstreamError("llm", "boom")), Scripted("not json"), Scripted(["x"]), Scripted(delay=0.3)])
async def test_failures_degrade_to_none_and_are_counted(llm):
    g = RecallLLM(llm, cfg(RECALL_LLM_TIMEOUT_SEC=0.05))
    assert await g.json("plan", "p") is None
    u = g.usage()
    assert u.failures == 1 and u.fallbacks == 1 and u.calls == 1 and not u.budget_exhausted


async def test_usage_is_json_serialisable():
    g = RecallLLM(Scripted(), cfg())
    await g.json("plan", "p")
    assert set(g.usage().model_dump()) >= {"enabled", "calls", "failures", "fallbacks", "max_calls", "budget_exhausted", "by_purpose"}


# ── LLM 客户端：独立线程池，并发度不被默认线程池限制 / 抢占 ─────────────────────────
async def test_openai_client_uses_a_dedicated_pool_with_configured_size(monkeypatch):
    c = OpenAICompatClient(endpoint="http://x", api_key="k", model="m", max_workers=3)
    state = {"active": 0, "peak": 0, "threads": set()}
    lock = threading.Lock()

    def fake_call(body):
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            state["threads"].add(threading.current_thread().name)
        time.sleep(0.05)
        with lock:
            state["active"] -= 1
        return "{}"

    monkeypatch.setattr(c, "_call_sync", fake_call)
    out = await asyncio.gather(*(c.complete(prompt="p") for _ in range(9)))
    assert out == ["{}"] * 9 and state["peak"] == 3
    assert all(t.startswith("llm") for t in state["threads"])             # 不在 asyncio 默认线程池里跑


async def test_openai_client_wraps_errors_as_upstream_error(monkeypatch):
    c = OpenAICompatClient(endpoint="http://x", api_key="k", model="m")
    monkeypatch.setattr(c, "_call_sync", lambda body: (_ for _ in ()).throw(RuntimeError("down")))
    with pytest.raises(UpstreamError):
        await c.complete(prompt="p")
