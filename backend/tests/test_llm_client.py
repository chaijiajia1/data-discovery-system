"""LLM 客户端：对齐旧版 drr-service-debug 的 runway 网关调用方式（api-key 头 / 8192 max_tokens / 不强制 json mode / 结构化日志 / 带码错误 / 截断 JSON 兜底）。"""
import io
import json
import socket
import urllib.error

import pytest

from app import llm_client as lc
from app.errors import UpstreamError
from app.llm_client import NullLLM, OpenAICompatClient, build_llm
from app.settings import Settings
from app.stages.s2_coordination.agent import parse_llm_json


class FakeResp:
    def __init__(self, payload):
        self.raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()

    def read(self):
        return self.raw

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


OK = {"choices": [{"message": {"content": '{"a": 1}'}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}


@pytest.fixture
def wire(monkeypatch):
    """替换 urlopen，记录请求；返回 (captured, set_response)。"""
    cap: dict = {}
    state = {"resp": OK, "exc": None}

    def fake_urlopen(req, timeout=None):
        cap["url"], cap["timeout"] = req.full_url, timeout
        cap["headers"] = {k.lower(): v for k, v in req.header_items()}
        cap["body"] = json.loads(req.data.decode("utf-8"))
        cap["method"] = req.get_method()
        if state["exc"] is not None:
            raise state["exc"]
        return FakeResp(state["resp"])

    monkeypatch.setattr(lc.urllib.request, "urlopen", fake_urlopen)
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(lc, "log_event", lambda ev, level=None, **f: events.append((ev, f)))
    cap["state"], cap["events"] = state, events
    return cap


def client(**kw):
    return OpenAICompatClient(endpoint="https://gw.example/v1/chat/completions", api_key="SECRET-KEY", model="gpt-4o-mini", **kw)


# ── 请求格式 ───────────────────────────────────────────────────────────────────
async def test_request_uses_api_key_header_and_minimal_body(wire):
    out = await client().complete(prompt="你好 PROMPT-TEXT")
    assert out == '{"a": 1}' and wire["method"] == "POST" and wire["url"] == "https://gw.example/v1/chat/completions"
    assert wire["headers"]["api-key"] == "SECRET-KEY" and "authorization" not in wire["headers"] and wire["headers"]["content-type"] == "application/json"
    assert wire["body"] == {"model": "gpt-4o-mini", "max_tokens": 8192, "messages": [{"role": "user", "content": "你好 PROMPT-TEXT"}]}


async def test_system_prompt_goes_first_and_timeout_is_passed(wire):
    await client(timeout_sec=12.5).complete(prompt="p", system="sys")
    assert wire["body"]["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "p"}] and wire["timeout"] == 12.5


async def test_bearer_auth_is_selectable(wire):
    await client(auth_header="authorization").complete(prompt="p")
    assert wire["headers"]["authorization"] == "Bearer SECRET-KEY" and "api-key" not in wire["headers"]


async def test_json_mode_is_off_by_default_even_when_requested_and_can_be_enabled(wire):
    await client().complete(prompt="p", json_mode=True)
    assert "response_format" not in wire["body"]
    await client(json_mode=True).complete(prompt="p", json_mode=True)
    assert wire["body"]["response_format"] == {"type": "json_object"}
    await client(json_mode=True).complete(prompt="p", json_mode=False)
    assert "response_format" not in wire["body"]


async def test_max_tokens_is_configurable(wire):
    await client(max_tokens=1000).complete(prompt="p")
    assert wire["body"]["max_tokens"] == 1000


# ── 响应与日志 ─────────────────────────────────────────────────────────────────
async def test_logs_metadata_only_never_prompt_response_or_key(wire):
    await client().complete(prompt="PROMPT-TEXT", system="s")
    assert [e for e, _ in wire["events"]] == ["llm.request", "llm.response"]
    req, resp = wire["events"][0][1], wire["events"][1][1]
    assert req["prompt_chars"] == len("PROMPT-TEXT") and req["has_system"] is True and req["model"] == "gpt-4o-mini" and req["max_tokens"] == 8192
    assert resp["status"] == 200 and resp["stop_reason"] == "stop" and resp["usage"]["prompt_tokens"] == 3 and resp["text_chars"] == len('{"a": 1}')
    blob = json.dumps(wire["events"], ensure_ascii=False)
    assert "PROMPT-TEXT" not in blob and "SECRET-KEY" not in blob and '"a": 1' not in blob


async def test_truncated_output_is_flagged_but_still_returned(wire):
    wire["state"]["resp"] = {"choices": [{"message": {"content": '{"a": [1, 2'}, "finish_reason": "length"}]}
    assert await client().complete(prompt="p") == '{"a": [1, 2'
    assert wire["events"][-1][1]["stop_reason"] == "length"


async def test_empty_content_returns_empty_string(wire):
    wire["state"]["resp"] = {"choices": [{"message": {"content": None}}]}
    assert await client().complete(prompt="p") == ""


# ── 错误：带码的 UpstreamError ──────────────────────────────────────────────────────
@pytest.mark.parametrize("exc,needle", [
    (urllib.error.HTTPError("u", 401, "Unauthorized", {}, io.BytesIO(b"bad key")), "HTTP 401"),
    (urllib.error.HTTPError("u", 403, "Forbidden", {}, io.BytesIO(b"")), "HTTP 403"),
    (urllib.error.URLError("dns down"), "URL error"),
    (socket.timeout("slow"), "timeout"),
])
async def test_transport_errors_become_upstream_errors_with_the_cause(wire, exc, needle):
    wire["state"]["exc"] = exc
    with pytest.raises(UpstreamError) as e:
        await client().complete(prompt="p")
    assert needle in str(e.value) and "SECRET-KEY" not in str(e.value)
    assert wire["events"][-1][0] == "llm.response" and wire["events"][-1][1].get("error_type")


@pytest.mark.parametrize("payload", [b"<html>", {"choices": []}, {}])
async def test_malformed_responses_become_upstream_errors(wire, payload):
    wire["state"]["resp"] = payload
    if payload == {"choices": []} or payload == {}:
        assert await client().complete(prompt="p") == ""                    # 与旧版一致：没有 choices 视为空文本，由上层按"不可解析"回落
    else:
        with pytest.raises(UpstreamError):
            await client().complete(prompt="p")


# ── 配置与装配 ─────────────────────────────────────────────────────────────────
def test_defaults_point_at_the_internal_gateway_and_only_the_key_is_required():
    s = Settings(_env_file=None)
    assert s.LLM_ENDPOINT == "https://runway.devops.rednote.life/v1/chat/completions" and s.LLM_MODEL == "gpt-4o-mini"
    assert s.LLM_MAX_TOKENS == 8192 and s.LLM_AUTH_HEADER == "api-key" and s.LLM_JSON_MODE is False and s.LLM_API_KEY == ""


def test_build_llm_needs_only_a_key_and_falls_back_to_defaults():
    assert isinstance(build_llm(endpoint="", api_key="", model=""), NullLLM)
    c = build_llm(endpoint="", api_key="k", model="")
    assert isinstance(c, OpenAICompatClient) and c.endpoint.startswith("https://runway.devops.rednote.life") and c.model == "gpt-4o-mini"
    c2 = build_llm(endpoint="http://x", api_key="k", model="m", max_tokens=1234, auth_header="authorization", json_mode=True)
    assert (c2.endpoint, c2.model, c2.max_tokens, c2.auth_header, c2.json_mode) == ("http://x", "m", 1234, "authorization", True)


def test_deps_wires_all_llm_settings(monkeypatch):
    from app.deps import build_default_deps
    s = Settings(_env_file=None, LLM_API_KEY="k", LLM_MAX_TOKENS=777, LLM_AUTH_HEADER="authorization", LLM_JSON_MODE=True, CONTEXT_QUERY_ENABLED=False)
    d = build_default_deps(s)
    assert isinstance(d.llm, OpenAICompatClient) and d.llm.max_tokens == 777 and d.llm.auth_header == "authorization" and d.llm.json_mode is True


# ── 截断 JSON 的兜底（旧版 call_json 的 partial parse）──────────────────────────────────
def test_parse_llm_json_recovers_a_truncated_object():
    out = parse_llm_json('{"plans": [{"metric": "DAU", "query": "DAU 日活"}, {"metric": "GMV", "que')
    assert out == {"plans": [{"metric": "DAU", "query": "DAU 日活"}]}


def test_parse_llm_json_recovers_truncated_fenced_output():
    out = parse_llm_json('```json\n{"judgements": [{"asset_id": "a", "relevant": true}, {"asset_id": "b", "rele')
    assert out == {"judgements": [{"asset_id": "a", "relevant": True}]}


@pytest.mark.parametrize("raw", ["not json", "", "[1, 2", '{"a": "unterminated', "{"])
def test_parse_llm_json_still_rejects_garbage(raw):
    with pytest.raises(ValueError):
        parse_llm_json(raw)


def test_parse_llm_json_complete_output_is_unchanged():
    assert parse_llm_json('前言 {"a": 1} 后记') == {"a": 1} and parse_llm_json('```json\n{"a": [1]}\n```') == {"a": [1]}
