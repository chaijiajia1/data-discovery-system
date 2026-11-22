"""外网演示模式（DEMO_MODE）：只用假数据、不连内网、不用 LLM、不碰 cookie；公开接口要有限流；页面不带遥测脚本。"""
import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.demo import RateLimiter, client_ip, render_html
from app.deps import build_default_deps
from app.llm_client import NullLLM
from app.main import create_app
from app.services.fakes import FakeBIRetrieval, FakeKnowledgeRetrieval, FakeLineageRetrieval
from app.settings import Settings
from app.stores import JsonlPairStore, build_pair_store


DEMO_SEED = Path(__file__).resolve().parent.parent / "data" / "demo_history.jsonl"


def demo_settings(**kw):
    base = dict(_env_file=None, DEMO_MODE=True, LLM_API_KEY="k", CONTEXT_QUERY_ENABLED=True, RECALL_CHANNELS_ENABLED=True,
                RECALL_LLM_ENABLED=True, PLANNER_LLM_LEVEL_ENABLED=True, CONTEXT_QUERY_COOKIE="secret-cookie", ROLE_PROFILE_STORE_BACKEND="json")
    base.update(kw)
    return Settings(**base)


def demo_client(**kw):
    return TestClient(create_app(build_default_deps(demo_settings(**kw))))


# ── 装配：不管环境变量怎么写，演示模式只用假数据 ───────────────────────────────────────
def test_demo_mode_forces_fakes_no_llm_and_no_persistence_even_with_hostile_env():
    d = build_default_deps(demo_settings())
    assert isinstance(d.kh, FakeKnowledgeRetrieval) and isinstance(d.bi, FakeBIRetrieval) and isinstance(d.lineage, FakeLineageRetrieval)
    assert isinstance(d.llm, NullLLM)
    s = d.settings
    assert not (s.CONTEXT_QUERY_ENABLED or s.RECALL_CHANNELS_ENABLED or s.RECALL_LLM_ENABLED or s.PLANNER_LLM_LEVEL_ENABLED or s.CONTEXT_QUERY_COOKIE or s.LLM_API_KEY)
    assert s.ROLE_PROFILE_STORE_BACKEND == "memory" and s.PLANNER_LLM_EXTRA == 0


def test_default_mode_is_unchanged():
    assert Settings(_env_file=None).DEMO_MODE is False
    assert not isinstance(build_default_deps(Settings(_env_file=None, LLM_API_KEY="k")).llm, NullLLM)


# ── 对外暴露面 ────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("method,path", [("get", "/api/cookie/fetch"), ("post", "/api/cookie/diagnose"), ("get", "/api/me"), ("get", "/api/pairs"),
                                          ("get", "/api/prompts/selection"), ("get", "/docs"), ("get", "/redoc"), ("get", "/openapi.json")])
def test_internal_and_debug_endpoints_are_not_exposed_in_demo_mode(method, path):
    c = demo_client()
    assert getattr(c, method)(path).status_code == 404


def test_the_demo_endpoints_still_work_and_health_says_demo():
    c = demo_client()
    assert c.get("/api/health").json()["demo"] is True
    assert len(c.get("/api/roles").json()) == 5
    assert c.post("/api/stage1", json={"query": "Last 7 days DAU and GMV by channel"}).status_code == 200


def test_non_demo_still_exposes_them():
    c = TestClient(create_app(build_default_deps(Settings(_env_file=None, CONTEXT_QUERY_ENABLED=False))))
    assert c.get("/api/health").json()["demo"] is False and c.get("/api/pairs").status_code == 200 and c.get("/docs").status_code == 200


def test_client_supplied_cookies_are_ignored_so_the_server_never_calls_an_internal_host(monkeypatch):
    import app.main as m
    monkeypatch.setattr(m, "username_from_cookie", lambda c: (_ for _ in ()).throw(AssertionError("must not look the user up")))
    c = demo_client()
    r = c.post("/api/stage1", json={"query": "Last 7 days DAU by channel", "cookie": "common-internal-access-token-prod=x"},
               headers={"X-Forward-Cookie": "common-internal-access-token-prod=y", "Cookie": "a=b"})
    assert r.status_code == 200


def test_overlong_queries_are_rejected():
    assert demo_client().post("/api/stage1", json={"query": "x" * 2000}).status_code == 422


# ── 限流 ─────────────────────────────────────────────────────────────────────────────
def test_token_bucket_allows_a_burst_then_refills_with_time():
    now = [0.0]
    rl = RateLimiter(per_minute=6, burst=3, clock=lambda: now[0])
    assert [rl.allow("a")[0] for _ in range(4)] == [True, True, True, False]
    assert rl.allow("b")[0] is True                                         # 各 IP 独立
    ok, retry = rl.allow("a")
    assert ok is False and 9 <= retry <= 11                                 # 6/min = 每 10 秒补一个
    now[0] += 10
    assert rl.allow("a")[0] is True


def test_bucket_table_is_bounded():
    rl = RateLimiter(per_minute=60, burst=1, clock=lambda: 0.0, max_keys=50)
    for i in range(500):
        rl.allow(f"ip{i}")
    assert len(rl._buckets) <= 50


def test_the_rightmost_forwarded_address_is_the_client_the_trusted_proxy_saw():
    assert client_ip({"x-forwarded-for": "6.6.6.6, 1.2.3.4"}, "10.0.0.1") == "1.2.3.4"           # 左边是客户端自己写的，不可信
    assert client_ip({}, "10.0.0.1") == "10.0.0.1"


def test_http_rate_limit_applies_to_heavy_endpoints_only_and_is_per_client():
    c = demo_client(DEMO_RATE_LIMIT_PER_MIN=3, DEMO_RATE_BURST=3)
    h = {"X-Forwarded-For": "1.1.1.1"}
    codes = [c.post("/api/stage1", json={"query": "Last 7 days DAU by channel"}, headers=h).status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3:] == [429, 429]
    r = c.post("/api/stage1", json={"query": "Last 7 days DAU by channel"}, headers=h)
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1 and "rate" in r.json()["detail"].lower()
    assert c.post("/api/stage1", json={"query": "Last 7 days DAU by channel"}, headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 200
    assert all(c.get("/api/health", headers=h).status_code == 200 for _ in range(10))


def test_no_rate_limit_outside_demo_mode():
    c = TestClient(create_app(build_default_deps(Settings(_env_file=None, CONTEXT_QUERY_ENABLED=False, DEMO_RATE_LIMIT_PER_MIN=1, DEMO_RATE_BURST=1))))
    assert all(c.post("/api/stage1", json={"query": "DAU"}).status_code == 200 for _ in range(3))


# ── 页面 ────────────────────────────────────────────────────────────────────────────
SAVED = ('<html><head><!-- saved from url=(0045)https://cowork.xiaohongshu.com/f/sr-1/ -->\n<title>t</title>'
         '<script src="./X_files/uba-script.js"></script><script src="./X_files/apm-insight.min.js" async="" crossorigin="anonymous"></script>'
         '<script src="./X_files/104101b831udegmhb2q06chlqrg9000000000018ckkuf4.js"></script></head><body><script>var keep=1;</script></body></html>')


def test_render_html_strips_telemetry_and_flags_demo():
    out = render_html(SAVED, demo=True)
    assert "uba-script" not in out and "apm-insight" not in out and "104101b831" not in out and "cowork.xiaohongshu.com" not in out and "_files/" not in out
    assert "window.DD_DEMO=true" in out and "var keep=1" in out and out.index("DD_DEMO") < out.index("var keep")


def test_render_html_is_untouched_outside_demo_mode():
    assert render_html(SAVED, demo=False) == SAVED


def test_served_page_in_demo_mode_has_no_telemetry_and_no_static_file_mount():
    c = demo_client()
    r = c.get("/")
    assert r.status_code == 200 and "uba-script" not in r.text and "apm-insight" not in r.text and "DD_DEMO=true" in r.text
    assert c.get("/Data Discovery System · Interactive Walkthrough_files/uba-script.js").status_code == 404


# ── 种子数据只读 ─────────────────────────────────────────────────────────────────────
def test_readonly_jsonl_store_loads_the_seed_but_never_writes(tmp_path):
    from app.schemas.knowledge import VerifiedPair
    seed = tmp_path / "seed.jsonl"
    seed.write_text(json.dumps(VerifiedPair(pair_id="p1", query="q", terms=["Q"], asset_ids=["a"]).model_dump(mode="json")) + "\n")
    before = seed.read_text()
    st = build_pair_store("jsonl_ro", seed)
    assert [p.pair_id for p in st.all()] == ["p1"]
    st.add(VerifiedPair(pair_id="p2", query="q2", terms=["Q2"], asset_ids=["b"]))
    st.record_outcome(["p1"], True)
    assert {p.pair_id for p in st.all()} == {"p1", "p2"} and seed.read_text() == before          # 内存里生效，文件不动
    assert isinstance(build_pair_store("jsonl", seed), JsonlPairStore)


# ── 前端示例问句必须都能得到结果 ─────────────────────────────────────────────────────────
def _chips():
    html = open(Settings(_env_file=None).UI_HTML_PATH, encoding="utf-8").read()
    return re.findall(r'data-q="([^"]+)"', html)


def test_there_are_example_questions_and_each_is_english():
    chips = _chips()
    assert len(chips) == 3 and all(not re.search(r"[\u4e00-\u9fff]", q) for q in chips)


@pytest.mark.parametrize("i", [0, 1, 2])
def test_every_example_question_selects_a_primary_asset_for_every_metric_on_demo_data(i):
    q = _chips()[i]
    b = demo_client(PAIR_STORE_BACKEND="jsonl_ro", PAIR_STORE_PATH=DEMO_SEED).post("/api/stage5", json={"query": q}).json()
    sel = b["selection"]
    assert b["contract"]["metrics"]["value"], q
    assert sel["primary"] and not any(f.startswith(("no_asset_selected", "metrics_not_covered")) for f in sel["risk_flags"]), (q, sel["risk_flags"])
    assert not re.search(r"[\u4e00-\u9fff]", json.dumps([d["asset_id"] for d in sel["primary"]], ensure_ascii=False))


def test_demo_seed_is_english_only_and_loads():
    assert not re.search(r"[\u4e00-\u9fff]", DEMO_SEED.read_text(encoding="utf-8"))
    assert len(build_pair_store("jsonl_ro", DEMO_SEED).all()) == 3


def test_the_third_example_skips_recall_while_the_first_two_run_the_full_flow():
    c = demo_client(PAIR_STORE_BACKEND="jsonl_ro", PAIR_STORE_PATH=DEMO_SEED)
    lv = [c.post("/api/stage3", json={"query": q}).json()["plan"]["level"] for q in _chips()]
    assert lv[0] == "L2" and lv[1] == "L2" and lv[2] == "L0"
