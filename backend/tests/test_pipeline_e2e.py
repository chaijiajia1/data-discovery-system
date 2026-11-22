import pytest
from fastapi.testclient import TestClient

from app.deps import Deps
from app.main import create_app
from app.pipeline import stream_pipeline
from app.stages import RunContext


async def test_missing_metric_never_pauses_for_user():
    evs = [e async for e in stream_pipeline(RunContext(query="帮我看看数据", deps=Deps()))]
    names = [e[0] for e in evs]
    assert "clarify" not in names and names[:3] == ["stage1", "stage2", "stage3"] and names[-1] == "done"


async def test_supplementary_metrics_input_is_optional_and_still_works():
    evs = [e async for e in stream_pipeline(RunContext(query="帮我看看数据", deps=Deps(), clarifications={"metrics": ["DAU"]}))]
    assert evs[-1][0] == "done" and evs[-1][1]["status"] == "success"


async def test_closed_loop_second_run_gets_pair_preinjected():
    deps = Deps()
    _ = [e async for e in stream_pipeline(RunContext(query="最近7天各渠道DAU", deps=deps))]
    ctx2 = RunContext(query="昨天各渠道DAU", deps=deps)
    _ = [e async for e in stream_pipeline(ctx2)]
    assert ctx2.preinjection.pair_hits >= 1                     # ⑥ → ① 回流生效
    pair = deps.pair_store.all()[0]
    assert pair.hit_count >= 1 and pair.success_count >= 1


def test_http_sync_and_sse():
    client = TestClient(create_app(Deps()))
    assert client.get("/api/health").json()["status"] == "ok"
    r = client.post("/api/discover/sync", json={"query": "最近7天DAU"})
    assert [e["event"] for e in r.json()["events"]] == ["stage1", "stage2", "stage3", "stage4", "stage5", "stage6", "done"]
    r = client.post("/api/discover", json={"query": "最近7天DAU"})
    assert r.headers["content-type"].startswith("text/event-stream")
    assert "event: stage6" in r.text and "event: done" in r.text
    assert client.get("/api/pairs").json()
    assert client.get("/api/prompts/coordination").status_code == 200
