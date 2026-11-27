"""/api/stage5 复用 run 缓存里的 ④ 结果：前端刚跑完 ④（含用户确认的图）后接着跑 ⑤，不重跑 ③④，Phase 4 与 Phase 5 的候选池因此一致。"""
from fastapi.testclient import TestClient

from app.deps import Deps
from app.main import create_app
from app.settings import Settings
from app.services.fakes import FakeBIRetrieval


class SpyBI(FakeBIRetrieval):
    def __init__(self):
        super().__init__()
        self.calls: list[str] = []

    async def search(self, query, **kw):
        self.calls.append(query)
        return await super().search(query, limit=5)


GRAPH = {"nodes": [{"id": "bi_asset_discovery@DAU", "role": "bi_asset_discovery", "targets": ["DAU"]}], "edges": []}


def client():
    bi = SpyBI()
    return TestClient(create_app(Deps(bi=bi))), bi


def test_stage5_reuses_the_recall_stage4_just_produced():
    c, bi = client()
    s4 = c.post("/api/stage4", json={"query": "最近7天各渠道的DAU", "graph": GRAPH}).json()
    n = len(bi.calls)
    s5 = c.post("/api/stage5", json={"query": "最近7天各渠道的DAU", "run_id": s4["run_id"]}).json()
    assert s5["recall_reused"] is True and len(bi.calls) == n                                 # ④ 没有再跑
    assert s5["run_id"] == s4["run_id"] and s5["selection"]["primary"]
    assert [x["asset_id"] for x in s5["recall"]["candidates"]] == [x["asset_id"] for x in s4["recall"]["candidates"]]
    assert [r["node_id"] for r in s5["recall"]["role_reports"]] == ["bi_asset_discovery@DAU"]          # 用的是用户确认的那张图，不是重新规划的


def test_stage5_without_a_prior_recall_runs_the_earlier_stages_itself():
    c, bi = client()
    s5 = c.post("/api/stage5", json={"query": "最近7天各渠道的UV"}).json()                    # 演示知识里没有 UV → L2：要真的召回
    assert s5["recall_reused"] is False and bi.calls and s5["selection"] is not None


def test_a_different_query_or_changed_supplements_do_not_reuse_the_cached_recall():
    c, bi = client()
    s4 = c.post("/api/stage4", json={"query": "最近7天各渠道的DAU", "graph": GRAPH}).json()
    other = c.post("/api/stage5", json={"query": "最近7天各渠道的GMV", "run_id": s4["run_id"]}).json()
    assert other["recall_reused"] is False
    changed = c.post("/api/stage5", json={"query": "最近7天各渠道的DAU", "run_id": s4["run_id"], "clarifications": {"metrics": ["GMV"]}}).json()
    assert changed["recall_reused"] is False and changed["contract"]["metrics"]["value"] == ["GMV"]


def test_stage5_can_run_repeatedly_on_the_same_cached_recall():
    c, bi = client()
    s4 = c.post("/api/stage4", json={"query": "最近7天各渠道的DAU"}).json()
    n = len(bi.calls)
    a = c.post("/api/stage5", json={"query": "最近7天各渠道的DAU", "run_id": s4["run_id"]}).json()
    b = c.post("/api/stage5", json={"query": "最近7天各渠道的DAU", "run_id": s4["run_id"]}).json()
    assert a["recall_reused"] and b["recall_reused"] and len(bi.calls) == n


def test_l0_known_assets_need_no_cached_recall():
    c, bi = client()
    s5 = c.post("/api/stage5", json={"query": "DAU", "confirmed_assets": {"DAU": ["dws.user_active_d"]}}).json()
    assert s5["recall_reused"] is False and [d["asset_id"] for d in s5["selection"]["primary"]] == ["dws.user_active_d"] and bi.calls == []


def test_demo_answer_is_a_reference_independent_of_the_workflow():
    c = TestClient(create_app(Deps(settings=Settings(_env_file=None, DEMO_MODE=True))))
    q = "Last 7 days DAU and CTR by platform"
    # A workflow that only searches DAU must still be graded against both metrics.
    s4 = c.post("/api/stage4", json={"query": q, "graph": GRAPH}).json()
    result = c.post("/api/stage5", json={"query": q, "run_id": s4["run_id"]}).json()
    assert result["demo_answer"] == {"DAU": "app.user_active_d", "CTR": "app.ad_perf_board_d"}
    for query, expected in [
        ("Last 30 days CTR by platform", {"CTR": "app.ad_perf_board_d"}),
        ("Last 7 days DAU and GMV by channel", {"DAU": "app.user_active_d", "GMV": "app.trade_overview_d"}),
        ("Last 7 days CTR by region", None),
    ]:
        assert c.post("/api/stage5", json={"query": query}).json()["demo_answer"] == expected
    changed = c.post("/api/stage5", json={"query": q, "clarifications": {"metrics": ["GMV"]}}).json()
    assert changed["demo_answer"] is None


def test_live_mode_has_no_synthetic_reference():
    c, _ = client()
    result = c.post("/api/stage5", json={"query": "Last 30 days CTR by platform"}).json()
    assert result["demo_answer"] is None
