"""③ 的 HTTP 面：/api/stage3、/api/plan/evaluate、/api/roles、run 缓存。"""
from fastapi.testclient import TestClient

from app.deps import Deps
from app.main import create_app
from app.services.fakes import FakeKnowledgeRetrieval


class CountingKH(FakeKnowledgeRetrieval):
    def __init__(self):
        super().__init__()
        self.retrieves = 0

    async def retrieve(self, query, **kw):
        self.retrieves += 1
        return await super().retrieve(query, **kw)


def _client(**kw):
    return TestClient(create_app(Deps(**kw)))


def test_stage3_returns_levels_units_candidates_and_context():
    r = _client().post("/api/stage3", json={"query": "最近7天各渠道的DAU", "session_id": "ui", "max_parallel": 3})
    assert r.status_code == 200
    b = r.json()
    assert b["status"] == "ready" and b["contract"]["intent"]["domain"] and b["preinjection"]["injected"] is not None
    plan = b["plan"]
    assert plan["max_parallel"] == 3 and plan["context"]["max_parallel"] == 3
    assert {g["kind"] for g in plan["sub_goals"]} == {"metric", "dimension"} and all(g["status"] in ("covered", "doubtful", "uncovered") and "level" not in g for g in plan["sub_goals"])
    assert plan["level"] in ("L0", "L1", "L2") and plan["level_reason"] and plan["level_basis"] and plan["level_rule_id"].startswith("level.")
    assert plan["candidates"] and plan["chosen"]["name"] == plan["candidates"][0]["name"]
    c = plan["candidates"][0]
    assert {"score", "score_breakdown", "schedule", "covered_pairs", "feasible", "violations", "graph"} <= set(c)
    assert {"coverage", "parallelism", "cost", "agents", "work", "makespan"} <= set(c["score_breakdown"])
    ctx = plan["context"]
    assert {"units", "channels", "required_dims", "role_channels", "w_coverage", "w_parallel", "w_cost"} <= set(ctx) and "theta" not in ctx and "lam" not in ctx
    assert ctx["channels"] == ["bi", "graph", "knowledge"] and {"caliber", "channel"} <= set(ctx["units"][0])
    for cand in plan["candidates"]:           # A_max 按排程逐时间片检查
        span = [(x["start"], x["end"]) for x in cand["schedule"]]
        assert all(sum(1 for s, e in span if s <= t < e) <= 3 for t in range(max((e for _, e in span), default=0)))


def test_stage3_reuses_stage1_via_run_id_and_replans_with_new_a_max():
    kh = CountingKH()
    c = _client(kh=kh)
    s1 = c.post("/api/stage1", json={"query": "最近7天各渠道DAU"}).json()
    assert kh.retrieves == 1
    b = c.post("/api/stage3", json={"query": "最近7天各渠道DAU", "run_id": s1["run_id"], "max_parallel": 1}).json()
    assert kh.retrieves == 1 and b["run_id"] == s1["run_id"]                   # 不重复跑 ①
    assert b["plan"]["max_parallel"] == 1
    assert all(c["score_breakdown"]["makespan"] == c["score_breakdown"]["work"] for c in b["plan"]["candidates"])   # A_max=1：完全串行
    other = c.post("/api/stage3", json={"query": "换个问题 GMV", "run_id": s1["run_id"]}).json()
    assert kh.retrieves == 2 and other["run_id"] != s1["run_id"]               # query 不同 → 不复用


def _layers(graph):
    ids = [n["id"] for n in graph["nodes"]]
    indeg = {i: 0 for i in ids}
    for e in graph["edges"]:
        indeg[e["dst"]] += 1
    layers, left = [], dict(indeg)
    while left:
        cur = sorted(i for i, d in left.items() if d == 0)
        layers.append(cur)
        for i in cur:
            left.pop(i)
            for e in graph["edges"]:
                if e["src"] == i and e["dst"] in left:
                    left[e["dst"]] -= 1
    return layers


def test_stage3_missing_metric_still_plans_without_asking():
    b = _client().post("/api/stage3", json={"query": "帮我看看数据"}).json()
    assert b["status"] == "ready" and b["plan"] is not None and "open_questions" not in b["contract"]


def test_stage3_missing_metric_uses_keywords_as_inferred_metrics():
    b = _client().post("/api/stage3", json={"query": "千帆推广投放效果"}).json()
    m = b["contract"]["metrics"]
    assert b["status"] == "ready" and m["source"] == "inferred" and m["value"] == b["contract"]["intent"]["keywords"]


def test_stage3_confirmed_assets_make_execute_mode():
    b = _client().post("/api/stage3", json={"query": "DAU", "confirmed_assets": {"DAU": ["dws.user_active_d"]}}).json()
    assert b["plan"]["mode"] == "execute" and b["plan"]["known_assets"] == ["dws.user_active_d"] and b["plan"]["chosen"]["graph"]["nodes"] == []


def test_evaluate_custom_graph_scores_with_same_context():
    c = _client()
    s3 = c.post("/api/stage3", json={"query": "最近7天各渠道UV", "max_parallel": 4}).json()          # 演示知识里没有 UV → L2，才有待覆盖的对
    graph = {"nodes": [{"id": "bi_asset_discovery@UV", "role": "bi_asset_discovery", "targets": ["UV"]},
                       {"id": "caliber_lookup@UV", "role": "caliber_lookup", "targets": ["UV"]}],
             "edges": [{"src": "bi_asset_discovery@UV", "dst": "caliber_lookup@UV", "kind": "dep"}]}
    r = c.post("/api/plan/evaluate", json={"run_id": s3["run_id"], "graph": graph})
    assert r.status_code == 200
    cand = r.json()
    assert cand["source"] == "custom" and cand["score_breakdown"]["agents"] == 2 and cand["covered_pairs"]
    # 同一图在候选里出现过时，分数一致（单一事实源）
    same = next((x for x in s3["plan"]["candidates"] if {n["id"] for n in x["graph"]["nodes"]} == {n["id"] for n in graph["nodes"]}), None)
    if same:
        assert cand["score"] == same["score"]
    tight = c.post("/api/plan/evaluate", json={"run_id": s3["run_id"], "max_parallel": 1, "graph": {
        "nodes": graph["nodes"], "edges": []}}).json()
    assert tight["feasible"] is False and any("Peak concurrency" in v for v in tight["violations"])
    batched = c.post("/api/plan/evaluate", json={"run_id": s3["run_id"], "graph": {"nodes": [
        {"id": "o", "role": "omni_discovery", "targets": ["UV"]}],
        "edges": []}}).json()
    assert batched["feasible"] and batched["score_breakdown"]["coverage"] == 1.0 and batched["score_breakdown"]["work"] == 3   # 全通道角色独自提供全部评估维度；工作量 = 3 个通道


def test_evaluate_rejects_unknown_run_and_bad_graph():
    c = _client()
    assert c.post("/api/plan/evaluate", json={"run_id": "nope", "graph": {"nodes": [], "edges": []}}).status_code == 404
    s3 = c.post("/api/stage3", json={"query": "最近7天DAU"}).json()
    r = c.post("/api/plan/evaluate", json={"run_id": s3["run_id"], "graph": {"nodes": [{"id": "x", "role": "ghost"}], "edges": [{"src": "x", "dst": "y"}]}})
    assert r.status_code == 200 and r.json()["feasible"] is False


def test_roles_endpoint_exposes_templates_with_profiles():
    roles = {r["name"]: r for r in _client().get("/api/roles").json()}
    assert set(roles) == {"bi_asset_discovery", "lineage_discovery", "caliber_lookup", "omni_discovery", "dim_value_resolver"}          # 选择类角色已并入召回角色
    r = roles["lineage_discovery"]
    assert r["phase"] == "recall" and r["channels"] == ["graph"] and r["max_concurrency"] == 2
    assert roles["bi_asset_discovery"]["channels"] == ["bi"] and roles["caliber_lookup"]["channels"] == ["knowledge"]
    assert roles["omni_discovery"]["channels"] == ["bi", "graph", "knowledge"] and {"grain_fit", "freshness"} <= set(roles["bi_asset_discovery"]["dims"])
    assert "profile" not in r and "covers" not in r and "verifies" not in roles["caliber_lookup"] and all(x["phase"] == "recall" for x in roles.values())


def test_sse_stage3_event_carries_plan():
    ev = _client().post("/api/discover/sync", json={"query": "最近7天DAU", "max_parallel": 2}).json()["events"]
    s3 = next(e for e in ev if e["event"] == "stage3")
    assert s3["result"]["max_parallel"] == 2 and s3["result"]["sub_goals"]
    assert ev[-1]["event"] == "done"


def test_stage3_supplementary_answers_resume_and_optional_slots_are_user_sourced():
    c = _client()
    first = c.post("/api/stage3", json={"query": "帮我看看数据"}).json()
    assert first["status"] == "ready" and first["contract"]["metrics"]["source"] != "user"
    ans = {"metrics": ["DAU"], "time_range": "yesterday", "grain": "week", "dimensions": ["渠道"]}
    b = c.post("/api/stage3", json={"query": "帮我看看数据", "run_id": first["run_id"], "clarifications": ans}).json()
    ct = b["contract"]
    assert b["status"] == "ready" and b["plan"]
    assert {k: (ct[k]["value"], ct[k]["source"]) for k in ("metrics", "time_range", "grain", "dimensions")} == {
        "metrics": (["DAU"], "user"), "time_range": ("yesterday", "user"), "grain": ("week", "user"), "dimensions": (["渠道"], "user")}
    assert {g["kind"] for g in b["plan"]["sub_goals"]} == {"metric", "dimension"}
    # 之后因 A_max 变化重新规划时带同样的补充信息，用户给的值保持不变
    again = c.post("/api/stage3", json={"query": "帮我看看数据", "run_id": b["run_id"], "clarifications": ans, "max_parallel": 2}).json()
    assert again["status"] == "ready" and again["plan"]["max_parallel"] == 2
    # 不带补充信息则回到系统兜底，不会被追问
    assert c.post("/api/stage3", json={"query": "帮我看看数据", "run_id": b["run_id"], "max_parallel": 2}).json()["status"] == "ready"


# ── ④ 按用户确认的图运行：Roles / Communication 面板与图一致 ─────────────────────────
GRAPH = {"nodes": [{"id": "bi_asset_discovery@DAU", "role": "bi_asset_discovery", "targets": ["DAU"]},
                   {"id": "bi_asset_discovery@GMV", "role": "bi_asset_discovery", "targets": ["GMV"]},
                   {"id": "lineage_discovery@DAU", "role": "lineage_discovery", "targets": ["DAU"]},
                   {"id": "caliber_lookup@DAU", "role": "caliber_lookup", "targets": ["DAU"]}],
         "edges": [{"src": "bi_asset_discovery@DAU", "dst": "lineage_discovery@DAU", "kind": "dep"},
                   {"src": "bi_asset_discovery@DAU", "dst": "caliber_lookup@DAU", "kind": "dep"},
                   {"src": "bi_asset_discovery@GMV", "dst": "caliber_lookup@DAU", "kind": "dep"},
                   {"src": "lineage_discovery@DAU", "dst": "caliber_lookup@DAU", "kind": "dep"}]}


def test_stage4_runs_exactly_the_confirmed_graph_one_agent_per_node():
    c = _client()
    b = c.post("/api/stage4", json={"query": "最近7天DAU和GMV", "max_parallel": 4, "graph": GRAPH}).json()
    rec = b["recall"]
    assert [r["node_id"] for r in rec["role_reports"]] == [n["id"] for n in GRAPH["nodes"]]
    assert [a["agent_id"] for a in rec["collab"]["agents"]] == [n["id"] for n in GRAPH["nodes"]]
    assert {a["role"] for a in rec["collab"]["agents"]} == {"bi_asset_discovery", "lineage_discovery", "caliber_lookup"}
    rep = {a["agent_id"]: (a["replica"], a["replicas"]) for a in rec["collab"]["agents"]}
    assert rep["bi_asset_discovery@DAU"] == (1, 2) and rep["bi_asset_discovery@GMV"] == (2, 2) and rep["lineage_discovery@DAU"] == (1, 1)
    assert b["plan"]["chosen"]["graph"]["nodes"] and {n["id"] for n in b["plan"]["chosen"]["graph"]["nodes"]} == {n["id"] for n in GRAPH["nodes"]}
    assert rec["collab"]["events"] and all(e["a"] in rep and (e["b"] is None or e["b"] in rep) for e in rec["collab"]["events"])


def test_stage4_graph_override_changes_the_run_when_the_graph_changes():
    c = _client()
    small = {"nodes": GRAPH["nodes"][:1], "edges": []}
    a = c.post("/api/stage4", json={"query": "最近7天DAU", "graph": small}).json()["recall"]
    b = c.post("/api/stage4", json={"query": "最近7天DAU", "graph": GRAPH}).json()["recall"]
    assert [x["agent_id"] for x in a["collab"]["agents"]] == ["bi_asset_discovery@DAU"] and len(b["collab"]["agents"]) == 4


def test_stage4_rejects_unknown_roles_and_cyclic_graphs():
    c = _client()
    r = c.post("/api/stage4", json={"query": "最近7天DAU", "graph": {"nodes": [{"id": "x", "role": "ghost"}], "edges": []}})
    assert r.status_code == 400 and "ghost" in r.json()["detail"]
    cyc = {"nodes": GRAPH["nodes"][:2], "edges": [{"src": GRAPH["nodes"][0]["id"], "dst": GRAPH["nodes"][1]["id"]}, {"src": GRAPH["nodes"][1]["id"], "dst": GRAPH["nodes"][0]["id"]}]}
    assert c.post("/api/stage4", json={"query": "最近7天DAU", "graph": cyc}).status_code == 400


def test_stage4_without_graph_still_runs_the_planned_chosen_graph():
    b = _client().post("/api/stage4", json={"query": "最近7天DAU"}).json()
    assert [a["agent_id"] for a in b["recall"]["collab"]["agents"]] == [n["id"] for n in b["plan"]["chosen"]["graph"]["nodes"]]
