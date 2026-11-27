"""外网演示的三条示例问句：假数据要让每个面板都有内容（历史命中、注入/丢弃知识、候选/备选/丢弃/无权限），且每个指标都选得出主资产。"""
from pathlib import Path

import pytest

from app.deps import build_default_deps
from app.pipeline import stream_pipeline
from app.settings import Settings
from app.stages import RunContext

SEED = Path(__file__).resolve().parent.parent / "data" / "demo_history.jsonl"
QUESTIONS = {
    "Last 7 days DAU and CTR by platform": {"DAU", "CTR"},
    "Last 30 days CTR by platform": {"CTR"},
    "Last 7 days DAU and GMV by channel": {"DAU", "GMV"},
}


def demo_deps():
    return build_default_deps(Settings(_env_file=None, DEMO_MODE=True, PAIR_STORE_BACKEND="jsonl_ro", PAIR_STORE_PATH=SEED))


async def run(q):
    out = {}
    async for ev, p in stream_pipeline(RunContext(query=q, deps=demo_deps())):
        out[ev] = p.get("result") if isinstance(p, dict) else p
        out.setdefault("_events", []).append(ev)
    return out


@pytest.fixture(scope="module")
def results():
    import asyncio
    return {q: asyncio.run(run(q)) for q in QUESTIONS}


@pytest.mark.parametrize("q", list(QUESTIONS))
def test_every_metric_gets_a_primary_asset_and_nothing_blocks(results, q):
    sel = results[q]["stage5"]
    assert {m for d in sel["primary"] for m in d["covers"]} == QUESTIONS[q]
    assert not [f for f in sel["risk_flags"] if f.startswith(("no_asset_selected", "metrics_not_covered", "conflict"))]


@pytest.mark.parametrize("q", list(QUESTIONS))
def test_phase1_has_history_hits_and_a_multi_block_selection(results, q):
    pre = results[q]["stage1"]
    assert pre["pair_matches"] and len(pre["injected"]) >= 3 and pre["route_status"] == {"pair": "ok", "retrieval": "ok"}
    kinds = {i["kind"] for i in pre["injected"]}
    assert {"metric_def", "asset_meta"} <= kinds
    assert all(t["delta"] != 0 for t in pre["trace"])


@pytest.mark.parametrize("q", list(QUESTIONS))
def test_phase5_shows_alternates_and_rejected_candidates(results, q):
    sel = results[q]["stage5"]
    assert len(sel["alternates"]) >= 2 and len(sel["discarded"]) >= 1
    assert all(d["score"] and d["score"]["weights"] for d in sel["primary"])


def test_permission_gated_asset_is_offered_as_alternate_for_gmv(results):
    sel = results["Last 7 days DAU and GMV by channel"]["stage5"]
    assert any(d["needs_permission"] for d in sel["alternates"])


def test_ambiguity_exists_where_two_assets_tie_on_coverage(results):
    sel = results["Last 7 days DAU and CTR by platform"]["stage5"]
    for p in sel["primary"]:
        tie = [a for a in sel["alternates"] if set(a["covers"]) & set(p["covers"]) and abs(a["score"]["dims"]["coverage"] - p["score"]["dims"]["coverage"]) < 0.005]
        assert tie, p["asset_id"]


def test_the_three_questions_exercise_different_entry_levels(results):
    levels = {q: results[q]["stage3"]["level"] for q in QUESTIONS}
    assert levels["Last 7 days DAU and CTR by platform"] == "L2" and levels["Last 30 days CTR by platform"] == "L2"
    assert levels["Last 7 days DAU and GMV by channel"] == "L0"                 # 知识已覆盖全部子目标 → 跳过召回，直接进入选择
    assert results["Last 7 days DAU and GMV by channel"]["stage3"]["known_assets"]


def test_l2_questions_show_agent_collaboration(results):
    for q in ("Last 7 days DAU and CTR by platform", "Last 30 days CTR by platform"):
        st = results[q]["stage4"]["collab"]["stats"]
        assert st["retrievals"] >= 4 and st["reuse_hits"] + st["asks"] + st["handoffs"] >= 1


def test_demo_data_is_english_only():
    import re
    from app.services import demo_data as d
    text = " ".join(f"{a.name} {a.meta.description or ''}" for a, _ in d.DEMO_ASSETS) + " ".join(k.content + k.title for k in d.DEMO_KNOWLEDGE)
    assert not re.search(r"[\u4e00-\u9fff]", text)


@pytest.mark.parametrize(("query", "baseline_ids", "decoy_id", "decision"), [
    ("Last 7 days DAU and CTR by platform", {"app.user_active_d", "dws.ad_ctr_m"}, "dws.ad_ctr_m", "discarded"),
    ("Last 30 days CTR by platform", {"dws.ad_ctr_m"}, "dws.ad_ctr_m", "discarded"),
    ("Last 7 days DAU and GMV by channel", {"app.user_active_d", "ads.finance_gmv_audit"}, "ads.finance_gmv_audit", "alternates"),
])
def test_equal_budget_comparison_on_the_shared_pool(results, query, baseline_ids, decoy_id, decision):
    from app.demo import DEMO_CASE_ANSWERS

    candidates = results[query]["stage4"]["candidates"]
    selection = results[query]["stage5"]
    reference = DEMO_CASE_ANSWERS[query.casefold()]
    assert {metric: pick["asset_id"] for pick in selection["primary"] for metric in pick["covers"]} == reference
    ranked = sorted(candidates, key=lambda asset: (-asset["retrieval_score"], asset["asset_id"]))
    k = len(selection["primary"])
    assert {asset["asset_id"] for asset in ranked[:k]} == baseline_ids
    decoy = next(pick for pick in selection[decision] if pick["asset_id"] == decoy_id)
    metadata = next(asset["meta"] for asset in candidates if asset["asset_id"] == decoy_id)
    if decision == "discarded":
        assert metadata["grain"] == "month"
    else:
        assert metadata["permission_scope"] == "finance" and decoy["needs_permission"]
