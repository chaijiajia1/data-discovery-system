"""⑤ 读 ④ 的证据，而不是各算各的：粒度冲突 / 新鲜度 / LLM 相关性评估 / 佐证核验 → 软降权、风险标记、盲区信号、进 LLM prompt。
权限仍是 ⑤ 独有（④ 不做），硬护栏（无指标字段等）不被 ④ 的 LLM 结论替代。"""
from app.schemas.knowledge import KnowledgeItem
from app.schemas.recall import AssetMeta, CandidateAsset, Conflict, Evidence, EvidenceKind
from app.schemas.intent import Intent
from app.schemas.task import Slot, TaskContract, ValueSource
from app.stages.s5_selection import SelectionInput, rule_select
from app.stages.s5_selection.agent import build_prompt
from app.stages.s5_selection.rules import build_facts, evaluate_group


def C(metric="DAU", grain=None, dims=()):
    return TaskContract(query="q", metrics=Slot(value=[metric], source=ValueSource.USER), grain=Slot(value=grain, source=ValueSource.USER),
                        dimensions=Slot(value=list(dims), source=ValueSource.USER), intent=Intent(query="q", domain="transaction"))


def A(i, *, grain="day", fields=("dt", "dau"), score=0.8, typ="dataset"):
    return CandidateAsset(asset_id=i, name=i, asset_type=typ, retrieval_score=score,
                          meta=AssetMeta(grain=grain, partition_field="dt", partition_latest="2026-11-12", fields=list(fields), primary_keys=list(fields[:1])))


def ev(asset, kind, source, value, statement="x", subject="", conf=0.6, i=[0]):
    i[0] += 1
    return Evidence(id=f"e{i[0]}", asset_id=asset, kind=kind, statement=statement, value=value, role="r", source=source, confidence=conf, subject=subject)


def S(contract, cands, **kw):
    return SelectionInput(contract=contract, candidates=list(cands), **kw)


def tiers(res):
    return ([d.asset_id for d in res.primary], [d.asset_id for d in res.alternates], [d.asset_id for d in res.discarded])


# ── 粒度：④ 合并时静默择一，冲突只留在 Conflict 里；⑤ 不能据此直接丢弃 ────────────────────────────────
def test_grain_hard_discard_still_applies_without_an_upstream_conflict():
    res = rule_select(S(C(grain="day"), [A("m", grain="month"), A("d")]))
    assert tiers(res) == (["d"], [], ["m"]) and "grain_hard" in res.discarded[0].rule_ids


def test_a_grain_conflict_that_includes_an_acceptable_value_is_not_discarded():
    conflict = Conflict(asset_id="m", topic=EvidenceKind.GRAIN, evidence_ids=["e1"], values=["month", "day"])
    res = rule_select(S(C(grain="day"), [A("m", grain="month"), A("d")], conflicts=[conflict]))
    assert tiers(res) == (["d"], ["m"], []) and "grain_conflict" in res.alternates[0].rule_ids
    assert "Phase 4" in res.alternates[0].reason.conclusion and any("grain" in x for x in res.alternates[0].reason.downweights)


def test_a_grain_conflict_whose_every_value_is_too_coarse_is_still_discarded():
    conflict = Conflict(asset_id="m", topic=EvidenceKind.GRAIN, evidence_ids=["e1"], values=["month", "year"])
    assert tiers(rule_select(S(C(grain="day"), [A("m", grain="month"), A("d")], conflicts=[conflict])))[2] == ["m"]


def test_a_sole_candidate_with_a_grain_conflict_is_still_chosen_but_flagged():
    conflict = Conflict(asset_id="m", topic=EvidenceKind.GRAIN, evidence_ids=["e1"], values=["month", "day"])
    res = rule_select(S(C(grain="day"), [A("m", grain="month")], conflicts=[conflict]))
    assert tiers(res)[0] == ["m"] and "conflict:m:grain" in res.risk_flags and "grain_conflict" in res.primary[0].rule_ids       # 不让主资产为空；冲突显式保留


# ── ④ 的 LLM 相关性评估：软降权，不是丢弃依据 ──────────────────────────────────────────────────────
IRR = lambda a, subj="DAU": ev(a, EvidenceKind.OTHER, "llm_filter", "irrelevant", f"LLM relevance check ({subj}): irrelevant — this is the eDAU table, a different caliber", subject=subj)


def test_an_asset_judged_irrelevant_upstream_yields_to_a_relevant_one():
    res = rule_select(S(C(), [A("a", score=0.95), A("b", score=0.6)], evidences=[IRR("a")]))
    assert tiers(res) == (["b"], ["a"], []) and "upstream_irrelevant" in res.alternates[0].rule_ids
    assert "Phase 4" in res.alternates[0].reason.conclusion and "eDAU" in res.alternates[0].reason.conclusion


def test_irrelevance_for_another_metric_does_not_count():
    res = rule_select(S(C(), [A("a", score=0.95), A("b", score=0.6)], evidences=[IRR("a", subj="GMV")]))
    assert tiers(res)[0] == ["a"]


def test_when_everything_is_judged_irrelevant_the_best_is_still_chosen_with_a_downweight():
    res = rule_select(S(C(), [A("a", score=0.95), A("b", score=0.6)], evidences=[IRR("a"), IRR("b")]))
    assert tiers(res)[0] == ["a"] and any("Phase 4" in x for x in res.primary[0].reason.downweights)


def test_upstream_relevance_never_replaces_the_hard_guards():
    res = rule_select(S(C(), [A("a", fields=("dt", "gmv")), A("b")], evidences=[ev("a", EvidenceKind.OTHER, "llm_filter", "relevant", "相关", subject="DAU")]))
    assert tiers(res) == (["b"], [], ["a"])                                                # ④ 说相关也救不了「无目标指标字段」


# ── 新鲜度：④ 取不到最新分区 → 风险标记与盲区 ───────────────────────────────────────────────────────
def test_unknown_latest_partition_from_the_upstream_check_is_flagged_and_visible_as_a_blind_spot():
    unknown = ev("a", EvidenceKind.FRESHNESS, "partition_meta", None, "a 最新分区 未知", conf=0.2)
    res = rule_select(S(C(), [A("a")], evidences=[unknown]))
    assert "freshness_unknown:a" in res.risk_flags and any("Latest partition" in b for b in res.primary[0].reason.blind_spots)


def test_known_latest_partition_or_no_upstream_check_means_no_flag():
    known = ev("a", EvidenceKind.FRESHNESS, "partition_meta", "2026-11-12", "a 最新分区 2026-11-12")
    assert not any(f.startswith("freshness_unknown") for f in rule_select(S(C(), [A("a")], evidences=[known])).risk_flags)
    assert not any(f.startswith("freshness_unknown") for f in rule_select(S(C(), [A("a")])).risk_flags)          # ④ 没核对过（如 L0）：不臆测


# ── 佐证核验：与其他候选没有共同字段 → 盲区 ─────────────────────────────────────────────────────────
def test_a_cross_check_with_no_shared_fields_becomes_a_blind_spot():
    x = ev("a", EvidenceKind.OTHER, "cross_check", [], "与 b 的共同字段 无", conf=0.4)
    res = rule_select(S(C(), [A("a")], evidences=[x]))
    assert any("Shares no fields" in b for b in res.primary[0].reason.blind_spots)


def test_blind_spots_from_upstream_are_real_signals_and_listed_in_facts():
    unknown = ev("a", EvidenceKind.FRESHNESS, "partition_meta", None, conf=0.2)
    facts = build_facts(S(C(), [A("a")], evidences=[unknown]))
    assert any("Latest partition" in s for s in facts["a"].signals)


# ── 权限仍是 ⑤ 独有 ───────────────────────────────────────────────────────────────────────────────
def test_permission_is_still_decided_in_selection_only():
    res = rule_select(S(C(), [A("a", score=0.95), A("b", score=0.6)], perms={"a": (False, "no scope")}))
    assert tiers(res) == (["b"], ["a"], []) and res.alternates[0].needs_permission


# ── 进 LLM prompt ─────────────────────────────────────────────────────────────────────────────────
def test_the_selection_prompt_carries_the_upstream_evidence_per_candidate():
    inp = S(C(), [A("a"), A("b")], evidences=[IRR("a"), ev("b", EvidenceKind.FRESHNESS, "partition_meta", "2026-11-12", "b 最新分区 2026-11-12"),
                                              ev("b", EvidenceKind.CALIBER, "knowledge_hybrid", "DAU 按 user_id 去重", "DAU 按 user_id 去重")])
    evals = evaluate_group(inp, build_facts(inp), "DAU", {})
    p = build_prompt(inp, "DAU", evals)
    assert "upstream_evidence" in p and "irrelevant" in p and "eDAU" in p and "2026-11-12" in p and "user_id 去重" in p and "{{" not in p
    import json, re
    cands = json.loads(re.search(r"```json\n(\[.*?\n\])\n```", p, re.S).group(1))
    by = {c["id"]: c for c in cands}
    assert any(u["source"] == "llm_filter" for u in by["a"]["upstream_evidence"]) and any(u["kind"] == "freshness" for u in by["b"]["upstream_evidence"])


def test_upstream_evidence_is_scoped_to_the_group_metric_and_capped():
    many = [ev("a", EvidenceKind.OTHER, "cross_check", ["dt"], f"与 x{i} 的共同字段 dt") for i in range(20)] + [IRR("a", subj="GMV")]
    inp = S(C(), [A("a")], evidences=many)
    p = build_prompt(inp, "DAU", evaluate_group(inp, build_facts(inp), "DAU", {}))
    import json, re
    cands = json.loads(re.search(r"```json\n(\[.*?\n\])\n```", p, re.S).group(1))
    up = cands[0]["upstream_evidence"]
    assert len(up) <= 6 and not any(u["source"] == "llm_filter" for u in up)          # 另一个指标的评估不混进这个口径


def test_selection_prompt_template_explains_upstream_evidence():
    from app.prompts import load_prompt
    t = load_prompt("selection")
    assert "upstream_evidence" in t and "不要重复" in t and "llm_filter" in t
