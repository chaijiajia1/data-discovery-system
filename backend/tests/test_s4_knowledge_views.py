"""④ 各角色按自己的知识视图取知识（而不是共用 env.knowledge）。"""
import json

from app.collab.hub import build_hub
from app.deps import Deps
from app.roles import build_roles, list_role_specs
from app.roles.llm_agent import knowledge_json
from app.roles.retrieve import fetch_caliber, lineage_seeds
from app.schemas.intent import Intent
from app.schemas.knowledge import KnowledgeItem, KnowledgeKind, KnowledgeOrigin, recall_consumer
from app.schemas.task import Slot, TaskContract, ValueSource

K = KnowledgeKind


def _k(i, kind=K.METRIC_DEF, assets=(), title="DAU", rel=0.9):
    return KnowledgeItem(id=i, kind=kind, title=title, content=f"{title} 的知识 {i}", origin=KnowledgeOrigin.RETRIEVAL, relevance=rel,
                         token_cost=10, asset_ids=list(assets))


def _contract():
    return TaskContract(query="q", metrics=Slot(value=["DAU"], source=ValueSource.USER), intent=Intent(query="q", domain="transaction", sub_domain=[], dim_terms=[]))


def _hub(knowledge, views=None, names=("lineage_discovery", "caliber_lookup", "omni_discovery")):
    deps = Deps()
    deps.profiles.ensure(list_role_specs())
    return build_hub(build_roles(list(names)), deps, contract=_contract(), knowledge=knowledge, views=views, cookie=None, recall_params={})


def test_knowledge_for_falls_back_to_full_list_without_views():
    k = [_k("a"), _k("b")]
    hub = _hub(k)
    assert hub.knowledge_for("lineage_discovery") == k


def test_knowledge_for_uses_role_view_and_empty_view_is_empty():
    a, b = _k("a"), _k("b")
    hub = _hub([a, b], views={recall_consumer("lineage_discovery"): [b], recall_consumer("caliber_lookup"): []})
    assert hub.knowledge_for("lineage_discovery") == [b]
    assert hub.knowledge_for("caliber_lookup") == []               # 视图为空 ≠ 回落全量
    assert hub.knowledge_for("omni_discovery") == [a, b]           # 没登记视图的角色回落全量


def test_knowledge_for_resolves_agent_id_to_role_name():
    from app.roles import REGISTRY
    deps = Deps()
    deps.profiles.ensure(list_role_specs())
    agents = {"n1": REGISTRY["lineage_discovery"](agent_id="n1")}
    b = _k("b")
    hub = build_hub(agents, deps, contract=_contract(), knowledge=[_k("a"), b], views={recall_consumer("lineage_discovery"): [b]}, cookie=None, recall_params={})
    assert hub.knowledge_for("n1") == [b]                           # 同一角色的多个副本看到同一份视图


def test_lineage_seeds_come_from_the_lineage_view_only():
    a, b = _k("a", K.ASSET_META, ["t_a"]), _k("b", K.ASSET_META, ["t_b"])
    hub = _hub([a, b], views={recall_consumer("lineage_discovery"): [b]})
    assert lineage_seeds(hub, "DAU", None, role="lineage_discovery") == ["t_b"]
    assert lineage_seeds(hub, "DAU", None, role="omni_discovery") == ["t_a", "t_b"]     # omni 没有视图 → 全量


async def test_fetch_caliber_reuses_only_the_caliber_view():
    class KB:
        async def search(self, query, **kw):
            return []
    a, b = _k("a", K.METRIC_DEF, ["t_a"]), _k("b", K.METRIC_DEF, ["t_b"])
    hub = _hub([a, b], views={recall_consumer("caliber_lookup"): [a]})
    hub.env.deps.caliber_kb = KB()
    card = await fetch_caliber(hub, "caliber_lookup", "DAU", "caliber_lookup")
    assert card.asset_ids == ["t_a"]


def test_llm_prompt_knowledge_is_filtered_per_role_and_truncated():
    a, b = _k("a", K.METRIC_DEF, ["t_a"]), _k("b", K.METRIC_DEF, ["t_b"])
    hub = _hub([a, b], views={recall_consumer("lineage_discovery"): [b]})
    ids = [x["id"] for x in json.loads(knowledge_json(hub, ["DAU"], role="lineage_discovery"))]
    assert ids == ["b"]
    assert knowledge_json(_hub([a, b], views={recall_consumer("caliber_lookup"): []}), ["DAU"], role="caliber_lookup") == "（无）"


async def test_recall_stage_passes_distinct_views_to_hub(make_ctx):
    from app.schemas.knowledge import KnowledgeView, PreInjectionResult
    from app.stages.s4_recall import views_for_hub
    a, b = _k("a"), _k("b", K.DIMENSION, title="渠道")
    pre = PreInjectionResult(injected=[a, b], views={recall_consumer("dim_value_resolver"): KnowledgeView(consumer=recall_consumer("dim_value_resolver"), item_ids=["b"]),
                                                     recall_consumer("caliber_lookup"): KnowledgeView(consumer=recall_consumer("caliber_lookup"), item_ids=["a"])})
    views = views_for_hub(pre)
    assert views == {recall_consumer("dim_value_resolver"): [b], recall_consumer("caliber_lookup"): [a]}
    assert views_for_hub(None) == {} and views_for_hub(PreInjectionResult(injected=[a])) == {}


# ── ② ③ ⑤ ⑥ 消费视图 ───────────────────────────────────────────────────────
def _pre(views):
    from app.schemas.knowledge import KnowledgeView, PreInjectionResult
    inj = [_k("k_keep", K.METRIC_DEF, ["t1"], title="KEEP_METRIC"), _k("k_other", K.ASSET_META, ["t2"], title="OTHER_ASSET")]
    return PreInjectionResult(injected=inj, views={c: KnowledgeView(consumer=c, item_ids=ids) for c, ids in views.items()})


def test_coordination_prompt_uses_coordination_view():
    from app.schemas.knowledge import CONSUMER_COORDINATION
    from app.stages.s2_coordination.agent import build_prompt
    p = build_prompt("q", _pre({CONSUMER_COORDINATION: ["k_keep"]}))
    assert "KEEP_METRIC" in p and "OTHER_ASSET" not in p
    assert "OTHER_ASSET" in build_prompt("q", _pre({}))                    # 没有视图：回落全量


def test_planner_prompt_uses_planner_view_but_keeps_conflicts():
    from app.schemas.knowledge import CONSUMER_PLANNER, DroppedKnowledge
    from app.stages.s3_planning.level_agent import build_prompt
    pre = _pre({CONSUMER_PLANNER: ["k_other"]})
    pre.dropped.append(DroppedKnowledge(item_id="x", title="CONFLICT_TITLE", reason="conflict", detail="d"))
    p = build_prompt(_contract(), pre, confirmed={})
    assert "OTHER_ASSET" in p and "KEEP_METRIC" not in p
    assert "CONFLICT_TITLE" in p                                           # 冲突信息不随视图裁剪


def test_selection_input_uses_selection_view():
    from app.schemas.knowledge import CONSUMER_SELECTION
    from app.stages.base import RunContext
    from app.schemas.recall import RecallResult
    from app.stages.s5_selection.stage import build_input
    ctx = RunContext(query="q", deps=Deps())
    ctx.contract = _contract()
    ctx.recall = RecallResult()
    ctx.preinjection = _pre({CONSUMER_SELECTION: ["k_keep"]})
    assert [k.id for k in build_input(ctx).knowledge] == ["k_keep"]


def test_selection_input_still_excludes_pair_blocks():
    from app.schemas.knowledge import KnowledgeView, PreInjectionResult
    from app.stages.base import RunContext
    from app.schemas.recall import RecallResult
    from app.stages.s5_selection.stage import build_input
    ctx = RunContext(query="q", deps=Deps())
    ctx.contract, ctx.recall = _contract(), RecallResult()
    ctx.preinjection = PreInjectionResult(injected=[_k("m"), _k("p", K.PAIR)])   # 没有视图 → 回落全量，仍要剔除 PAIR
    assert [k.id for k in build_input(ctx).knowledge] == ["m"]


def test_write_back_counts_only_pairs_seen_by_some_consumer():
    from app.schemas.knowledge import KnowledgeView, PreInjectionResult
    from app.stages.s6_execution.feedback import write_back
    from app.stores import InMemoryPairStore
    seen, unseen = _k("p1", K.PAIR), _k("p2", K.PAIR)
    seen.source_pair_id, unseen.source_pair_id = "pair1", "pair2"
    pre = PreInjectionResult(injected=[seen, unseen], views={"planner": KnowledgeView(consumer="planner", item_ids=["p1"])})
    calls = []
    store = InMemoryPairStore()
    store.record_outcome = lambda ids, ok: calls.append((list(ids), ok))
    write_back(store, None, pre, True)
    assert calls == [(["pair1"], True)]
