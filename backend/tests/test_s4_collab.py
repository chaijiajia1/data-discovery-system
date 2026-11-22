"""④ 协作：黑板证据卡 / 能力画像注册表 / 路由打分 / 协作中枢（复用优先 · 定向 · 广播 · 婉拒降级）。"""
import asyncio
import json

import pytest

from app.collab.blackboard import Blackboard
from app.collab.hub import Resolution, build_hub
from app.collab.registry import JsonProfileStore, ProfileRegistry
from app.collab.router import decide, score_profile
from app.deps import Deps
from app.roles import REGISTRY, build_roles, list_role_specs
from app.schemas.collab import AskMode, EvidenceCard, Need, ReplyKind
from app.schemas.intent import Intent
from app.schemas.recall import AssetMeta, CandidateAsset
from app.schemas.task import Slot, TaskContract, ValueSource
from app.services.fakes import FakeBIRetrieval


def _need(kind="assets", subject="DAU", domains=("transaction",), asset_type=None):
    return Need(kind=kind, subject=subject, domains=list(domains), asset_type=asset_type or {"assets": "dataset", "lineage": "table", "caliber": "knowledge", "dimension": "table"}[kind])


def _card(key="assets:DAU", author="bi_asset_discovery", conf=0.8, assets=("dws.user_active_d",), domains=("transaction",), seq=0):
    kind, subject = key.split(":", 1)
    return EvidenceCard(id="c_" + key, key=key, kind=kind, subject=subject, author=author, asset_ids=list(assets), confidence=conf, domains=list(domains))


def _contract(domain="transaction", metrics=("DAU",)):
    return TaskContract(query="q", metrics=Slot(value=list(metrics), source=ValueSource.USER),
                        intent=Intent(query="q", domain=domain, sub_domain=[], dim_terms=[]))


def _registry(**kw):
    r = ProfileRegistry(**kw)
    r.ensure(list_role_specs())
    return r


ALL = ("bi_asset_discovery", "lineage_discovery", "caliber_lookup", "omni_discovery")


def make_hub(names=ALL, deps=None, contract=None, **settings):
    deps = deps or Deps()
    if settings:
        deps.settings = deps.settings.model_copy(update=settings)
    deps.profiles.ensure(list_role_specs())
    return build_hub(build_roles(list(names)), deps, contract=contract or _contract(), knowledge=[], cookie=None, recall_params={})


# ── 黑板：证据卡 / 认领 / 等待 ─────────────────────────────────────────────────
async def test_claim_is_exclusive_and_idempotent_for_the_same_owner():
    b = Blackboard()
    r = await asyncio.gather(b.try_claim("r1", "t"), b.try_claim("r2", "t"))
    assert sorted(r) == [False, True]
    owner = b.claimed_by("t")
    assert await b.try_claim(owner, "t") is True


async def test_publish_then_lookup_reuses_card_and_pools_assets():
    b = Blackboard()
    a = CandidateAsset(asset_id="dws.t", name="t", retrieval_score=0.7)
    card = await b.publish(_card(assets=("dws.t",)), assets=[a])
    assert card.seq == 1 and b.lookup(_need()).id == card.id
    assert [x.asset_id for x in b.assets_of(card)] == ["dws.t"]
    assert b.pool.get("dws.t") is not None


async def test_lookup_respects_confidence_scope_and_emptiness():
    b = Blackboard()
    await b.publish(_card(conf=0.2))
    assert b.lookup(_need(), min_conf=0.3) is None                                   # 置信度不够，不复用
    await b.publish(_card(key="assets:GMV", domains=("community",)))
    assert b.lookup(_need(subject="GMV", domains=("transaction",))) is None            # 业务域不相容
    assert b.lookup(_need(subject="GMV", domains=("community",))) is not None
    await b.publish(_card(key="lineage:DAU", assets=()))
    assert b.lookup(_need("lineage"), require_content=True) is None                    # 空卡：不能当种子
    assert b.lookup(_need("lineage"), require_content=False) is not None               # 但可用来避免重复探查


async def test_wait_card_resolves_when_published_and_times_out_otherwise():
    b = Blackboard()
    waiter = asyncio.create_task(b.wait_card("assets:DAU", timeout=2))
    await asyncio.sleep(0.02)
    await b.publish(_card())
    assert (await waiter).author == "bi_asset_discovery"
    assert await b.wait_card("assets:NONE", timeout=0.05) is None


async def test_release_unblocks_waiters_without_a_card():
    b = Blackboard()
    await b.try_claim("r1", "assets:DAU")
    waiter = asyncio.create_task(b.wait_card("assets:DAU", timeout=2))
    await asyncio.sleep(0.02)
    b.release("assets:DAU")
    assert await waiter is None and b.claimed_by("assets:DAU") is None


# ── 能力画像注册表 ───────────────────────────────────────────────────────────
def test_registry_seeds_profiles_from_role_specs():
    r = _registry()
    p = r.get("lineage_discovery")
    assert p.serves == ["lineage"] and "table" in p.asset_types and "血缘" in p.keywords and p.quality == 0.7 and len(p.domains) == 5
    assert r.get("dim_value_resolver").domains == ["commercialization", "community", "transaction"]
    assert r.get("omni_discovery").serves == ["assets", "caliber", "lineage"] and r.get("freshness_precheck") is None


def test_registry_reseed_keeps_learned_state_but_refreshes_static_fields():
    r = _registry()
    r.record("bi_asset_discovery", "adopted")
    q = r.get("bi_asset_discovery").quality
    r.ensure(list_role_specs())
    assert r.get("bi_asset_discovery").quality == q and r.get("bi_asset_discovery").stats.adopted == 1


def test_registry_search_ranks_by_need_and_is_limited_to_present_roles():
    r = _registry()
    top = lambda need, roles=None: [s.role for s in r.search(need, roles=roles)][0]
    assert top(_need("assets")) == "bi_asset_discovery" and top(_need("lineage")) == "lineage_discovery" and top(_need("caliber")) == "caliber_lookup"
    assert top(_need("dimension", asset_type="table")) == "dim_value_resolver"
    assert [s.role for s in r.search(_need("assets"), roles=["omni_discovery", "lineage_discovery"])][0] == "omni_discovery"
    assert all(s.role in {"omni_discovery", "lineage_discovery"} for s in r.search(_need("assets"), roles=["omni_discovery", "lineage_discovery"]))


@pytest.mark.parametrize("outcome,expected_q", [("adopted", 0.76), ("ignored", 0.56), ("declined", 0.63)])
def test_registry_record_updates_quality_ema_and_stats(outcome, expected_q):
    r = _registry()
    u = r.record("bi_asset_discovery", outcome)
    p = r.get("bi_asset_discovery")
    assert (u.quality_before, round(u.quality_after, 4)) == (0.7, expected_q) and round(p.quality, 4) == expected_q
    assert getattr(p.stats, outcome) == 1 and u.outcome == outcome


def test_registry_reuse_adoption_counts_with_lower_weight():
    r = _registry()
    r.record("bi_asset_discovery", "adopted", weight=0.5)
    assert round(r.get("bi_asset_discovery").quality, 4) == 0.73                       # α·weight = 0.1


def test_registry_persists_to_json_and_survives_corruption(tmp_path):
    path = tmp_path / "profiles.json"
    r = _registry(store=JsonProfileStore(path))
    r.record("caliber_lookup", "ignored")
    r.flush()
    again = ProfileRegistry(store=JsonProfileStore(path))
    again.ensure(list_role_specs())
    assert round(again.get("caliber_lookup").quality, 4) == 0.56 and again.get("caliber_lookup").stats.ignored == 1
    path.write_text("{not json")
    broken = ProfileRegistry(store=JsonProfileStore(path))
    broken.ensure(list_role_specs())
    assert broken.get("caliber_lookup").quality == 0.7                                  # 画像文件损坏 → 回退种子，不影响主流程


# ── 路由打分：定向 or 广播 ───────────────────────────────────────────────────
def _profiles(reg, names=ALL):
    return [reg.get(n) for n in names]


def test_score_profile_parts_and_quality_factor():
    reg = _registry()
    s = score_profile(_need("assets"), reg.get("bi_asset_discovery"))
    assert s.parts["kind"] == 1 and s.parts["asset_type"] == 1 and s.parts["domain"] == 1 and s.parts["focus"] == 1
    assert s.parts["quality_factor"] == pytest.approx(0.5 + 0.5 * 0.7) and 0.6 < s.score < 1
    assert score_profile(_need("assets"), reg.get("lineage_discovery")).parts["kind"] == 0
    assert score_profile(_need("assets"), reg.get("omni_discovery")).score < s.score                  # 专才 > 通才


@pytest.mark.parametrize("kind,target", [("assets", "bi_asset_discovery"), ("lineage", "lineage_discovery"), ("caliber", "caliber_lookup")])
def test_high_confidence_hit_goes_directed(kind, target):
    d = decide(_need(kind), _profiles(_registry()), direct_threshold=0.6)
    assert d.mode == AskMode.DIRECTED and d.target == target and d.confidence >= 0.6 and d.scores[0].role == target


def test_low_confidence_falls_back_to_broadcast():
    d = decide(_need("assets"), _profiles(_registry()), direct_threshold=0.99)
    assert d.mode == AskMode.BROADCAST and d.target is None and "low_confidence" in d.reason


def test_cross_domain_request_is_broadcast_even_if_a_profile_matches():
    d = decide(_need("assets", domains=("transaction", "community")), _profiles(_registry()))
    assert d.mode == AskMode.BROADCAST and "cross_domain" in d.reason


def test_request_outside_the_profiles_domain_coverage_is_broadcast():
    d = decide(_need("dimension", domains=("resource_cost",), asset_type="table"), _profiles(_registry(), ("dim_value_resolver", "bi_asset_discovery")))
    assert d.mode == AskMode.BROADCAST and "domain_uncovered" in d.reason


def test_request_no_present_profile_serves_is_broadcast():
    d = decide(_need("lineage"), _profiles(_registry(), ("bi_asset_discovery", "caliber_lookup")))
    assert d.mode == AskMode.BROADCAST and "uncovered" in d.reason


def test_poor_history_stops_direct_routing_and_shifts_to_the_next_best_role():
    reg = _registry()
    for _ in range(12):
        reg.record("bi_asset_discovery", "ignored")
    assert reg.get("bi_asset_discovery").quality < 0.1
    alone = decide(_need("assets"), _profiles(reg, ("bi_asset_discovery", "lineage_discovery", "caliber_lookup")))
    assert alone.mode == AskMode.BROADCAST and "low_confidence" in alone.reason                  # 没有备选：降级为广播
    with_omni = decide(_need("assets"), _profiles(reg))
    assert with_omni.mode == AskMode.DIRECTED and with_omni.target == "omni_discovery"          # 有备选：改点备选


# ── 协作中枢：复用优先 / 认领等待 / 求助 ─────────────────────────────────────────
async def test_reuse_first_skips_produce_and_ask():
    hub = make_hub()
    await hub.board.publish(_card())
    calls = []

    async def produce():
        calls.append(1)

    res = await hub.obtain(_need(), requester="caliber_lookup", who="n1", produce=produce)
    assert res.how == "reuse" and res.card.id == "c_assets:DAU" and calls == []
    assert hub.trace.asks == [] and hub.trace.reuse[0].by == "caliber_lookup" and hub.trace.reuse[0].author == "bi_asset_discovery"
    assert hub.board.cards["assets:DAU"].reuse_count == 1
    assert hub.registry.get("bi_asset_discovery").stats.adopted == 1                       # 证据卡被复用 = 被采纳


async def test_reuse_first_also_applies_before_asking():
    hub = make_hub()
    await hub.board.publish(_card())
    res = await hub.obtain(_need(), requester="lineage_discovery", who="n1", produce=None)
    assert res.how == "reuse" and hub.trace.asks == []                                    # 有卡直接用：不询问、不广播


async def test_concurrent_identical_needs_are_probed_once():
    hub = make_hub()
    calls = []

    async def produce():
        calls.append(1)
        await asyncio.sleep(0.05)
        return await hub.board.publish(_card(author="bi_asset_discovery"))

    r = await asyncio.gather(hub.obtain(_need(), "bi_asset_discovery", "n1", produce), hub.obtain(_need(), "omni_discovery", "n2", produce))
    assert len(calls) == 1 and sorted(x.how for x in r) == ["own", "reuse_wait"]
    assert hub.trace.retrievals == ["assets:DAU"] and hub.trace.reuse[0].waited is True


async def test_wait_timeout_falls_back_to_own_production():
    hub = make_hub(COLLAB_WAIT_SEC=0.05)
    await hub.board.try_claim("ghost", "assets:DAU")                                     # 有人认领却一直不产出
    res = await hub.obtain(_need(), "omni_discovery", "n2", lambda: hub.board.publish(_card(author="omni_discovery")))
    assert res.how == "own" and res.card.author == "omni_discovery"


async def test_failed_production_releases_the_claim_for_waiters():
    hub = make_hub(COLLAB_WAIT_SEC=2)

    async def boom():
        await asyncio.sleep(0.03)
        raise RuntimeError("upstream down")

    async def ok():
        return await hub.board.publish(_card(author="omni_discovery"))

    t1 = asyncio.create_task(hub.obtain(_need(), "bi_asset_discovery", "n1", boom))
    await asyncio.sleep(0.005)
    t2 = asyncio.create_task(hub.obtain(_need(), "omni_discovery", "n2", ok))
    with pytest.raises(RuntimeError):
        await t1
    assert (await t2).how == "own"                                                         # 不被拖死：重新自己探查


async def test_ask_goes_directed_to_the_matching_role_and_is_adopted():
    hub = make_hub()
    res = await hub.obtain(_need(), requester="lineage_discovery", who="n1", produce=None)
    rec = hub.trace.asks[0]
    assert res.how == "ask" and res.card.author == "bi_asset_discovery" and res.card.asset_ids
    assert rec.route.mode == AskMode.DIRECTED and rec.route.target == "bi_asset_discovery" and rec.served_by == "bi_asset_discovery"
    assert [r.kind for r in rec.replies] == [ReplyKind.ANSWER] and rec.outcome == "adopted" and rec.bids == []         # 定向：没有全量广播 / 自评估
    assert hub.registry.get("bi_asset_discovery").stats.adopted == 1 and hub.registry.get("bi_asset_discovery").quality > 0.7
    assert [u.outcome for u in hub.trace.profile_updates] == ["adopted"]
    assert hub.trace.retrievals == ["assets:DAU"]


async def test_requester_never_asks_itself():
    hub = make_hub(("bi_asset_discovery", "caliber_lookup"))
    await hub.obtain(_need(), requester="bi_asset_discovery", who="n1", produce=None)
    assert all(r.from_role != "bi_asset_discovery" for rec in hub.trace.asks for r in rec.replies)
    assert hub.trace.asks[0].route.mode == AskMode.BROADCAST or hub.trace.asks[0].route.target != "bi_asset_discovery"


async def test_directed_decline_with_reason_downgrades_to_broadcast_and_volunteer_answers():
    hub = make_hub()
    hub.roles["bi_asset_discovery"]._active = hub.roles["bi_asset_discovery"].spec.max_concurrency       # 正忙
    res = await hub.obtain(_need(), requester="lineage_discovery", who="n1", produce=None)
    rec = hub.trace.asks[0]
    kinds = [r.kind for r in rec.replies]
    assert kinds[:2] == [ReplyKind.DECLINE, ReplyKind.DOWNGRADE] and ReplyKind.VOLUNTEER in kinds
    decline = rec.replies[0]
    assert decline.from_role == "bi_asset_discovery" and decline.reason.startswith("busy")             # 婉拒必须说明理由
    assert rec.route.mode == AskMode.DIRECTED and rec.served_by == "omni_discovery" and res.card.author == "omni_discovery"
    assert rec.outcome == "adopted" and any(b.role == "omni_discovery" and b.volunteered for b in rec.bids)
    assert hub.registry.get("bi_asset_discovery").stats.declined == 1 and hub.registry.get("omni_discovery").stats.adopted == 1
    assert hub.trace.summarize()["downgraded"] == 1 and hub.trace.stats["declined"] == 1


async def test_declining_role_is_not_asked_again_in_the_broadcast():
    hub = make_hub()
    hub.roles["bi_asset_discovery"]._active = 99
    await hub.obtain(_need(), requester="lineage_discovery", who="n1", produce=None)
    assert "bi_asset_discovery" not in [b.role for b in hub.trace.asks[0].bids]


async def test_broadcast_best_fit_volunteer_serves_the_cross_domain_request():
    hub = make_hub()
    need = _need("assets", domains=("transaction", "community"))
    res = await hub.obtain(need, requester="caliber_lookup", who="n1", produce=None)
    rec = hub.trace.asks[0]
    assert rec.route.mode == AskMode.BROADCAST and "cross_domain" in rec.route.reason
    bids = {b.role: b for b in rec.bids}
    assert bids["bi_asset_discovery"].volunteered and not bids["lineage_discovery"].volunteered and bids["lineage_discovery"].reason.startswith("out_of_scope")
    assert rec.served_by == "bi_asset_discovery" and res.card.author == "bi_asset_discovery"                # 揭榜者里匹配度最高的
    assert [r.kind for r in rec.replies] == [ReplyKind.VOLUNTEER]
    assert any(e.kind.value == "broadcast" for e in hub.board.entries())                                    # 广播留痕在黑板上


async def test_nobody_volunteers_leaves_the_ask_unanswered():
    hub = make_hub(("caliber_lookup", "dim_value_resolver"))
    res = await hub.obtain(_need("lineage"), requester="caliber_lookup", who="n1", produce=None)
    rec = hub.trace.asks[0]
    assert res.card is None and res.how == "none" and rec.outcome == "unanswered" and rec.served_by is None and rec.route.mode == AskMode.BROADCAST


async def test_empty_answer_is_ignored_and_hurts_the_responders_profile():
    class NoBI(FakeBIRetrieval):
        async def search(self, query, **kw):
            return []

    hub = make_hub(deps=Deps(bi=NoBI()))
    res = await hub.obtain(_need(), requester="lineage_discovery", who="n1", produce=None)
    assert res.card is None and hub.trace.asks[0].outcome == "ignored"
    p = hub.registry.get("bi_asset_discovery")
    assert p.stats.ignored == 1 and p.quality < 0.7


async def test_responder_autonomy_out_of_scope_domain_uncovered_busy_and_low_fit():
    hub = make_hub(("dim_value_resolver", "bi_asset_discovery"))
    dim, bi = hub.roles["dim_value_resolver"], hub.roles["bi_asset_discovery"]
    from app.schemas.collab import Ask
    mk = lambda need: Ask(id="a1", from_role="x", to_role="y", mode=AskMode.DIRECTED, question=need.text, need=need)
    r = await bi.respond(mk(_need("lineage")), hub)
    assert r.kind == ReplyKind.DECLINE and r.reason.startswith("out_of_scope")
    r = await dim.respond(mk(_need("dimension", domains=("resource_cost",), asset_type="table")), hub)
    assert r.kind == ReplyKind.DECLINE and r.reason.startswith("domain_uncovered") and "resource_cost" in r.reason
    bi._active = bi.spec.max_concurrency
    assert (await bi.respond(mk(_need()), hub)).reason.startswith("busy")
    bi._active = 0
    hub.settings = hub.settings.model_copy(update={"COLLAB_ACCEPT_MIN": 0.95})
    assert (await bi.respond(mk(_need()), hub)).reason.startswith("low_fit")


async def test_a_role_declines_needs_outside_what_it_serves():
    hub = make_hub(("dim_value_resolver", "bi_asset_discovery"))
    from app.schemas.collab import Ask
    r = await hub.roles["dim_value_resolver"].respond(Ask(id="a", from_role="x", mode=AskMode.DIRECTED, question="q", need=_need()), hub)
    assert r.kind == ReplyKind.DECLINE and r.reason.startswith("out_of_scope")
