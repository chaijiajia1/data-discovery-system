"""内置角色模板（召回阶段 + 选择阶段）。

覆盖单元 = 口径 × 通道（bi / graph / knowledge）。每个召回节点查询一个或多个口径（targets），经一个或多个通道（角色的 channels）。
召回阶段：
  bi_asset_discovery   基础通道：已认证看板 / 数据集 / 图表（bi）
  lineage_discovery    补充通道：沿知识图谱血缘扩展，覆盖问法与命名脱节的漏召回（graph）
  caliber_lookup       口径证据：指标定义 / 业务术语 / 使用约束（knowledge）
  omni_discovery       全通道发现：一个 Agent 查 bi + graph + knowledge（Agent 少；各指标并发，同一指标内血缘 / 口径在 BI 之后并发）
  dim_value_resolver   维值解析：把"自然堂 / 美妆"这类具名维度取值落到字段（dim_value，不在覆盖网格内）
评估维度（原选择阶段的核对）由召回角色内联提供（见 checks.py）：bi = grain_fit / freshness，lineage = dim_coverage，caliber = caliber，omni = 全部。
"""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable

from ..collab.hub import CollabHub, Resolution
from ..schemas.collab import EvidenceCard, Need
from ..schemas.recall import CandidateAsset, Evidence, EvidenceKind, RoleReport
from .base import Role, RoleContext, RoleSpec, new_evidence_id
from .checks import gather_candidates, inline_checks
from .llm_agent import annotate_results, plan_queries
from .retrieve import fetch_assets, fetch_caliber, fetch_dimension, fetch_lineage, make_need

# 检索通道
CH_BI, CH_GRAPH, CH_KNOWLEDGE, CH_DIM = "bi", "graph", "knowledge", "dim_value"
GRID_CHANNELS = (CH_BI, CH_GRAPH, CH_KNOWLEDGE)       # 覆盖网格的通道（口径 × 通道）


def _metrics_of(rc: RoleContext) -> list[str]:
    return list(rc.node.targets) or [str(m) for m in (rc.contract.metrics.value or [])] or [rc.contract.query]


async def _all(coros) -> list[Any]:
    """并发执行并等全部结束；任何一个失败则在全部结束后抛出第一个异常（不留下孤儿任务，认领已在 hub 里释放）。"""
    res = await asyncio.gather(*coros, return_exceptions=True)
    for r in res:
        if isinstance(r, BaseException):
            raise r
    return list(res)


async def discover(role: Role, rc: RoleContext, kinds: tuple[str, ...]) -> RoleReport:
    """召回角色的通用流程：分到的每个「指标 × 通道」并发探查。

    LLM 可用时多两步（各一次调用，整个节点最多两次）：探查前改写 query（并拿到换角度的备选说法）；探查后评估结果，
    结果为空或被判定全部不相关时，换个角度再查（走转交机制）。LLM 不可用时这两步为空操作，行为与纯规则一致。
    """
    hub, me = rc.hub, role.agent_id
    metrics = _metrics_of(rc)
    if rc.mode == "select_only":                       # L1：已有知识指明了资产、只是口径存疑 → 不再召回；只对口径去读知识全文，其余核对已知资产
        kinds = tuple(k for k in kinds if k == "caliber")
    plans = await plan_queries(hub, role.spec, metrics, kinds) if kinds else {}

    def need_for(kind: str, m: str):
        n = make_need(hub, kind, m)
        plan = plans.get(m)
        if plan is not None:
            if plan.query:
                n = n.model_copy(update={"query": plan.query})
            if plan.limit is not None:                        # 召回多少条由 LLM 决定
                n = n.model_copy(update={"limit": plan.limit})
            if plan.angles:
                hub.angles.setdefault(n.key, list(plan.angles))
        hub.needs.setdefault(n.key, n)
        if n.limit:
            hub.limits.setdefault(n.key, n.limit)
        return n

    async def one(kind: str, m: str) -> Resolution:
        n = need_for(kind, m)
        return await hub.obtain(n, me, me, lambda: role.produce(n, hub, me))

    async def per_metric(m: str) -> list[Resolution]:
        out: list[Resolution] = []
        rest = list(kinds)
        if "assets" in kinds and len(kinds) > 1:           # 血缘以 BI 的证据卡为种子，口径无证据时反向找承载资产：先 BI，其余通道随后并发
            out.append(await one("assets", m))
            rest.remove("assets")
        return out + await _all(one(k, m) for k in rest)

    resolutions = [r for group in await _all(per_metric(m) for m in metrics) for r in group] if kinds else []
    own = [r.card for r in resolutions if r.how == "own" and r.card is not None and not r.card.empty]
    res = await annotate_results(hub, me, role.spec, own)
    for card in res.all_irrelevant:
        await hub.retry_other_angle(make_need(hub, card.kind, card.subject), me)
    for _ in range(max(0, hub.settings.RECALL_MAX_SUPPLEMENTS)):     # 评估后觉得不够：按更大的条数继续召回，直到够了 / 没有新增 / 次数用完
        if not res.more or not hub.llm.has_budget:
            break
        updated = [c for c in await _all(
            hub.supplement(key, me, limit, q, lambda n, role=role: role.produce(n, hub, me)) for key, (limit, q) in res.more.items()) if c is not None]
        if not updated:
            break
        res = await annotate_results(hub, me, role.spec, updated)
        for card in res.all_irrelevant:
            await hub.retry_other_angle(make_need(hub, card.kind, card.subject), me)
    rep = _report_from(role, rc, resolutions)
    rep.evidences += await inline_checks(role, rc, metrics)          # 边召回边核对：新鲜度 / 维度覆盖 / 佐证（按角色提供的评估维度）
    return rep


def hub_role(rc: RoleContext, agent_id: str) -> str:
    return rc.hub.role_of(agent_id)


def _report_from(role: Role, rc: RoleContext, resolutions: list[Resolution]) -> RoleReport:
    """把一批需求的解决结果汇总成角色报告：证据卡 → 候选 / 证据；复用别人证据卡得到的候选，found_by 记为原作者。"""
    board = rc.hub.board
    cands: dict[str, CandidateAsset] = {}
    evs: dict[str, Evidence] = {}
    asks, replies = [], []
    for res in resolutions:
        if res.card is not None:
            for a in board.assets_of(res.card):
                cands.setdefault(a.asset_id, a.model_copy(update={"found_by": [hub_role(rc, res.card.author)]}, deep=True))
            for e in board.evidences_of(res.card):
                evs.setdefault(e.id, e)
        if res.record is not None and res.record.ask is not None:
            asks.append(res.record.ask)
            replies += res.record.replies
    return role._report(rc, candidates=list(cands.values()), evidences=list(evs.values()), asks=asks, replies=replies)


class BIAssetDiscovery(Role):
    spec = RoleSpec(
        "bi_asset_discovery", "BI 资产发现：检索已认证的看板、数据集和指标层，产出可直接使用的候选资产",
        phase="recall", channels=frozenset({CH_BI}), dims=frozenset({"metric_match", "grain_fit", "freshness"}), max_concurrency=3,
        serves=frozenset({"assets"}), asset_types=frozenset({"dataset", "chart", "dashboard"}),
        keywords=("资产", "看板", "数据集", "图表", "指标层", "认证", "承载"),
    )

    async def produce(self, need: Need, hub: CollabHub, who: str) -> EvidenceCard | None:
        return await fetch_assets(hub, self.agent_id, need.subject, need.hint_assets, need.query or None, need.limit or None) if need.kind == "assets" else None

    async def run(self, rc: RoleContext) -> RoleReport:
        return await discover(self, rc, ("assets",))


class LineageDiscovery(Role):
    spec = RoleSpec(
        "lineage_discovery", "表与字段血缘发现：沿知识图谱的血缘关系扩展，覆盖问法与命名脱节的漏召回",
        phase="recall", channels=frozenset({CH_GRAPH}), dims=frozenset({"metric_match", "dim_coverage"}), max_concurrency=2,
        serves=frozenset({"lineage"}), asset_types=frozenset({"table", "dataset"}),
        keywords=("血缘", "关联", "明细表", "字段", "上游", "下游", "图谱", "命名"),
    )

    async def produce(self, need: Need, hub: CollabHub, who: str) -> EvidenceCard | None:
        return await fetch_lineage(hub, self.agent_id, need.subject, who, need.hint_assets, need.query or None, need.limit or None) if need.kind == "lineage" else None

    async def run(self, rc: RoleContext) -> RoleReport:
        return await discover(self, rc, ("lineage",))


class CaliberLookup(Role):
    spec = RoleSpec(
        "caliber_lookup", "口径查找：检索指标定义、业务术语与使用约束，为选择阶段提供口径证据",
        phase="recall", channels=frozenset({CH_KNOWLEDGE}), dims=frozenset({"caliber"}), max_concurrency=3,
        serves=frozenset({"caliber"}), asset_types=frozenset({"knowledge"}),
        keywords=("口径", "定义", "约束", "术语", "指标定义", "文档"),
    )

    async def produce(self, need: Need, hub: CollabHub, who: str) -> EvidenceCard | None:
        return await fetch_caliber(hub, self.agent_id, need.subject, who, need.hint_assets, need.query or None, need.limit or None) if need.kind == "caliber" else None

    async def run(self, rc: RoleContext) -> RoleReport:
        return await discover(self, rc, ("caliber",))


class OmniDiscovery(Role):
    """一个 Agent 依次查 bi → graph → knowledge。Agent 数最少，但节点内串行（工作量 = 口径数 × 3）。"""

    spec = RoleSpec(
        "omni_discovery", "全通道发现：一个 Agent 依次查 BI、血缘、口径三个通道",
        phase="recall", channels=frozenset(GRID_CHANNELS), dims=frozenset({"metric_match", "caliber", "grain_fit", "freshness", "dim_coverage"}), max_concurrency=2,
        serves=frozenset({"assets", "lineage", "caliber"}), asset_types=frozenset({"dataset", "chart", "dashboard", "table", "knowledge"}),
        keywords=("资产", "血缘", "口径", "全通道"),
    )

    async def produce(self, need: Need, hub: CollabHub, who: str) -> EvidenceCard | None:
        n = self.agent_id
        if need.kind == "assets":
            return await fetch_assets(hub, n, need.subject, need.hint_assets, need.query or None, need.limit or None)
        if need.kind == "lineage":
            return await fetch_lineage(hub, n, need.subject, who, need.hint_assets, need.query or None, need.limit or None)
        if need.kind == "caliber":
            return await fetch_caliber(hub, n, need.subject, who, need.hint_assets, need.query or None, need.limit or None)
        return None

    async def run(self, rc: RoleContext) -> RoleReport:
        return await discover(self, rc, ("assets", "lineage", "caliber"))


class DimValueResolver(Role):
    spec = RoleSpec(
        "dim_value_resolver", "维值解析：把具名维度取值（品牌 / 商家 / 类目）解析到字段，让后续召回更精准",
        phase="recall", channels=frozenset({CH_DIM}), dims=frozenset({"dim_coverage"}), max_concurrency=2,
        serves=frozenset({"dimension"}), asset_types=frozenset({"dataset", "table"}), domains=frozenset({"transaction", "community", "commercialization"}),
        keywords=("维度", "维值", "取值", "字段", "品牌", "商家", "类目"),
    )

    async def produce(self, need: Need, hub: CollabHub, who: str) -> EvidenceCard | None:
        return await fetch_dimension(hub, self.agent_id, need.subject) if need.kind == "dimension" else None

    async def run(self, rc: RoleContext) -> RoleReport:
        hub, n = rc.hub, self.agent_id
        terms = list(rc.contract.intent.dim_terms) if rc.contract.intent else []
        out = await _all(hub.obtain(make_need(hub, "dimension", t), n, rc.node.id, lambda t=t: fetch_dimension(hub, n, t)) for t in terms)
        return _report_from(self, rc, out)
