"""④ 角色的取数：把 BI 通道与知识通道（向量 + 图谱）的结果归一化进统一候选池，并发布证据卡。

每个 fetch_* 对应一类需求（Need.kind），产出一张证据卡（EvidenceCard）发布到黑板，供其他角色复用：
  assets     BI 通道                 → 真实可用的基础资产（元信息最可信）
  lineage    知识通道·图谱路          → 沿血缘关联的表 / 数据集 / 图表；以 assets 卡里的资产作种子（复用优先，没有再求助）
  caliber    知识通道·向量路          → 口径定义 + chunk 关联的资产（经 BI 取回元信息）+ 关联指标 / 维度实体
  dimension  维值解析                 → 维度取值对应的字段
"""
from __future__ import annotations

from statistics import mean

from ..collab.hub import CollabHub
from ..schemas.collab import EvidenceCard, Need, new_card_id
from ..schemas.knowledge import KnowledgeKind
from ..schemas.recall import CandidateAsset, Evidence, EvidenceKind
from ..services.normalize import CH_BI, CH_KGRAPH, CH_KVEC, from_bi, from_graph, from_knowledge
from .base import new_evidence_id

DEFAULT_ASSET_TYPE = {"assets": "dataset", "lineage": "table", "caliber": "knowledge", "dimension": "table"}


def make_need(hub: CollabHub, kind: str, subject: str) -> Need:
    return Need(kind=kind, subject=subject, domains=hub.env.domains, asset_type=DEFAULT_ASSET_TYPE[kind])  # type: ignore[arg-type]


def _conf(cands: list[CandidateAsset]) -> float:
    """证据卡置信度：前三个候选的平均检索分；没有候选 = 0（空卡）。"""
    top = sorted((c.retrieval_score for c in cands), reverse=True)[:3]
    return round(mean(top), 4) if top else 0.0


def _topk(hub: CollabHub, limit: int | None = None) -> int:
    """召回多少条：LLM 决定的 limit（已在改写 query 时钳制到 [MIN, MAX]）优先，否则默认 top-k。"""
    return limit if limit and limit > 0 else hub.deps.settings.PREINJECT_RETRIEVAL_TOPK


async def _carrier_card(hub: CollabHub, role: str, metric: str, who: str):
    """血缘 / 口径角色需要「谁承载这个指标」的 BI 证据卡：复用优先；没有时——自己就能查 BI（如全通道角色）就自己查，
    否则才向别的角色求助（求助同类角色纯属多余的开销，还会白花一次 LLM 自评）。"""
    agent = hub.agents.get(role)
    produce = (lambda: fetch_assets(hub, role, metric)) if agent is not None and "assets" in agent.spec.serves else None
    return await hub.obtain(make_need(hub, "assets", metric), requester=role, who=who, produce=produce)


async def fetch_assets(hub: CollabHub, role: str, metric: str, hints: list[str] | None = None, query: str | None = None, limit: int | None = None) -> EvidenceCard:
    """BI 通道。hints = 其他角色的发现里已知的资产 id（如血缘找到的表）：一并解析元信息，认证的数据集 / 图表挂到同一张卡上。"""
    env = hub.env
    found = {c.asset_id: c for c in await env.deps.bi.search(query or metric, limit=_topk(hub, limit), cookie=env.cookie, params=env.recall_params)}
    thin = lambda c: not c.meta.grain or not c.meta.partition_latest                  # 元信息缺失的资产需要重新解析
    hinted = [h for h in (hints or []) if h not in found or thin(found[h])]
    for c in await env.deps.bi.get(hinted) if hinted else []:
        if c.asset_id not in found or not thin(c):                                    # 解析得到的元信息更完整才替换
            found[c.asset_id] = c
    found = list(found.values())
    cands = from_bi(found)
    evs = [Evidence(id=new_evidence_id(), asset_id=c.asset_id, kind=EvidenceKind.GRAIN, statement=f"{c.name} 粒度={c.meta.grain or '未知'}",
                    value=c.meta.grain, role=hub.role_of(role), source="bi_asset", confidence=0.7) for c in cands]
    card = EvidenceCard(id=new_card_id(), key=f"assets:{metric}", kind="assets", subject=metric, author=role, asset_ids=[c.asset_id for c in cands],
                        evidence_ids=[e.id for e in evs], confidence=_conf(cands), domains=env.domains, channels=[CH_BI], seeds=list(hints or []),
                        summary=f"BI 通道为「{metric}」找到 {len(cands)} 个基础资产" + (f"：{', '.join(c.asset_id for c in cands[:3])}" if cands else "")
                        + (f"（含转交来的 {len(hinted)} 个待解析资产）" if hinted else ""))
    return await hub.board.publish(card, assets=cands, evidences=evs)


def lineage_seeds(hub: CollabHub, metric: str, bi_card: EvidenceCard | None, limit: int = 5) -> list[str]:
    """血缘通道的种子 = 已有召回内容：预注入知识指明的该口径资产（含会话确认）在前，BI 通道召回的资产（经黑板复用）在后。"""
    from ..stages.s3_planning.levels import mentions
    env = hub.env
    seeds: list[str] = []
    for k in env.knowledge:
        if k.kind != KnowledgeKind.DIMENSION and mentions(k, metric):
            seeds += [a for a in k.asset_ids if a not in seeds]
    seeds += [a for a in env.known_assets if a not in seeds]
    if bi_card is not None and not bi_card.empty:
        seeds += [a for a in bi_card.asset_ids if a not in seeds]
    return seeds[:limit]


async def fetch_lineage(hub: CollabHub, role: str, metric: str, who: str, hints: list[str] | None = None, query: str | None = None, limit: int | None = None) -> EvidenceCard:
    """知识通道·图谱路：以已有召回内容为种子沿血缘找关联的表 / 数据集 / 图表。"""
    env = hub.env
    bi = await _carrier_card(hub, role, metric, who)          # 复用优先；自己能查 BI 就自己查，否则求助 BI 角色；都没有就只用知识种子
    seeds = list(dict.fromkeys([*(hints or []), *lineage_seeds(hub, metric, bi.card)]))[:5]          # 转交带来的资产在前（如口径文档关联的资产）
    names = [n for a in seeds if (n := (hub.board.pool.get(a).name if hub.board.pool.get(a) is not None else a))]
    params = {**env.recall_params, "seed_assets": seeds, "seed_names": list(dict.fromkeys(names))}
    cands = from_graph(await env.deps.lineage.search(query or metric, limit=_topk(hub, limit), cookie=env.cookie, params=params))
    for c in cands:
        c.meta.extra["lineage_seeds"] = list(seeds)
    evs = [Evidence(id=new_evidence_id(), asset_id=c.asset_id, kind=EvidenceKind.USAGE, statement=f"血缘通道命中 {c.name}", role=hub.role_of(role),
                    source="graph", confidence=0.6) for c in cands]
    card = EvidenceCard(id=new_card_id(), key=f"lineage:{metric}", kind="lineage", subject=metric, author=role, asset_ids=[c.asset_id for c in cands],
                        evidence_ids=[e.id for e in evs], confidence=_conf(cands), domains=env.domains, channels=[CH_KGRAPH], seeds=list(seeds),
                        summary=f"知识图谱沿血缘为「{metric}」扩展出 {len(cands)} 个关联资产" + (f"（种子 {seeds}）" if seeds else "（无种子，按口径名检索）"))
    return await hub.board.publish(card, assets=cands, evidences=evs)


async def fetch_caliber(hub: CollabHub, role: str, metric: str, who: str, hints: list[str] | None = None, query: str | None = None, limit: int | None = None) -> EvidenceCard:
    env = hub.env
    items = list(await env.deps.caliber_kb.search(query or metric, limit=_topk(hub, limit), cookie=env.cookie, params=env.recall_params))
    items += [k for k in env.knowledge if k.id not in {i.id for i in items}]              # 复用预注入知识，避免重复检索
    linked = [i for i in items if i.asset_ids and i.kind in (KnowledgeKind.METRIC_DEF, KnowledgeKind.ASSET_META, KnowledgeKind.DIMENSION)]
    ids = list(dict.fromkeys(a for i in linked for a in i.asset_ids))
    resolved = {c.asset_id: c for c in await env.deps.bi.get(ids)} if ids else {}           # 知识关联的资产：经 BI 取回真实元信息
    cands, ents = from_knowledge(linked, resolved)
    evs = [Evidence(id=new_evidence_id(), asset_id=aid, kind=EvidenceKind.CALIBER, statement=it.content, value=it.content, role=hub.role_of(role),
                    source="knowledge_hybrid", confidence=max(0.3, it.relevance))
           for it in items if it.kind == KnowledgeKind.METRIC_DEF for aid in it.asset_ids]
    asset_ids = [c.asset_id for c in cands]
    if not evs:                                                                              # 没有口径证据：去问谁承载这个指标
        carriers = await _carrier_card(hub, role, metric, who)
        if carriers.card is not None:
            asset_ids += [a for a in carriers.card.asset_ids if a not in asset_ids]
    conf = round(max((e.confidence for e in evs), default=0.3 if asset_ids else 0.0), 4)
    card = EvidenceCard(id=new_card_id(), key=f"caliber:{metric}", kind="caliber", subject=metric, author=role, asset_ids=asset_ids,
                        evidence_ids=[e.id for e in evs], entities=ents, confidence=conf, domains=env.domains, channels=[CH_KVEC], seeds=list(hints or []),
                        summary=f"知识通道为「{metric}」找到 {len(evs)} 条口径证据，关联 {len(asset_ids)} 个资产、{len(ents)} 个实体")
    return await hub.board.publish(card, assets=cands, evidences=evs)


async def fetch_dimension(hub: CollabHub, role: str, term: str) -> EvidenceCard:
    env = hub.env
    items = [i for i in await env.deps.kh.search(term, limit=3) if i.kind == KnowledgeKind.DIMENSION]
    evs = [Evidence(id=new_evidence_id(), asset_id=aid, kind=EvidenceKind.FIELD, statement=f"维值「{term}」→ 字段 {it.title}", value=it.title,
                    role=hub.role_of(role), source="dim_value", confidence=max(0.3, it.relevance)) for it in items for aid in it.asset_ids]
    cands, ents = from_knowledge(items, {})
    card = EvidenceCard(id=new_card_id(), key=f"dimension:{term}", kind="dimension", subject=term, author=role, asset_ids=[c.asset_id for c in cands],
                        evidence_ids=[e.id for e in evs], entities=ents, confidence=round(max((i.relevance for i in items), default=0.0), 4),
                        domains=env.domains, channels=[CH_KVEC], summary=f"维值「{term}」对应 {len(items)} 个维度字段")
    return await hub.board.publish(card, assets=cands, evidences=evs)
