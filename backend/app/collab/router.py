"""路由打分：角色产生需求且黑板上没有可复用证据时，决定「定向询问」还是「黑板广播」。

对每个候选角色按画像打分（均 [0,1]）：
    kind      该角色是否可服务这类需求（assets / lineage / caliber / dimension）            权重 0.40
    asset     期望资产类型是否在其擅长范围内（需求未指定 → 中性 0.5）                      权重 0.15
    domain    请求涉及的业务域是否都在其覆盖域内（需求未指定 → 中性 0.5）                   权重 0.15
    terms     需求文本与画像关键词的重叠系数                                               权重 0.15
    focus     专才加成 = 1 / 可服务类型数（全通道角色是通才，打折）                         权重 0.15
    score = Σ权重·分项 × (0.5 + 0.5·quality)   quality 是历史响应质量，被忽略 / 婉拒越多越低；质量掉到底，最高分 ≤ 0.5，不再被定向

定向条件（全部满足）：最高分角色服务该类需求；请求不跨域；请求的域在其覆盖域内；最高分 ≥ 阈值。
否则回退广播：uncovered（没人服务）/ cross_domain（跨域）/ domain_uncovered（域画像未覆盖）/ low_confidence（置信度低）。
"""
from __future__ import annotations

from ..schemas.collab import AskMode, Need, RoleProfile, RouteDecision, RoutingScore
from ..text_utils import extract_terms, overlap_coeff

W_KIND, W_ASSET, W_DOMAIN, W_TERMS, W_FOCUS = 0.40, 0.15, 0.15, 0.15, 0.15


def score_profile(need: Need, p: RoleProfile) -> RoutingScore:
    kind = 1.0 if need.kind in p.serves else 0.0
    asset = 0.5 if need.asset_type is None else (1.0 if need.asset_type in p.asset_types else 0.0)
    doms = set(need.domains)
    domain = 0.5 if not doms else (1.0 if doms <= set(p.domains) else 0.0)
    terms = overlap_coeff(set(extract_terms(need.text)), set(extract_terms(" ".join(p.keywords)))) if p.keywords else 0.0
    focus = 1.0 / len(p.serves) if p.serves else 0.0
    qf = 0.5 + 0.5 * max(0.0, min(1.0, p.quality))
    base = W_KIND * kind + W_ASSET * asset + W_DOMAIN * domain + W_TERMS * terms + W_FOCUS * focus
    parts = {"kind": kind, "asset_type": asset, "domain": domain, "terms": round(terms, 4), "focus": round(focus, 4), "quality_factor": round(qf, 4)}
    return RoutingScore(role=p.role, score=round(base * qf, 4), parts=parts)


def rank(need: Need, profiles: list[RoleProfile]) -> list[RoutingScore]:
    return sorted((score_profile(need, p) for p in profiles), key=lambda s: (-s.score, s.role))


def decide(need: Need, profiles: list[RoleProfile], *, direct_threshold: float = 0.6) -> RouteDecision:
    scores = rank(need, profiles)
    by_role = {p.role: p for p in profiles}
    top = scores[0] if scores else None

    def broadcast(reason: str) -> RouteDecision:
        return RouteDecision(mode=AskMode.BROADCAST, target=None, confidence=top.score if top else 0.0, scores=scores, reason=reason)

    if top is None or top.parts["kind"] == 0:
        return broadcast("uncovered：现有角色画像都不服务该类需求")
    if len(set(need.domains)) > 1:
        return broadcast(f"cross_domain：请求跨域 {sorted(set(need.domains))}")
    if need.domains and not set(need.domains) <= set(by_role[top.role].domains):
        return broadcast(f"domain_uncovered：{top.role} 的画像不覆盖业务域 {need.domains}")
    if top.score < direct_threshold:
        return broadcast(f"low_confidence：最高匹配度 {top.score:.2f} < 阈值 {direct_threshold:g}")
    return RouteDecision(mode=AskMode.DIRECTED, target=top.role, confidence=top.score, scores=scores,
                         reason=f"高置信命中 {top.role} 的画像（{top.score:.2f} ≥ {direct_threshold:g}）")
