"""内存假实现：本地联调 / 单测使用。资产、表名、看板名均为示意占位。"""
from __future__ import annotations

from typing import Any

from ..schemas.execution import StepResult
from ..schemas.knowledge import KnowledgeItem, KnowledgeKind, KnowledgeOrigin, RetrievalPack
from ..schemas.recall import AssetMeta, CandidateAsset
from ..schemas.task import TaskContract
from ..text_utils import estimate_tokens, extract_terms


def _asset(aid, name, grain, fields, *, part="dt", latest="2026-11-12", scope="public", typ="table"):
    return CandidateAsset(
        asset_id=aid, name=name, asset_type=typ,
        meta=AssetMeta(grain=grain, partition_field=part, partition_latest=latest, fields=fields, permission_scope=scope),
    )


DEMO_ASSETS: list[tuple[CandidateAsset, list[str]]] = [
    (_asset("dws.user_active_d", "User daily-active summary", "day", ["dt", "channel", "platform", "dau"]), ["DAU", "日活", "留存"]),
    (_asset("dws.user_active_valid_d", "Valid daily-active summary", "day", ["dt", "channel", "edau"]), ["EDAU", "有效日活"]),
    (_asset("dws.trade_gmv_d", "Trade GMV daily summary", "day", ["dt", "channel", "gmv"]), ["GMV", "交易", "渠道"]),
    (_asset("dws.trade_gmv_m", "Trade GMV monthly summary", "month", ["dt", "channel", "gmv"]), ["GMV", "交易"]),
    (_asset("dws.ad_ctr_d", "Ad CTR / conversion daily table", "day", ["dt", "platform", "ctr", "cvr"]), ["CTR", "转化", "平台"]),
    (_asset("ads.secret_finance", "Finance confidential table", "day", ["dt", "gmv"], scope="finance"), ["GMV"]),
]


class FakeBIRetrieval:
    def __init__(self, assets=None):
        self.assets = assets if assets is not None else DEMO_ASSETS

    async def get(self, asset_ids: list[str]) -> list[CandidateAsset]:
        by_id = {a.asset_id: a for a, _ in self.assets}
        return [by_id[i].model_copy(update={"retrieval_score": 0.6}, deep=True) if i in by_id else CandidateAsset(asset_id=i, name=i, retrieval_score=0.5)
                for i in asset_ids]

    async def search(self, query: str, *, limit: int = 10, cookie=None, params=None) -> list[CandidateAsset]:
        terms = set(extract_terms(query))
        out: list[CandidateAsset] = []
        for asset, keywords in self.assets:
            hit = len(terms & {k.upper() for k in keywords} | {t for t in terms if any(t in k for k in keywords)})
            if hit:
                out.append(asset.model_copy(update={"retrieval_score": min(1.0, 0.4 + 0.2 * hit)}, deep=True))
        out.sort(key=lambda a: a.retrieval_score, reverse=True)
        return out[:limit]


class FakeLineageRetrieval:
    """血缘通道假实现：只返回明细表；有种子时同时按种子名 / id 里的词做匹配（模拟沿种子资产的血缘扩展），分数低于 BI 通道。"""

    def __init__(self, assets=None):
        self._bi = FakeBIRetrieval(assets)

    async def search(self, query: str, *, limit: int = 10, cookie=None, params=None) -> list[CandidateAsset]:
        seeds = " ".join((params or {}).get("seed_names", []) + (params or {}).get("seed_assets", []))
        hits = {a.asset_id: a for a in await self._bi.search(f"{query} {seeds}".strip(), limit=limit * 3) if a.asset_type == "table"}
        return [a.model_copy(update={"retrieval_score": round(a.retrieval_score * 0.9, 4)}) for a in hits.values()][:limit]


class FakeKnowledgeRetrieval:
    def __init__(self, items: list[KnowledgeItem] | None = None):
        self.items = items if items is not None else self._demo()

    @staticmethod
    def _demo() -> list[KnowledgeItem]:
        def k(i, kind, title, content, terms, assets):
            return KnowledgeItem(
                id=i, kind=kind, title=title, content=content, origin=KnowledgeOrigin.RETRIEVAL,
                terms=terms, asset_ids=assets, token_cost=estimate_tokens(content),
            )
        return [
            k("k_dau", KnowledgeKind.METRIC_DEF, "DAU", "Daily active users, deduplicated by user_id; caliber = any app launch that day", ["DAU", "日活"], ["dws.user_active_d"]),
            k("k_edau", KnowledgeKind.METRIC_DEF, "eDAU", "Valid daily active users: DAU after removing cheating devices; caliber = eDAU", ["DAU", "EDAU", "日活"], ["dws.user_active_valid_d"]),
            k("k_gmv", KnowledgeKind.METRIC_DEF, "GMV", "Gross merchandise value: paid orders, refunds not deducted", ["GMV", "交易"], ["dws.trade_gmv_d", "dws.trade_gmv_m"]),
            k("k_ctr", KnowledgeKind.METRIC_DEF, "CTR", "Click-through rate = clicks / impressions", ["CTR"], ["dws.ad_ctr_d"]),
            k("k_channel", KnowledgeKind.DIMENSION, "Channel", "The channel field is the acquisition channel of the user", ["渠道", "CHANNEL"], []),
        ]

    async def retrieve(self, query, *, user_id=None, cookie=None, max_tokens=None) -> RetrievalPack:
        return RetrievalPack(items=await self.search(query, limit=50), context_pack_id="fake-pack")

    async def search(self, query: str, *, limit: int = 10, cookie=None, params=None) -> list[KnowledgeItem]:
        terms = set(extract_terms(query))
        scored = []
        for it in self.items:
            overlap = len(terms & {t.upper() for t in it.terms}) + len({t for t in terms if any(t in x for x in it.terms)})
            if overlap:
                scored.append((overlap, it.model_copy(update={"relevance": min(1.0, 0.5 + 0.25 * overlap)}, deep=True)))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in scored[:limit]]


class FakePermission:
    def __init__(self, granted_scopes: set[str] | None = None):
        self.granted = granted_scopes if granted_scopes is not None else {"public"}

    async def check(self, user_id, asset: CandidateAsset) -> tuple[bool, str]:
        scope = asset.meta.permission_scope
        if scope and scope not in self.granted:
            return False, f"user lacks scope '{scope}'"
        return True, ""


class FakeEngine:
    """三步执行：resolve_fields → build_query → run_query。可注入失败用于测试回滚。"""

    def __init__(self, fail_at: dict[tuple[str, int], str] | None = None):
        self.fail_at = fail_at or {}   # (step_id, 第几次调用该步) -> 错误信息
        self._calls: dict[str, int] = {}

    def steps(self) -> list[str]:
        return ["resolve_fields", "build_query", "run_query"]

    async def run_step(self, step_id, *, asset, contract, state: dict[str, Any]) -> StepResult:
        n = self._calls[step_id] = self._calls.get(step_id, 0) + 1
        if (step_id, n) in self.fail_at:
            return StepResult(step_id=step_id, status="failed", error=self.fail_at[(step_id, n)])
        if step_id == "resolve_fields":
            return StepResult(step_id=step_id, output={"fields": asset.meta.fields})
        if step_id == "build_query":
            return StepResult(step_id=step_id, output=f"SELECT * FROM {asset.asset_id} LIMIT 10")
        return StepResult(step_id=step_id, output={"columns": asset.meta.fields, "rows": [[1] * len(asset.meta.fields)]})
