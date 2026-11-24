"""① 知识预注入相关数据结构。"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class KnowledgeKind(str, Enum):
    METRIC_DEF = "metric_def"          # 指标定义 / 口径
    BUSINESS_TERM = "business_term"    # 业务术语
    ASSET_META = "asset_meta"          # 资产元数据（粒度、分区、字段）
    DIMENSION = "dimension"
    PAIR = "pair"                      # 历史已验证 查询—资产 选择的摘要


class KnowledgeOrigin(str, Enum):
    PAIR = "pair"            # 路 2：历史已验证 query
    RETRIEVAL = "retrieval"  # 路 1：context-query 检索


class KnowledgeItem(BaseModel):
    """知识块 k。utility(u_k) 由 relevance 推出，cost(c_k) 即 token_cost。"""

    id: str
    kind: KnowledgeKind
    title: str
    content: str
    origin: KnowledgeOrigin
    routes: list[str] = Field(default_factory=list)      # 命中该块的路：pair / retrieval（双路命中会同时包含）
    asset_ids: list[str] = Field(default_factory=list)   # 关联资产
    terms: list[str] = Field(default_factory=list)       # 显式覆盖的查询词/概念
    relevance: float = 0.0                               # 与当前 query 的相关度 [0,1]
    token_cost: int = 0
    source_pair_id: str | None = None
    source_uri: str = ""
    consistency: str = "none"                            # none | possible_inconsistent（来自 context-query）


class KnowledgeSnapshot(BaseModel):
    """历史 pair 中"用到的知识"的快照：检索服务的 chunk 可能变更，这里保存当时的内容。"""

    id: str
    kind: KnowledgeKind
    title: str
    content: str
    source_uri: str = ""


class RetrievalPack(BaseModel):
    """context-query 的一次响应（路 1）。"""

    items: list[KnowledgeItem] = Field(default_factory=list)
    context_pack_id: str | None = None
    partial: bool = False
    failed_sources: list[str] = Field(default_factory=list)
    failed_reasons: list[dict[str, Any]] = Field(default_factory=list)


class DroppedKnowledge(BaseModel):
    item_id: str
    title: str = ""
    reason: str  # over_budget | redundant | conflict | low_gain（① 全局选择）；kind_filtered | topk | over_budget（视图排除）
    detail: str = ""


class SelectionStep(BaseModel):
    """贪心每一轮的选择，便于回放与调参。"""

    step: int
    item_id: str
    utility: float                                       # u_k
    cost: int                                            # c_k
    delta: float                                         # Δ(k|S)
    ratio: float                                         # Δ / c_k
    interactions: dict[str, float] = Field(default_factory=dict)   # 与已选块的非零 w_jk


class PairMatch(BaseModel):
    """路 2 命中的一条历史 query（展示 / 调试用）。"""

    pair_id: str
    query: str
    similarity: float
    asset_ids: list[str] = Field(default_factory=list)
    knowledge_titles: list[str] = Field(default_factory=list)
    token_cost: int = 0
    reliability: float = 1.0


class RouteStat(BaseModel):
    """某一路召回（进入选择之前）的概况。"""

    count: int = 0
    tokens: int = 0
    top_relevance: float = 0.0
    kinds: dict[str, int] = Field(default_factory=dict)


# ── 按消费方分发的知识视图 ────────────────────────────────────────────────
CONSUMER_COORDINATION = "coordination"     # ② 意图 / 契约补全
CONSUMER_PLANNER = "planner"               # ③ 入口定级 / 规划
CONSUMER_SELECTION = "selection"           # ⑤ 资产选择
CONSUMER_CHECKS = "recall.checks"          # ④ 角色内联核对（维度覆盖的知识桥接）：不属于某个角色，各角色共用


def recall_consumer(role: str) -> str:
    """④ 某个角色模板（RoleSpec.name）的视图名。同一角色的多个副本看到同一份视图。"""
    return f"recall.{role}"


class KnowledgeView(BaseModel):
    """某个消费方（agent）看到的知识：`injected` 的子集，只存 id，不重复存正文（体积小，且不可能凭空多出知识）。

    冲突 / 冗余已在全局选择里裁决一次，视图只做「哪些对该消费方有用」的投影，不会让不同 agent 对同一冲突各自择一。"""

    consumer: str
    item_ids: list[str] = Field(default_factory=list)       # 保持 injected 的顺序
    token_budget: int | None = None                         # 该视图的 token 上限（None = 不限）
    used_tokens: int = 0
    excluded: list[DroppedKnowledge] = Field(default_factory=list)   # reason: kind_filtered | topk | over_budget


class PreInjectionResult(BaseModel):
    injected: list[KnowledgeItem] = Field(default_factory=list)
    views: dict[str, KnowledgeView] = Field(default_factory=dict)    # consumer -> 视图；为空表示未分发（消费方回落到 injected 全量）
    dropped: list[DroppedKnowledge] = Field(default_factory=list)
    trace: list[SelectionStep] = Field(default_factory=list)
    token_budget: int = 0                                # 容量 L（硬上界）
    tau: float = 0.0                                     # 停止阈值 τ
    used_tokens: int = 0
    candidate_count: int = 0                             # 合并去重后进入选择的块数
    route_status: dict[str, str] = Field(default_factory=dict)   # pair/retrieval -> ok | partial:.. | failed:..
    route_stats: dict[str, RouteStat] = Field(default_factory=dict)
    pair_matches: list[PairMatch] = Field(default_factory=list)
    total_value: float = 0.0                             # V(S) = Σ 各轮 Δ
    covered_terms: list[str] = Field(default_factory=list)
    uncovered_terms: list[str] = Field(default_factory=list)   # 供 ③ 判断是否需要更多探查角色
    pair_hits: int = 0
    retrieval_hits: int = 0

    @property
    def coverage(self) -> float:
        total = len(self.covered_terms) + len(self.uncovered_terms)
        return len(self.covered_terms) / total if total else 0.0

    def view(self, consumer: str) -> list[KnowledgeItem]:
        """该消费方看到的知识。没有分发视图（未启用 / 未登记的消费方）→ 回落到 `injected` 全量；
        视图存在但为空 → 就是空（该消费方不需要知识，不能回落）。"""
        v = self.views.get(consumer)
        if v is None:
            return list(self.injected)
        wanted = set(v.item_ids)
        return [i for i in self.injected if i.id in wanted]

    def consumed_ids(self) -> set[str]:
        """至少被一个消费方看到的知识 id；没有分发视图时视为全部被消费。"""
        if not self.views:
            return {i.id for i in self.injected}
        return {i for v in self.views.values() for i in v.item_ids}


class VerifiedPair(BaseModel):
    """通过 ⑥ 核验的 一条历史 query：问题 + 用到的知识 + 最终选择的资产。闭环回流的最小单元。"""

    pair_id: str
    query: str
    terms: list[str] = Field(default_factory=list)
    asset_ids: list[str] = Field(default_factory=list)       # 最终选择的资产，主资产在前
    knowledge: list[KnowledgeSnapshot] = Field(default_factory=list)   # 用到的知识（快照）
    knowledge_ids: list[str] = Field(default_factory=list)
    contract_digest: dict[str, str] = Field(default_factory=dict)  # 口径/粒度/时间范围等关键槽位
    verification_checks: list[str] = Field(default_factory=list)
    verified_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    hit_count: int = 0           # 被预注入的次数
    success_count: int = 0       # 被预注入且最终通过核验的次数
