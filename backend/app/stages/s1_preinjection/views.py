"""① 知识分发：把全局选出的 `injected` 投影成每个消费方（agent）自己的视图。

为什么在 ① 做：之前各下游自己临时裁剪（② 取前 5 条、④ 取前 8 条并截断、⑤ 剔除 PAIR……），规则散落、不可测；
现在「谁需要什么知识」集中声明在 profile 里，下游只读 `pre.view(consumer)`。

不变量（对应 AGENTS.md 铁律）：
  - 视图 ⊆ injected：冲突 / 冗余在全局贪心选择里只裁决一次，不会因视图不同让各 agent 对同一冲突各自择一（铁律 5）。
  - 不引入新的外部依赖：纯函数，输入就是已选出的知识（铁律 2）。
  - 视图缺失时下游回落到 injected 全量，规则兜底不受影响（铁律 3）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ...schemas.knowledge import (
    CONSUMER_CHECKS, CONSUMER_COORDINATION, CONSUMER_PLANNER, CONSUMER_SELECTION, DroppedKnowledge, KnowledgeItem, KnowledgeKind,
    KnowledgeView, recall_consumer,
)

K = KnowledgeKind


@dataclass(frozen=True)
class KnowledgeProfile:
    """某个消费方对知识的偏好。

    kinds       允许的知识类型 → 优先级权重（> 0）；不在其中的类型直接排除
    topk        最多保留几块（按 relevance × 权重取最高；None = 不限）
    token_budget  视图的 token 硬上限（None = 不限）
    """

    consumer: str
    kinds: Mapping[KnowledgeKind, float]
    topk: int | None = None
    token_budget: int | None = None

    def __post_init__(self) -> None:
        if any(w <= 0 for w in self.kinds.values()):
            raise ValueError(f"profile {self.consumer}: kind weights must be > 0")


def _profile(consumer: str, kinds: dict[KnowledgeKind, float], **kw) -> tuple[str, KnowledgeProfile]:
    return consumer, KnowledgeProfile(consumer, kinds, **kw)


# 默认画像。权重 / topk 只表达「相对偏好」；改契约式的约束（谁绝不能看到什么）体现在 kinds 里。
DEFAULT_PROFILES: dict[str, KnowledgeProfile] = dict([
    # ② 意图 / 契约补全：要指标口径、业务术语、维度；历史摘要只作弱提示
    _profile(CONSUMER_COORDINATION, {K.METRIC_DEF: 1.0, K.BUSINESS_TERM: 1.0, K.DIMENSION: 0.9, K.ASSET_META: 0.6, K.PAIR: 0.6}),
    # ③ 入口定级：判断「知识是否已覆盖」，需要全类型（含历史选择指明的资产）；不裁剪，保证覆盖判断不漏
    _profile(CONSUMER_PLANNER, {K.METRIC_DEF: 1.0, K.ASSET_META: 1.0, K.BUSINESS_TERM: 0.9, K.DIMENSION: 0.9, K.PAIR: 0.8}),
    # ④ 召回角色：各取所需
    _profile(recall_consumer("bi_asset_discovery"), {K.ASSET_META: 1.2, K.METRIC_DEF: 1.0, K.PAIR: 1.0}, topk=6),
    _profile(recall_consumer("lineage_discovery"), {K.ASSET_META: 1.2, K.METRIC_DEF: 1.0, K.PAIR: 0.8}, topk=6),   # 要带 asset_ids 的块作血缘种子
    _profile(recall_consumer("caliber_lookup"), {K.METRIC_DEF: 1.3, K.BUSINESS_TERM: 1.0, K.ASSET_META: 0.8, K.DIMENSION: 0.8}, topk=8),
    _profile(recall_consumer("dim_value_resolver"), {K.DIMENSION: 1.5, K.BUSINESS_TERM: 0.8, K.ASSET_META: 0.6}, topk=5),
    _profile(recall_consumer("omni_discovery"), {K.METRIC_DEF: 1.0, K.ASSET_META: 1.0, K.BUSINESS_TERM: 0.9, K.DIMENSION: 0.9, K.PAIR: 0.8}, topk=8),
    # ④ 内联核对 / ⑤ 选择：做业务词 → 字段的同义桥接，要术语 / 维度 / 指标 / 元数据；历史摘要块会污染同义集，排除
    _profile(CONSUMER_CHECKS, {K.METRIC_DEF: 1.0, K.BUSINESS_TERM: 1.0, K.DIMENSION: 1.0, K.ASSET_META: 1.0}),
    _profile(CONSUMER_SELECTION, {K.METRIC_DEF: 1.0, K.BUSINESS_TERM: 1.0, K.DIMENSION: 1.0, K.ASSET_META: 1.0}),
])


def build_view(profile: KnowledgeProfile, injected: list[KnowledgeItem]) -> KnowledgeView:
    """按 profile 从 injected 里投影出视图。先按类型过滤，再按 relevance×权重 取 topk / token_budget 内的最优，结果保持 injected 的顺序。"""
    excluded: list[DroppedKnowledge] = []
    eligible: list[tuple[int, KnowledgeItem, float]] = []
    for pos, it in enumerate(injected):
        w = profile.kinds.get(it.kind)
        if w is None:
            excluded.append(DroppedKnowledge(item_id=it.id, title=it.title, reason="kind_filtered", detail=f"kind={it.kind.value}"))
        else:
            eligible.append((pos, it, it.relevance * w))

    kept: dict[int, KnowledgeItem] = {}
    used = 0
    for pos, it, score in sorted(eligible, key=lambda e: (-e[2], e[0])):
        if profile.topk is not None and len(kept) >= profile.topk:
            excluded.append(DroppedKnowledge(item_id=it.id, title=it.title, reason="topk", detail=f"score={score:.3f} topk={profile.topk}"))
            continue
        cost = max(1, it.token_cost)
        if profile.token_budget is not None and used + cost > profile.token_budget:
            excluded.append(DroppedKnowledge(item_id=it.id, title=it.title, reason="over_budget", detail=f"cost={cost} > remaining={profile.token_budget - used}"))
            continue
        kept[pos] = it
        used += cost
    return KnowledgeView(consumer=profile.consumer, item_ids=[kept[p].id for p in sorted(kept)], token_budget=profile.token_budget,
                         used_tokens=used, excluded=excluded)


def build_views(injected: list[KnowledgeItem], profiles: Mapping[str, KnowledgeProfile] | list[KnowledgeProfile] | None = None) -> dict[str, KnowledgeView]:
    ps = DEFAULT_PROFILES.values() if profiles is None else (profiles.values() if isinstance(profiles, Mapping) else profiles)
    return {p.consumer: build_view(p, injected) for p in ps}
