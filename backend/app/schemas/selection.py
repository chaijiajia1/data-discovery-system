"""⑤ 资产选择产物（对齐 old-version asset_selection.md 的三档输出）。

    primary    每个口径（指标）恰好 1 个主资产——多个口径各选各的，同一资产最优于多个口径时合并为一条
    alternates 备选：相关但不是最优 / 权限未开通（可申请）/ chart 条件不满足 / cube 让位于非 cube / scope 过宽 …
    discarded  不用：无目标指标字段 / 粒度或 scope 不满足 / 时间粒度比要求粗 / 无法支持维度过滤 …
所有候选必须且只能出现在三档之一；每个决定都带结构化理由（命中 / 未命中 / 采用规则 / 降权信号 / 盲区），可追溯。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Hit(BaseModel):
    """业务词桥接到候选的哪个位置。`matched` 逐字复制候选字段原文（含中文别名与 ★ 主键标记）。"""

    term: str
    type: Literal["metric", "dimension"]
    matched: str
    location: str                      # fields[i] / name / description / tags[i] / metric_ids[i] / supported_dims[i] / knowledge_context[i]
    how: Literal["字面相同", "同义", "部分", "推断"]


class Miss(BaseModel):
    term: str
    type: Literal["metric", "dimension"]
    note: str = ""


class Reason(BaseModel):
    conclusion: str = ""               # 一句话裁决依据
    hits: list[Hit] = Field(default_factory=list)
    misses: list[Miss] = Field(default_factory=list)
    rules: list[str] = Field(default_factory=list)          # 触发的核心选用原则
    downweights: list[str] = Field(default_factory=list)    # 本可加分但被压制：「规则名:被压制原因」
    blind_spots: list[str] = Field(default_factory=list)    # 只能引用后端客观计算出的 signals，不得自由生成


class FieldRef(BaseModel):
    """primary 在该资产上实际会用到的字段。"""

    field_name: str
    field_alias: str = ""
    role: Literal["metric", "dimension"]
    dim_values: list[str] | None = None     # 维值命中 / cube 切面 / 用户粒度切片时填
    note: str = ""                          # 如「分区字段，需显式指定时间范围」「用户粒度宽表，聚合方式用求和」


class RankScore(BaseModel):
    asset_id: str
    dims: dict[str, float] = Field(default_factory=dict)    # coverage / type_rank / layer_rank / heat / retrieval …（排序用的各项）
    total: float = 0.0                                      # 仅供展示的综合分；实际排序按选用原则的优先级，不是按它


class AssetDecision(BaseModel):
    asset_id: str
    role: Literal["primary", "alternate", "discarded"]
    reason: Reason = Field(default_factory=Reason)
    covers: list[str] = Field(default_factory=list)         # 该资产作为 primary 负责的口径；alternate / discarded 为其相关的口径
    query_fields: list[FieldRef] = Field(default_factory=list)   # 仅 primary
    score: RankScore | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    needs_permission: bool = False                          # 权限未开通（降级到 alternates，用户可申请）
    rule_ids: list[str] = Field(default_factory=list)

    @property
    def rationale(self) -> str:
        return self.reason.conclusion


class GroupOutcome(BaseModel):
    """一个口径（指标）的选择结果。"""

    caliber: str
    primary: str | None = None
    mode: Literal["llm", "rule"] = "rule"
    fallback_reason: str = ""
    keywords: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SelectionResult(BaseModel):
    primary: list[AssetDecision] = Field(default_factory=list)
    alternates: list[AssetDecision] = Field(default_factory=list)
    discarded: list[AssetDecision] = Field(default_factory=list)
    knowledge_refs: list[str] = Field(default_factory=list)      # 选用推理中实际参考过的知识条目 id
    ranking: list[RankScore] = Field(default_factory=list)       # 全量排序，便于回放
    risk_flags: list[str] = Field(default_factory=list)          # 如：存在口径冲突、无口径证据、分区未知
    warnings: list[str] = Field(default_factory=list)            # LLM 输出校验 / 兜底提示、② 的概念冲突提示
    groups: list[GroupOutcome] = Field(default_factory=list)
    mode: Literal["llm", "rule", "mixed"] = "rule"

    def decision(self, asset_id: str) -> AssetDecision | None:
        return next((d for d in (*self.primary, *self.alternates, *self.discarded) if d.asset_id == asset_id), None)
