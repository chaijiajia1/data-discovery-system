"""② 协调阶段产物：任务契约（Task Contract）。"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from .intent import Intent


class ValueSource(str, Enum):
    """槽位取值来源——契约里每个条件都必须可追溯。"""

    USER = "user"                # 用户原话 / 用户主动补充
    KNOWLEDGE = "knowledge"      # 预注入知识（口径定义等）
    DEFAULT_RULE = "default"     # 默认规则补全
    INFERRED = "inferred"        # 推断 / 兜底（如 metrics 缺失时取 intent.keywords）


class Slot(BaseModel):
    value: Any = None
    source: ValueSource = ValueSource.DEFAULT_RULE
    rule_id: str | None = None   # 命中的默认规则 / 知识条目 id
    note: str = ""


SLOT_NAMES = ("metrics", "caliber", "grain", "time_range", "dimensions", "filters")


class TaskContract(BaseModel):
    query: str
    intent: Intent | None = None                      # 协调 agent 的意图识别结果（domain / sub_domain / intent_type …）
    metrics: Slot = Field(default_factory=Slot)       # 指标
    caliber: Slot = Field(default_factory=Slot)       # 口径
    grain: Slot = Field(default_factory=Slot)         # 粒度
    time_range: Slot = Field(default_factory=Slot)    # 时间范围
    dimensions: Slot = Field(default_factory=Slot)    # 拆分维度
    filters: Slot = Field(default_factory=Slot)       # 过滤条件
    extras: dict[str, Slot] = Field(default_factory=dict)
    status: Literal["ready"] = "ready"                # 恒为 ready：缺失信息走规则兜底，不再暂停追问（字段保留以兼容 API 响应）

    def slots(self) -> dict[str, Slot]:
        return {n: getattr(self, n) for n in SLOT_NAMES} | self.extras

    def provenance(self) -> dict[str, str]:
        """槽位 → 来源，便于前端展示与审计。"""
        return {k: v.source.value for k, v in self.slots().items() if v.value not in (None, [], "")}
