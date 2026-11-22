"""④ 并行召回产物：证据、候选资产、共享黑板、冲突。"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from .collab import Ask, AskMode, AskReply, CollabTrace, EvidenceCard, RelatedEntity, ReplyKind  # noqa: F401  (Ask* 已迁入 collab，此处重导出)


class EvidenceKind(str, Enum):
    CALIBER = "caliber"        # 口径证据
    GRAIN = "grain"            # 粒度证据
    FIELD = "field"            # 字段/维度覆盖
    PARTITION = "partition"
    FRESHNESS = "freshness"
    USAGE = "usage"            # 使用热度 / 血缘
    OTHER = "other"


class Evidence(BaseModel):
    id: str
    asset_id: str
    kind: EvidenceKind
    statement: str                       # 例如 "DAU 按 user_id 去重"
    value: Any = None                    # 结构化取值（用于冲突检测，如 grain="day"）
    role: str                            # 产出该证据的角色
    source: str = ""                     # bi_asset | knowledge_hybrid | pair ...
    confidence: float = 0.5
    subject: str = ""                    # 这条证据针对哪个口径 / 指标（如 LLM 相关性评估是对着某个指标判的）；空 = 不限定


class AssetMeta(BaseModel):
    grain: str | None = None
    partition_field: str | None = None
    partition_latest: str | None = None  # 最新分区（用于新鲜度）
    fields: list[str] = Field(default_factory=list)
    owner: str | None = None
    permission_scope: str | None = None  # 权限所需 scope
    extra: dict[str, Any] = Field(default_factory=dict)
    # ── ⑤ 资产选择所需的类型化元信息（缺失即「盲区」，选择阶段会如实标注而不是臆测）──
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    field_aliases: dict[str, str] = Field(default_factory=dict)   # 物理字段名 → 中文名
    field_roles: dict[str, str] = Field(default_factory=dict)     # 物理字段名 → metric | dimension（BI 纠错后的角色）
    primary_keys: list[str] = Field(default_factory=list)         # 主键字段 = 分析粒度（参考提示词里的 ★）
    main_table: str | None = None                                 # 数据集背后的主表名
    heat_30d: int | None = None
    ref_count: int | None = None
    is_onedash: bool = False                                      # OneDash / 核心驾驶舱级资产
    is_strict: bool | None = None                                 # chart：指标 + 取数规则齐全，可直接取数
    metric_ids: list[str] = Field(default_factory=list)           # chart：展示的指标
    supported_dims: list[str] = Field(default_factory=list)       # chart：支持的维度；[] = 无限制
    cube_filters: dict[str, str] = Field(default_factory=dict)    # cube / 切面表必须限定的切面字段 → 取值（如 operate_type: 整体）
    sql_rules: list[str] = Field(default_factory=list)            # SQL 约束（如「日期分区默认筛最新数据日期」）
    knowledge_docs: list[str] = Field(default_factory=list)       # 关联的业务术语文档
    domains: list[str] = Field(default_factory=list)              # 资产所属业务域


class CandidateAsset(BaseModel):
    asset_id: str
    name: str
    asset_type: Literal["table", "dataset", "dashboard", "metric", "chart"] = "table"
    meta: AssetMeta = Field(default_factory=AssetMeta)
    evidence_ids: list[str] = Field(default_factory=list)
    found_by: list[str] = Field(default_factory=list)  # 去重后保留所有发现该资产的角色
    channels: list[str] = Field(default_factory=list)  # 来源通道：bi / knowledge:vector / knowledge:graph / known
    retrieval_score: float = 0.0


class Conflict(BaseModel):
    """冲突显式保留，不静默择一。"""

    asset_id: str
    topic: EvidenceKind
    evidence_ids: list[str]
    values: list[Any]
    note: str = ""


class BoardEntryKind(str, Enum):
    FINDING = "finding"
    CLAIM = "claim"          # 认领探查方向，避免重复探查
    BROADCAST = "broadcast"
    DECLINE = "decline"


class BoardEntry(BaseModel):
    seq: int
    author: str
    kind: BoardEntryKind
    topic: str
    payload: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.5


class RoleReport(BaseModel):
    node_id: str
    role: str
    status: Literal["ok", "failed", "timeout", "skipped"] = "ok"
    error: str = ""
    candidates: list[CandidateAsset] = Field(default_factory=list)
    evidences: list[Evidence] = Field(default_factory=list)
    asks: list[Ask] = Field(default_factory=list)
    replies: list[AskReply] = Field(default_factory=list)
    elapsed_ms: int = 0


class RecallResult(BaseModel):
    candidates: list[CandidateAsset] = Field(default_factory=list)   # 已合并去重
    evidences: list[Evidence] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    role_reports: list[RoleReport] = Field(default_factory=list)
    board: list[BoardEntry] = Field(default_factory=list)
    raw_candidate_count: int = 0
    dedup_removed: int = 0
    cards: list[EvidenceCard] = Field(default_factory=list)              # 黑板上的证据卡（可复用的探查产出）
    related_entities: list[RelatedEntity] = Field(default_factory=list)  # 知识通道带出的关联指标 / 维度 / 资产
    by_channel: dict[str, int] = Field(default_factory=dict)             # 候选池里各来源通道组合的资产数
    collab: CollabTrace = Field(default_factory=CollabTrace)             # 复用 / 询问 / 婉拒 / 画像更新轨迹
