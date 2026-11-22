"""④ 协作契约：需求、证据卡、角色能力画像、路由决策、协作轨迹。

协作模型：共享黑板 + 主动定向询问
  需求 Need ──▶ 复用优先（读黑板上的证据卡）──▶ 没有 ──▶ 路由打分 ──┬─ 高置信命中某角色画像 → 定向询问
                                                                  └─ 低置信 / 跨域 / 画像未覆盖 → 黑板广播，角色自愿揭榜
  被点名的角色保有自主权，可以婉拒并说明理由；请求方收到婉拒后降级为广播。
  每次响应被采纳 / 忽略 / 婉拒，都会更新该角色的能力画像（历史响应质量）。
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

NeedKind = Literal["assets", "lineage", "caliber", "dimension"]
Outcome = Literal["adopted", "ignored", "declined", "unanswered"]

QUESTION_TEMPLATES: dict[str, str] = {
    "assets": "哪些资产承载「{s}」？（看板、数据集、图表）",
    "lineage": "「{s}」沿血缘有哪些关联的表与字段？",
    "caliber": "「{s}」的口径定义与使用约束是什么？",
    "dimension": "维度取值「{s}」对应哪些字段？",
}


class Need(BaseModel):
    """角色产生的探查需求。key 相同的需求视为同一件事（黑板上复用 / 认领的单位）。"""

    kind: NeedKind
    subject: str                                   # 口径（指标）名 / 维度取值
    domains: list[str] = Field(default_factory=list)   # 请求涉及的业务域；多于一个即跨域
    asset_type: str | None = None                  # 期望的资产类型（dataset / table / knowledge …），无则中性
    question: str = ""
    # ── 转交（handoff）：由别的角色的发现触发的后续需求 ──
    hint_assets: list[str] = Field(default_factory=list)   # 触发它的发现里已知的资产（血缘的种子 / BI 要解析的 id）
    via: str = ""                                          # 触发它的证据卡 key，如 "caliber:DAU"；空 = 角色自己的需求
    hop: int = 0                                           # 转交深度：0 = 自己的需求，1 = 一次转交，…
    query: str = ""                                        # 检索用的自足 query（LLM 改写 / 换角度后的说法）；空 = 直接用 subject。不参与 key
    limit: int = 0                                         # 召回多少条（LLM 决定）；0 = 默认 top-k。不参与 key

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.subject}"

    @property
    def domain(self) -> str | None:
        return self.domains[0] if self.domains else None

    @property
    def text(self) -> str:
        return self.question or QUESTION_TEMPLATES[self.kind].format(s=self.subject)


class RelatedEntity(BaseModel):
    """知识通道带出的关联实体（指标 / 维度 / 资产）。"""

    kind: Literal["metric", "dimension", "asset"]
    name: str
    asset_ids: list[str] = Field(default_factory=list)
    source: str = ""


class EvidenceCard(BaseModel):
    """黑板上的证据卡：一次探查的可复用产出。复用优先——后来的角色先读卡，有卡就不再询问或广播。"""

    id: str
    key: str                                       # = Need.key
    kind: NeedKind
    subject: str
    author: str                                    # 产出该卡的角色
    asset_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    entities: list[RelatedEntity] = Field(default_factory=list)
    seeds: list[str] = Field(default_factory=list)       # lineage 卡：用来沿血缘扩展的种子资产
    summary: str = ""
    confidence: float = 0.0
    domains: list[str] = Field(default_factory=list)
    channels: list[str] = Field(default_factory=list)
    seq: int = 0
    reuse_count: int = 0

    @property
    def empty(self) -> bool:
        return not (self.asset_ids or self.evidence_ids or self.entities)


# ── 询问 / 回应 ──────────────────────────────────────────────────────────────
class AskMode(str, Enum):
    DIRECTED = "directed"    # 高置信命中某角色画像 → 定向询问
    BROADCAST = "broadcast"  # 低置信 / 跨域 / 画像未覆盖 / 定向被婉拒 → 黑板广播


class ReplyKind(str, Enum):
    ANSWER = "answer"
    DECLINE = "decline"        # 角色自主婉拒，必须说明理由
    DOWNGRADE = "downgrade"    # 请求方收到婉拒后降级为广播
    VOLUNTEER = "volunteer"    # 广播下角色自愿揭榜并回答


class Ask(BaseModel):
    id: str
    from_role: str
    to_role: str | None = None
    mode: AskMode
    question: str
    need: Need | None = None


class AskReply(BaseModel):
    ask_id: str
    from_role: str
    kind: ReplyKind
    answer: str = ""
    reason: str = ""            # 婉拒 / 揭榜的理由
    card_id: str | None = None
    confidence: float = 0.0


# ── 能力画像（离线构建、在线更新）─────────────────────────────────────────────
class RoleStats(BaseModel):
    asks: int = 0               # 被询问次数
    answered: int = 0
    adopted: int = 0            # 回应被请求方采纳（含证据卡被复用）
    ignored: int = 0            # 回应被请求方忽略
    declined: int = 0           # 婉拒


class RoleProfile(BaseModel):
    """轻量能力摘要：覆盖域、可服务的需求类型、资产类型、关键词，以及历史响应质量。"""

    role: str
    domains: list[str] = Field(default_factory=list)       # 覆盖的业务域
    serves: list[str] = Field(default_factory=list)        # 可服务的需求类型（assets / lineage / caliber / dimension）
    asset_types: list[str] = Field(default_factory=list)   # 擅长的资产类型
    keywords: list[str] = Field(default_factory=list)      # 检索用关键词
    channels: list[str] = Field(default_factory=list)
    quality: float = 0.7                                   # 历史响应质量（EMA，[0,1]）
    stats: RoleStats = Field(default_factory=RoleStats)
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


class RoutingScore(BaseModel):
    role: str
    score: float
    parts: dict[str, float] = Field(default_factory=dict)  # kind / asset_type / domain / terms / focus / quality_factor


class RouteDecision(BaseModel):
    mode: AskMode
    target: str | None = None
    confidence: float = 0.0
    scores: list[RoutingScore] = Field(default_factory=list)
    reason: str = ""


class Bid(BaseModel):
    role: str
    volunteered: bool
    fit: float = 0.0
    reason: str = ""


# ── 协作轨迹（可审计 / 前端展示）─────────────────────────────────────────────
class ReuseEvent(BaseModel):
    need_key: str
    card_id: str
    author: str
    by: str
    waited: bool = False        # True：对方正在探查，等到其证据卡后复用


class AskRecord(BaseModel):
    """requester / served_by / target_agent 是 agent id（同一角色多个副本时形如 `bi_asset_discovery@GMV`，单副本时等于角色名）。"""

    ask_id: str
    ask: Ask | None = None
    requester: str
    requester_role: str = ""
    target_agent: str | None = None    # 路由选中角色后，在该角色的副本里挑中的那个（定向询问）
    served_role: str | None = None
    need_key: str
    route: RouteDecision
    replies: list[AskReply] = Field(default_factory=list)
    bids: list[Bid] = Field(default_factory=list)
    served_by: str | None = None
    card_id: str | None = None
    outcome: Outcome = "unanswered"


class ProfileUpdate(BaseModel):
    role: str
    outcome: Literal["adopted", "ignored", "declined"]
    quality_before: float
    quality_after: float


class AgentInfo(BaseModel):
    """一个 agent 副本（工作流图里的一个节点）。同一角色在图里出现 n 次就有 n 个副本，各自独立运行、负载独立。"""

    agent_id: str
    role: str
    replica: int = 1
    replicas: int = 1
    targets: list[str] = Field(default_factory=list)
    status: str = ""
    produced: list[str] = Field(default_factory=list)     # 自己真正探查的需求 key
    reused: list[str] = Field(default_factory=list)       # 复用他人证据卡的需求 key
    asks_made: int = 0
    asks_served: int = 0
    declined: int = 0
    handoffs: int = 0          # 自己发起的转交次数（把发现交给别的角色继续查）
    checked: int = 0           # 选择阶段角色：检查了多少个候选
    issues: int = 0            # 选择阶段角色：发现了多少个问题（元信息缺失 / 维度没人覆盖 / 候选对不上）


class CollabEvent(BaseModel):
    """协作事件流（按发生顺序）：供前端按时间画泳道。a / b 是 agent id（工作流图的节点 id）。

    probe      a 自己探查了 need            reuse     b 复用了 a 产出的证据卡（箭头 a → b）
    ask        a 定向询问 b                  reply     b 回答了 a 的询问（箭头 b → a，sub = adopted / ignored）
    deliver    a 把候选交付给 b（b 是选择阶段的节点，note=N candidates）   check  a（选择阶段的角色）完成了自己的检查（note=结论）
    decline    a 婉拒了 b 的询问（note=理由） broadcast  a 在黑板广播（b=None；sub=downgraded 表示定向被婉拒后降级）
    volunteer  a 揭榜回答 b（sub = adopted / ignored）   unanswered a 的求助没有任何回应
    """

    seq: int
    kind: Literal["probe", "reuse", "ask", "reply", "decline", "broadcast", "volunteer", "unanswered", "deliver", "check"]
    a: str
    b: str | None = None
    need_key: str = ""
    note: str = ""
    sub: str = ""
    via: str = ""      # 非空 = 这是一次「发现驱动的转交」（值为触发它的证据卡 key）
    hop: int = 0


class LLMUsage(BaseModel):
    """④ 里 LLM 的使用情况（整次召回）：调用数受 RECALL_LLM_MAX_CALLS 总预算约束，失败 / 超预算的调用回落到规则。"""

    enabled: bool = False
    calls: int = 0
    failures: int = 0           # 调用了但失败（上游错误 / 超时 / 输出不可解析）
    fallbacks: int = 0          # 回落到规则的次数（失败 + 超预算）
    max_calls: int = 0
    budget_exhausted: bool = False
    by_purpose: dict[str, int] = Field(default_factory=dict)


class CollabTrace(BaseModel):
    reuse: list[ReuseEvent] = Field(default_factory=list)
    asks: list[AskRecord] = Field(default_factory=list)
    profile_updates: list[ProfileUpdate] = Field(default_factory=list)
    agents: list[AgentInfo] = Field(default_factory=list)   # 各副本的身份与协作统计
    events: list[CollabEvent] = Field(default_factory=list) # 按发生顺序的协作事件流（复用 / 询问 / 婉拒 / 广播 / 揭榜 …）
    retrievals: list[str] = Field(default_factory=list)    # 真正执行的探查（Need.key），重复即重复探查
    stats: dict[str, int] = Field(default_factory=dict)
    llm: LLMUsage = Field(default_factory=LLMUsage)
    supplements: list[str] = Field(default_factory=list)    # 评估后觉得不够、又继续召回的需求 key（重复出现 = 多次补召回；不计入 retrievals）

    def summarize(self) -> dict[str, int]:
        a = self.asks
        self.stats = {
            "retrievals": len(self.retrievals), "supplements": len(self.supplements), "reuse_hits": len(self.reuse), "reuse_waited": sum(1 for r in self.reuse if r.waited),
            "asks": len(a), "directed": sum(1 for x in a if x.route.mode == AskMode.DIRECTED),
            "broadcast": sum(1 for x in a if x.route.mode == AskMode.BROADCAST),
            "downgraded": sum(1 for x in a if any(r.kind == ReplyKind.DOWNGRADE for r in x.replies)),
            "declined": sum(1 for x in a for r in x.replies if r.kind == ReplyKind.DECLINE),
            "volunteered": sum(1 for x in a for r in x.replies if r.kind == ReplyKind.VOLUNTEER),
            "handoffs": sum(1 for x in a if x.ask is not None and x.ask.need is not None and x.ask.need.via),
            "adopted": sum(1 for x in a if x.outcome == "adopted"), "ignored": sum(1 for x in a if x.outcome == "ignored"),
            "unanswered": sum(1 for x in a if x.outcome == "unanswered"),
        }
        return self.stats


def new_card_id() -> str:
    import uuid
    return "card_" + uuid.uuid4().hex[:8]


__all__ = [n for n in dir() if not n.startswith("_") and n not in ("annotations", "datetime", "timezone", "Enum", "Any", "Literal", "BaseModel", "Field")]
