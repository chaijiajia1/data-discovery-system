"""角色抽象：并行召回阶段的探查单元。"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from ..collab.hub import CollabHub
from ..collab.blackboard import Blackboard
from ..collab.router import score_profile
from ..deps import Deps
from ..schemas.knowledge import KnowledgeItem
from ..schemas.collab import Ask, AskMode, AskReply, Bid, EvidenceCard, Need, ReplyKind
from ..schemas.recall import RoleReport
from ..schemas.task import TaskContract
from ..schemas.workflow import WorkflowNode


ALL_DOMAINS = ("transaction", "community", "commercialization", "resource_cost", "overseas")


@dataclass(frozen=True)
class RoleSpec:
    """角色模板。`channels` 是该角色查询的检索通道（覆盖单元 = 口径 × 通道：一个角色可以只查一个通道，
    也可以一个 Agent 查多个通道）；`dims` 是它能为选择阶段提供的评估维度。"""

    name: str
    description: str
    phase: str = "recall"                                        # recall | select（目前所有内置角色都是 recall：评估维度由召回角色内联提供）
    channels: frozenset[str] = frozenset()                       # 查询的通道：bi | graph | knowledge | dim_value
    dims: frozenset[str] = frozenset()                           # 能提供的选择评估维度
    max_concurrency: int = 3                                     # 该角色同层并发上限；也是同时响应他人询问的上限（超出即婉拒 busy）
    # ── 能力画像种子（离线）：注册表以此初始化，在线按响应被采纳 / 忽略 / 婉拒持续更新 ──
    serves: frozenset[str] = frozenset()                         # 可服务的需求类型：assets / lineage / caliber / dimension
    asset_types: frozenset[str] = frozenset()                    # 擅长的资产类型：dataset / chart / dashboard / table / knowledge …
    domains: frozenset[str] = frozenset(ALL_DOMAINS)             # 覆盖的业务域
    keywords: tuple[str, ...] = ()                               # 画像检索关键词
    template: str = ""                                           # 角色模板（prompts/ 下的名字）；空 = 召回角色默认 roles/<name>，选择角色没有模板

    @property
    def template_name(self) -> str | None:
        return self.template or (f"roles/{self.name}" if self.phase == "recall" else None)


@dataclass
class RoleContext:
    node: WorkflowNode
    contract: TaskContract
    knowledge: list[KnowledgeItem]
    hub: CollabHub
    upstream: dict[str, RoleReport] = field(default_factory=dict)   # 依赖节点的产出（L0/L1 已知资产以 "seed" 键注入，所有节点都能看到）
    mode: str = "recall_and_select"                                  # ③ 的 mode：select_only（L1）时不再召回，只核验已知资产

    @property
    def board(self) -> Blackboard:
        return self.hub.board

    @property
    def deps(self) -> Deps:
        return self.hub.env.deps

    @property
    def cookie(self) -> str | None:
        return self.hub.env.cookie

    @property
    def recall_params(self) -> dict[str, Any]:
        return self.hub.env.recall_params


def new_evidence_id() -> str:
    return "ev_" + uuid.uuid4().hex[:8]


class Role:
    spec: RoleSpec

    def __init__(self, agent_id: str | None = None, targets: list[str] | None = None, replica: int = 1, replicas: int = 1) -> None:
        """一个 Role 实例 = 一个 agent 副本。同一角色在工作流图里有 n 个节点就有 n 个独立实例，负载 / 身份互不相干。"""
        self.agent_id = agent_id or self.spec.name
        self.targets = list(targets or [])
        self.replica, self.replicas = replica, replicas
        self._active = 0            # 本副本正在响应他人询问的数量；达到 spec.max_concurrency 即婉拒（busy）

    async def run(self, rc: RoleContext) -> RoleReport:  # pragma: no cover - interface
        raise NotImplementedError

    async def produce(self, need: Need, hub: CollabHub, who: str) -> EvidenceCard | None:
        """按需求类型自己探查并发布证据卡。不服务的类型返回 None。"""
        return None

    # ── 被询问 / 广播：角色保有自主权 ──
    def _fit(self, need: Need, hub: CollabHub) -> float:
        p = hub.registry.get(self.spec.name)
        return score_profile(need, p).score if p is not None else 0.0

    def _refusal(self, need: Need, hub: CollabHub, *, skip_low_fit: bool = False) -> str | None:
        """不该接这个活的理由（带说明）；可以接返回 None。硬规则（超出能力 / 业务域不覆盖 / 忙）始终确定性判断；
        `skip_low_fit`：画像匹配度这一条留给 LLM 做语义判断（LLM 不可用时调用方再回落到它）。"""
        sp = self.spec
        if need.kind not in sp.serves:
            return f"out_of_scope：{sp.name} 不提供「{need.kind}」类探查（只服务 {sorted(sp.serves) or '无'}）"
        if need.domains and not set(need.domains) <= set(sp.domains):
            return f"domain_uncovered：{sp.name} 不覆盖业务域 {sorted(set(need.domains) - set(sp.domains))}"
        if not skip_low_fit:
            low = self._low_fit(need, hub)
            if low is not None:
                return low
        if self._active >= sp.max_concurrency:
            return f"busy：同时响应的询问已达上限 {sp.max_concurrency}"
        return None

    def _low_fit(self, need: Need, hub: CollabHub) -> str | None:
        fit = self._fit(need, hub)
        if fit < hub.settings.COLLAB_ACCEPT_MIN:
            return f"low_fit：画像匹配度 {fit:.2f} 低于 {hub.settings.COLLAB_ACCEPT_MIN:g}"
        return None

    async def _semantic_eval(self, need: Need, hub: CollabHub, requester: str = ""):
        from .llm_agent import self_evaluate           # 延迟导入：llm_agent 依赖本模块的 RoleSpec
        return await self_evaluate(hub, self.agent_id, self.spec, need, fit_rule=self._fit(need, hub), active=self._active, requester=requester)

    async def bid(self, need: Need, hub: CollabHub, requester: str = "") -> Bid:
        """黑板广播下的自评估：要不要揭榜。LLM 可用时：硬规则 → 画像粗筛（太低直接不揭榜，不花 LLM）→ LLM 判断语义上是否胜任；
        LLM 不可用 / 失败回落到规则（匹配度 ≥ 揭榜线）。"""
        fit = self._fit(need, hub)
        if hub.llm.enabled and not need.via:               # 转交来的需求已带 via / hint_assets 结构化上下文：只用规则判断，不花 LLM
            hard = self._refusal(need, hub, skip_low_fit=True)
            if hard is not None:
                return Bid(role=self.agent_id, volunteered=False, fit=fit, reason=hard)
            if fit < hub.settings.COLLAB_ACCEPT_MIN:
                return Bid(role=self.agent_id, volunteered=False, fit=fit, reason=f"low_fit：画像匹配度 {fit:.2f} 低于 {hub.settings.COLLAB_ACCEPT_MIN:g}（粗筛，未调用 LLM）")
            ev = await self._semantic_eval(need, hub, requester)
            if ev is not None:
                return Bid(role=self.agent_id, volunteered=ev.accept, fit=ev.fit,
                           reason=f"llm_accept：{ev.reason}" if ev.accept else f"llm_decline：{ev.reason}")
        reason = self._refusal(need, hub)
        if reason is None and fit < hub.settings.COLLAB_VOLUNTEER_MIN:
            reason = f"low_fit：匹配度 {fit:.2f} 低于揭榜线 {hub.settings.COLLAB_VOLUNTEER_MIN:g}"
        return Bid(role=self.agent_id, volunteered=reason is None, fit=fit, reason=reason or "自评估通过，揭榜")

    async def _decide(self, ask: Ask, need: Need, hub: CollabHub) -> str | None:
        """被点名后接不接：硬规则确定性判断；定向询问且 LLM 可用时，语义上是否胜任交给 LLM（婉拒必须说明理由），LLM 不可用回落匹配度规则。
        广播揭榜的角色已在 bid 里自评过，这里不再重复调用 LLM；转交来的需求（`need.via`）只用规则判断，不花 LLM。"""
        if ask.mode != AskMode.DIRECTED or not hub.llm.enabled or need.via:
            return self._refusal(need, hub)
        hard = self._refusal(need, hub, skip_low_fit=True)
        if hard is not None:
            return hard
        ev = await self._semantic_eval(need, hub, ask.from_role)
        if ev is None:
            return self._low_fit(need, hub)
        return None if ev.accept else f"llm_decline：{ev.reason}"

    async def respond(self, ask: Ask, hub: CollabHub) -> AskReply:
        """被点名 / 揭榜后回应：可以婉拒并说明理由；接下来用 obtain（复用优先、避免重复探查）完成探查。"""
        name, need = self.agent_id, ask.need
        reason = await self._decide(ask, need, hub) if need is not None else "out_of_scope：询问没有结构化需求"
        if reason is not None:
            return AskReply(ask_id=ask.id, from_role=name, kind=ReplyKind.DECLINE, reason=reason)
        assert need is not None
        who = f"ask:{ask.id}"
        self._active += 1
        try:
            res = await hub.obtain(need, requester=name, who=who, produce=lambda: self.produce(need, hub, who))
        except Exception as e:  # noqa: BLE001 - 探查失败：婉拒并说明
            return AskReply(ask_id=ask.id, from_role=name, kind=ReplyKind.DECLINE, reason=f"failed：探查失败 {type(e).__name__}: {e}")
        finally:
            self._active -= 1
        if res.card is None:
            return AskReply(ask_id=ask.id, from_role=name, kind=ReplyKind.DECLINE, reason="failed：没有产出证据卡")
        return AskReply(ask_id=ask.id, from_role=name, kind=ReplyKind.ANSWER, answer=res.card.summary, card_id=res.card.id, confidence=res.card.confidence)

    def _report(self, rc: RoleContext, **kw: Any) -> RoleReport:
        return RoleReport(node_id=rc.node.id, role=self.spec.name, **kw)
