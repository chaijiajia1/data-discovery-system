"""协作中枢：并行角色的「需求 → 证据」协议。

**agent 与角色**：工作流图里的每个节点是一个独立的 agent 副本（`agent_id` = 节点 id）；同一角色出现 n 次就有 n 个副本，
各自独立运行、负载独立。能力画像是**角色级**的（同角色副本共享，采纳 / 忽略 / 婉拒记在角色上）；路由先按画像选**角色**，再在该角色的副本里
挑一个：持有该需求（已认领 / 已产出）的优先，其次负载低的，最后按 id。副本之间通过黑板（证据卡 / 认领）、定向询问与广播沟通。

角色产生需求（Need）后，按固定顺序解决，越靠前越省：
  1. 复用优先   先读黑板：已有可复用的证据卡就直接用——不询问、不广播、不重复探查；
  2. 认领等待   同一需求别人正在探查（已认领）→ 等它的证据卡再复用，而不是各自重复探查（超时则自己探查）；
  3. 自己探查   角色有能力且给了 produce → 认领后探查，产出证据卡发布到黑板；
  4. 双通道求助 角色没有该能力（produce=None）→ 路由打分（见 router.py）：
        高置信命中某角色画像 → 定向询问该角色（省去全量广播与无关自评估）；
        低置信 / 跨域 / 画像未覆盖 → 黑板广播，各角色自评估、自愿揭榜，匹配度最高的揭榜者回答；
        被点名的角色保有自主权，可婉拒并说明理由（out_of_scope / domain_uncovered / low_fit / busy）；
        请求方收到婉拒后降级为广播（婉拒者不再被重复询问）。
  5. 采纳反馈   请求方判断回应是否可用（采纳 / 忽略），连同婉拒一起写回角色画像（历史响应质量）。
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from ..schemas.collab import (
    AgentInfo, Ask, AskMode, AskRecord, AskReply, CollabEvent, CollabTrace, EvidenceCard, Need, ReplyKind, ReuseEvent,
)
from ..schemas.recall import BoardEntryKind
from . import handoff as handoff_mod
from . import router
from .blackboard import Blackboard
from .llm import RecallLLM
from .registry import ProfileRegistry

if TYPE_CHECKING:  # pragma: no cover
    from ..deps import Deps
    from ..roles.base import Role
    from ..schemas.knowledge import KnowledgeItem
    from ..schemas.task import TaskContract

logger = logging.getLogger(__name__)
REUSE_ADOPT_WEIGHT = 0.5      # 证据卡被复用：算作被采纳，但权重低于一次明确的询问回应


@dataclass
class RoleEnv:
    """角色探查时共用的环境（检索通道 / 契约 / 预注入知识 / cookie / 检索参数）。"""

    deps: "Deps"
    contract: "TaskContract"
    knowledge: list["KnowledgeItem"] = field(default_factory=list)
    cookie: str | None = None
    recall_params: dict[str, Any] = field(default_factory=dict)
    known_assets: list[str] = field(default_factory=list)     # L0/L1 子目标已知资产（预注入 / 会话确认）：血缘通道的种子来源之一

    @property
    def domains(self) -> list[str]:
        d = self.contract.intent.domain if self.contract.intent else None
        return [d] if d else []


@dataclass
class Resolution:
    card: EvidenceCard | None
    how: str                                # reuse | reuse_wait | own | ask | none
    record: AskRecord | None = None


class CollabHub:
    def __init__(self, board: Blackboard, registry: ProfileRegistry, agents: dict[str, "Role"], env: RoleEnv):
        self.board, self.registry, self.agents, self.env = board, registry, agents, env
        self.settings = env.deps.settings
        self.trace = CollabTrace()
        self._ask_seq = 0
        self._handed: set[str] = set()                                   # 已处理过的转交需求（need key 去重）
        self._handoff_budget = self.settings.COLLAB_HANDOFF_MAX_TOTAL
        self.llm = RecallLLM(env.deps.llm, self.settings)               # 整次召回共用：开关 / 总预算 / 并发上限 / 统计
        self.limits: dict[str, int] = {}                                 # need key → 当前召回条数（LLM 决定；补召回后更新）
        self.needs: dict[str, Need] = {}                                 # need key → 角色探查时用的需求（含 LLM 改写的 query / limit），补召回时沿用
        self.judged: dict[str, set[str]] = {}                            # need key → 已被 LLM 评估过的资产 id（补召回只评估新增的）
        self.angles: dict[str, list[str]] = {}                           # need key → 换角度重试可用的其他说法（由 LLM 在改写 query 时一并给出）
        self.info: dict[str, AgentInfo] = {
            a: AgentInfo(agent_id=a, role=r.spec.name, replica=r.replica, replicas=r.replicas, targets=list(r.targets)) for a, r in agents.items()
        }

    def _emit(self, kind: str, a: str, b: str | None = None, *, need: Need | str = "", note: str = "", sub: str = "") -> None:
        key = need.key if isinstance(need, Need) else need
        via, hop = (need.via, need.hop) if isinstance(need, Need) else ("", 0)
        self.trace.events.append(CollabEvent(seq=len(self.trace.events) + 1, kind=kind, a=a, b=b, need_key=key, note=note, sub=sub, via=via, hop=hop))  # type: ignore[arg-type]

    @property
    def roles(self) -> dict[str, "Role"]:      # 兼容旧名：key 为 agent id（单副本时即角色名）
        return self.agents

    def role_of(self, agent_id: str) -> str:
        r = self.agents.get(agent_id)
        return r.spec.name if r is not None else agent_id

    @property
    def deps(self) -> "Deps":
        return self.env.deps

    # ── 画像更新 ──
    def _record(self, agent_id: str, outcome: str, weight: float = 1.0) -> None:
        """画像是角色级的：同角色的副本共享一份画像，结果记在角色上。"""
        role = self.role_of(agent_id)
        if self.registry.get(role) is not None:
            self.trace.profile_updates.append(self.registry.record(role, outcome, weight=weight))

    # ── 解决需求 ──
    async def obtain(self, need: Need, requester: str, who: str,
                     produce: Callable[[], Awaitable[EvidenceCard | None]] | None = None) -> Resolution:
        """requester：请求方角色名（画像 / 轨迹用）；who：认领令牌（节点 id 或 `ask:<id>`，同一角色的不同协程互不相同）。"""
        card = self.board.lookup(need, require_content=produce is None)
        if card is not None:
            return self._reuse(card, requester, waited=False)
        owner = self.board.claimed_by(need.key)
        if owner is not None and owner != who:
            card = await self.board.wait_card(need.key, self.settings.COLLAB_WAIT_SEC)
            if card is not None and (produce is not None or not card.empty):
                return self._reuse(card, requester, waited=True)
        if produce is not None:
            return await self._produce(need, requester, who, produce)
        return await self._ask(need, requester, who)

    def _reuse(self, card: EvidenceCard, requester: str, *, waited: bool) -> Resolution:
        self.board.mark_reused(card)
        self.trace.reuse.append(ReuseEvent(need_key=card.key, card_id=card.id, author=card.author, by=requester, waited=waited))
        if card.author != requester:                       # 复用自己产出的证据卡不是协作，不进事件流（泳道上不画）
            self._emit("reuse", card.author, requester, need=card.key, note=f"reuse {card.key}", sub="waited" if waited else "")
        if requester in self.info and card.key not in self.info[requester].reused:
            self.info[requester].reused.append(card.key)
        if card.author != requester:
            self._record(card.author, "adopted", REUSE_ADOPT_WEIGHT)
        return Resolution(card, "reuse_wait" if waited else "reuse")

    async def _produce(self, need: Need, requester: str, who: str, produce: Callable[[], Awaitable[EvidenceCard | None]]) -> Resolution:
        claimed = await self.board.try_claim(who, need.key)        # 认领失败 = 对方超时仍未产出：不再等，自己探查
        try:
            card = await produce()
        except BaseException:
            if claimed:
                self.board.release(need.key)
            raise
        if card is None and claimed:
            self.board.release(need.key)
        self.trace.retrievals.append(need.key)
        if requester in self.info and need.key not in self.info[requester].produced:
            self.info[requester].produced.append(need.key)
        self._emit("probe", requester, need=need, note=f"probe {need.key}")
        if card is not None and not card.empty:
            await self._handoff(card, need, requester)                 # 发现驱动的转交：把找到的东西交给其他角色继续查
        else:
            await self.retry_other_angle(need, requester)              # 空卡：换个角度再查（走转交机制，受深度 / 预算限制）
        return Resolution(card, "own")

    async def supplement(self, key: str, requester: str, limit: int, query: str | None,
                         produce: Callable[[Need], Awaitable[EvidenceCard | None]]) -> EvidenceCard | None:
        """评估后觉得不够，同一个需求按更大的条数（可换 query）再召回一次；新结果替换黑板上的卡（资产并入候选池）。
        失败 / 没有新增资产时保留原来的卡，返回 None。"""
        base = self.needs.get(key)
        old = self.board.cards.get(key)
        if base is None or old is None:
            return None
        need = base.model_copy(update={"limit": limit, **({"query": query} if query else {})})
        try:
            card = await produce(need)
        except Exception as e:  # noqa: BLE001 - 补召回失败不影响已有结果
            logger.info("supplement %s failed: %s", key, e)
            return None
        if card is None or len(set(card.asset_ids) - set(old.asset_ids)) == 0:
            return None
        self.limits[key] = limit
        self.needs[key] = need
        self.trace.supplements.append(key)
        self._emit("probe", requester, need=need, note=f"supplement {key} limit={limit}", sub="supplement")
        return card

    async def retry_other_angle(self, need: Need, requester: str) -> int:
        """结果为空 / 被判定全部不相关时，换个说法再查：把 LLM 给出的其他说法当作新的需求交给转交机制
        （换了说法 = 换了 key，不会被去重；深度、总预算、开关都沿用转交的限制）。没有备选说法（LLM 不可用）时什么也不做。"""
        alts = [a for a in self.angles.get(need.key, []) if a and a.casefold() != need.subject.casefold()]
        if not alts or need.hop >= self.settings.COLLAB_HANDOFF_MAX_DEPTH:
            return 0
        fan = self.settings.COLLAB_HANDOFF_FANOUT
        return await self.delegate(requester, [
            need.model_copy(update={"subject": a, "query": a, "question": "", "via": need.key, "hop": need.hop + 1, "hint_assets": []}) for a in alts[:fan]
        ])

    # ── 发现驱动的转交 ──
    async def _handoff(self, card: EvidenceCard, need: Need, author: str) -> None:
        """角色找到东西后主动交给其他角色继续查（见 handoff.py）；深度超限则不再转交。"""
        if need.hop >= self.settings.COLLAB_HANDOFF_MAX_DEPTH:
            return
        await self.delegate(author, handoff_mod.follow_ups(self, card, need))

    async def delegate(self, author: str, needs: list[Need]) -> int:
        """把一批后续需求交给其他角色（选择阶段角色发现问题时也用它）。自己能做 → 自己顺着查（不询问同类角色）；
        自己做不了但有别的 agent 服务该类需求 → 询问它（定向 / 广播由路由决定）；谁都做不了 → 跳过（不制造无人应答的噪音）。每个需求只处理一次，受总预算限制。返回实际发出的数量。"""
        if not self.settings.COLLAB_HANDOFF_ENABLED:
            return 0
        jobs = []
        for n in needs:
            if self._handoff_budget <= 0:
                break
            if n.key in self._handed or n.key in self.board.cards:
                continue
            me = self.agents.get(author)
            me_can = me is not None and n.kind in me.spec.serves
            # 自己就能做的线索自己顺着查（黑板的认领 / 复用保证不重复探查）；只有自己做不了的才交给别人——询问同类角色纯属多余开销
            others = [] if me_can else [a for a, r in self.agents.items() if a != author and n.kind in r.spec.serves]
            if not others and not me_can:
                continue
            self._handed.add(n.key)
            self._handoff_budget -= 1
            if author in self.info and others:
                self.info[author].handoffs += 1
            jobs.append(self._do_handoff(n, author, bool(others)))
        if jobs:
            for r in await asyncio.gather(*jobs, return_exceptions=True):
                if isinstance(r, BaseException):
                    logger.warning("handoff failed: %s", r)
        return len(jobs)

    # ── 选择阶段角色的通信 ──
    def deliver(self, src: str, dst: str, n: int) -> None:
        """上游节点把候选交付给选择阶段的节点（工作流图里的依赖边）。"""
        self._emit("deliver", src, dst, note=f"{n} candidates")

    def report_check(self, agent: str, checked: int, issues: int, note: str) -> None:
        if agent in self.info:
            self.info[agent].checked += checked
            self.info[agent].issues += issues
        self._emit("check", agent, note=note)

    async def _do_handoff(self, n: Need, author: str, to_others: bool) -> None:
        who = f"{author}#handoff:{n.key}"
        if to_others:
            await self._ask(n, author, who)
        else:
            await self.obtain(n, requester=author, who=who, produce=lambda: self.agents[author].produce(n, self, who))

    # ── 双通道求助 ──
    def _holds(self, agent_id: str, need: Need) -> bool:
        """该副本是否「持有」这个需求：认领着它，或证据卡就是它产出的，或它的目标口径就是这个需求的主题。"""
        card = self.board.cards.get(need.key)
        return self.board.claimed_by(need.key) == agent_id or (card is not None and card.author == agent_id) or need.subject in self.agents[agent_id].targets

    def _priority(self, agent_id: str, need: Need) -> tuple:
        """同一角色的多个副本之间的取舍：持有该需求的优先 → 负载低的优先 → 按 id（稳定）。"""
        return (0 if self._holds(agent_id, need) else 1, self.agents[agent_id]._active, agent_id)

    def _pick_agent(self, role: str, need: Need, exclude: set[str]) -> str | None:
        cands = [a for a, r in self.agents.items() if r.spec.name == role and a not in exclude]
        return min(cands, key=lambda a: self._priority(a, need)) if cands else None

    async def _ask(self, need: Need, requester: str, who: str) -> Resolution:
        self._ask_seq += 1
        me_role = self.role_of(requester)
        present = [a for a in self.agents if a != requester and self.registry.get(self.role_of(a)) is not None]
        roles = list(dict.fromkeys(self.role_of(a) for a in present))          # 路由按角色画像打分；同一角色的副本共用一份画像
        decision = router.decide(need, [self.registry.get(r) for r in roles], direct_threshold=self.settings.COLLAB_DIRECT_THRESHOLD)
        target = self._pick_agent(decision.target, need, {requester}) if decision.mode == AskMode.DIRECTED and decision.target else None
        directed = target is not None
        if decision.mode == AskMode.DIRECTED and not directed:
            decision = decision.model_copy(update={"mode": AskMode.BROADCAST, "target": None, "reason": "uncovered：目标角色没有可用副本"})
        if present and not need.question and not need.via:                      # 路由已定（确定性）；角色自己的需求再写成自足的问题（转交来的需求已带 via / hint_assets 结构化上下文，不再花 LLM）
            from ..roles.llm_agent import compose_question                      # 延迟导入：roles 依赖本模块
            q = await compose_question(self, me_role, need)
            if q:
                need = need.model_copy(update={"question": q})
        ask = Ask(id=f"ask{self._ask_seq}", from_role=requester, to_role=target, mode=AskMode.DIRECTED if directed else AskMode.BROADCAST,
                  question=need.text, need=need)
        rec = AskRecord(ask_id=ask.id, requester=requester, requester_role=me_role, target_agent=target, need_key=need.key, route=decision, ask=ask)
        self.trace.asks.append(rec)
        if requester in self.info:
            self.info[requester].asks_made += 1
        if directed:
            self._emit("ask", requester, target, need=need, note=need.text, sub=decision.reason)
        served: tuple[str, AskReply] | None = None
        declined: set[str] = set()

        if directed:
            reply = await self.agents[target].respond(ask, self)                 # type: ignore[index]
            rec.replies.append(reply)
            if reply.kind == ReplyKind.ANSWER:
                served = (target, reply)                                          # type: ignore[assignment]
            else:                                                                  # 婉拒 → 降级为广播（同角色的其他副本仍可揭榜）
                declined.add(target)                                               # type: ignore[arg-type]
                self.info[target].declined += 1                                    # type: ignore[index]
                self._record(target, "declined")                                   # type: ignore[arg-type]
                self._emit("decline", target, requester, need=need, note=reply.reason)       # type: ignore[arg-type]
                rec.replies.append(AskReply(ask_id=ask.id, from_role=requester, kind=ReplyKind.DOWNGRADE,
                                            answer="定向询问被婉拒，降级为黑板广播", reason=reply.reason))
        if served is None:
            served = await self._broadcast(ask, rec, present, declined)

        card = None
        if served is not None:
            agent, reply = served
            rec.served_by, rec.served_role, rec.card_id = agent, self.role_of(agent), reply.card_id
            via = "reply" if reply.kind == ReplyKind.ANSWER else "volunteer"
            card = self.board.get_card(reply.card_id)
            if card is not None and not card.empty and card.confidence >= self.settings.COLLAB_ADOPT_MIN_CONF:
                rec.outcome = "adopted"
                self.info[agent].asks_served += 1
                self._record(agent, "adopted")
                self._emit(via, agent, requester, need=need, note=card.summary, sub="adopted")
            else:
                rec.outcome, card = "ignored", None                                # 回应不可用：忽略，并计入响应质量
                if not need.via:                                                    # 转交里的空结果是「这条线索没东西」，不是回应质量差：不惩罚画像
                    self._record(agent, "ignored")
                self._emit(via, agent, requester, need=need, note="回应为空 / 置信度不足", sub="ignored")
        else:
            self._emit("unanswered", requester, need=need, note="没有角色揭榜 / 回应")
        return Resolution(card, "ask" if card is not None else "none", rec)

    async def _broadcast(self, ask: Ask, rec: AskRecord, present: list[str], declined: set[str]) -> tuple[str, AskReply] | None:
        need = ask.need
        assert need is not None
        await self.board.post(ask.from_role, BoardEntryKind.BROADCAST, f"ask:{need.key}", {"ask_id": ask.id, "question": ask.question}, 0.3)
        self._emit("broadcast", ask.from_role, need=need, note="定向被婉拒，降级为黑板广播" if declined else ask.question,
                   sub="downgraded" if declined else rec.route.reason)
        bids = list(await asyncio.gather(*(self.agents[a].bid(need, self, ask.from_role) for a in present if a not in declined)))      # 各角色并发自评（LLM 自评也是并发的）
        rec.bids.extend(bids)
        b_ask = ask.model_copy(update={"mode": AskMode.BROADCAST, "to_role": None})
        # 揭榜者按 匹配度 高 → 持有该需求 → 负载低 → id 排序；匹配度相同（如同角色的多个副本）时由后三者定先后
        for b in sorted((b for b in bids if b.volunteered), key=lambda b: (-b.fit, *self._priority(b.role, need))):
            reply = await self.agents[b.role].respond(b_ask, self)
            if reply.kind == ReplyKind.ANSWER:
                reply = reply.model_copy(update={"kind": ReplyKind.VOLUNTEER})
                rec.replies.append(reply)
                return b.role, reply
            rec.replies.append(reply)                                               # 揭榜后又婉拒（如刚好变忙）：换下一位
            self.info[b.role].declined += 1
            self._record(b.role, "declined")
        return None

    def finalize(self) -> CollabTrace:
        self.trace.agents = list(self.info.values())
        self.trace.llm = self.llm.usage()
        self.trace.summarize()
        self.registry.flush()
        return self.trace


def build_hub(agents: dict[str, "Role"], deps: "Deps", *, contract: "TaskContract", knowledge: list["KnowledgeItem"], cookie: str | None,
              recall_params: dict[str, Any], known_assets: list[str] | None = None, board: Blackboard | None = None) -> CollabHub:
    deps.profiles.ensure([r.spec for r in agents.values()])
    env = RoleEnv(deps=deps, contract=contract, knowledge=knowledge, cookie=cookie, recall_params=recall_params, known_assets=list(known_assets or []))
    return CollabHub(board or Blackboard(), deps.profiles, agents, env)
