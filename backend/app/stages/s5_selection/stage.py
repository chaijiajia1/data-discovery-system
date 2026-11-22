"""⑤ 资产选择：从 ④ 的候选池里为每个口径选出主资产，其余分入备选 / 丢弃，理由结构化可追溯。"""
from __future__ import annotations

from ...schemas.knowledge import KnowledgeItem, KnowledgeKind, KnowledgeOrigin
from ...schemas.selection import SelectionResult
from ...structured_log import log_event
from ..base import RunContext, Stage
from .agent import SelectionAgent
from .rules import DimRef, SelectionInput, rule_select


def build_input(ctx: RunContext) -> SelectionInput:
    """⑤ 的输入完全来自上游契约：任务契约 / ④ 候选池与证据 / ① 注入的知识 / 权限服务。"""
    rec = ctx.recall
    knowledge = [k for k in (ctx.preinjection.injected if ctx.preinjection else []) if k.kind != KnowledgeKind.PAIR]
    refs: list[DimRef] = []
    for card in rec.cards:
        for ent in card.entities:                    # 知识通道带出的关联实体：指标 / 维度也参与术语桥接
            knowledge.append(KnowledgeItem(id=f"entity:{ent.kind}:{ent.name}", kind=KnowledgeKind.DIMENSION if ent.kind == "dimension" else KnowledgeKind.METRIC_DEF,
                                           title=ent.name, content=ent.name, origin=KnowledgeOrigin.RETRIEVAL, asset_ids=ent.asset_ids))
        if card.kind == "dimension":                 # 维值解析的结果：具名取值 → 字段
            refs += [DimRef(term=card.subject, field=ent.name) for ent in card.entities if ent.kind == "dimension"]
    return SelectionInput(contract=ctx.contract, candidates=list(rec.candidates), knowledge=knowledge, dim_refs=refs,
                          evidences=list(rec.evidences), conflicts=list(rec.conflicts))


class SelectionStage(Stage[SelectionResult]):
    name = "stage5_selection"

    async def run(self, ctx: RunContext) -> SelectionResult:
        assert ctx.contract is not None and ctx.recall is not None, "selection requires contract & recall"
        inp = build_input(ctx)
        for c in inp.candidates:                     # 权限：无权限不丢弃，降级到备选（用户可申请）
            ok, reason = await ctx.deps.permission.check(ctx.user_id, c)
            if not ok:
                inp.perms[c.asset_id] = (False, reason)
        res = await SelectionAgent(ctx.deps.llm).run(inp)
        log_event("s5.selected", primary=[d.asset_id for d in res.primary], alternates=len(res.alternates), discarded=len(res.discarded), mode=res.mode)
        return res
