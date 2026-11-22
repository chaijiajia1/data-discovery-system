"""① 知识预注入。

两路召回 → 合并去重 → 带交互效用的贪心选择 → 输出注入知识。
  路 1：context-query（/v1/context/retrieve）
  路 2：历史 query 文件（query + 用到的知识 + 最终选择的 asset）
任一路（甚至两路）失败只降级为少知识/无知识继续，不中断流程；原因记录在 route_status。
"""
from __future__ import annotations

import asyncio
import json

from ...schemas.knowledge import KnowledgeItem, PairMatch, PreInjectionResult, RetrievalPack, RouteStat
from ...structured_log import log_event
from ...text_utils import extract_terms
from ..base import RunContext, Stage
from .merge import merge_routes
from .pair_source import PairKnowledgeSource, PairRoute
from .retrieval_source import RetrievalKnowledgeSource
from .selector import SelectorParams, greedy_select


def _covered(items, query_terms: list[str]) -> list[str]:
    """查询词被覆盖：显式 terms 命中，或出现在块的标题/正文中（检索块没有显式 terms）。"""
    out = []
    for t in query_terms:
        tu = t.upper()
        if any(tu in {x.upper() for x in i.terms} or tu in f"{i.title} {i.content}".upper() for i in items):
            out.append(t)
    return out


def _route_stat(items: list[KnowledgeItem]) -> RouteStat:
    kinds: dict[str, int] = {}
    for i in items:
        kinds[i.kind.value] = kinds.get(i.kind.value, 0) + 1
    return RouteStat(count=len(items), tokens=sum(i.token_cost for i in items),
                     top_relevance=max((i.relevance for i in items), default=0.0), kinds=kinds)


_REASON_KEYS = ("reason", "message", "msg", "errorMsg", "error", "detail")
MAX_REASON_CHARS, MAX_REASONS = 160, 3


def _reason_text(r: dict) -> str:
    src = r.get("source") or r.get("sourceType")
    why = next((str(r[k]) for k in _REASON_KEYS if r.get(k)), "") or (json.dumps(r, ensure_ascii=False) if src else "")
    text = f"{src}: {why}" if src and why else why or str(src or "")
    return text if len(text) <= MAX_REASON_CHARS else text[: MAX_REASON_CHARS - 1] + "…"


def describe_pack_status(pack: RetrievalPack) -> str:
    """路 1 的状态：ok / partial: [来源] · 失败原因。上游 context-query 的 failedReasons 要让界面和日志看得见，否则只知道「哪个来源挂了」不知道为什么。"""
    if not pack.partial:
        return "ok"
    base = f"partial: {pack.failed_sources}"
    reasons = [t for r in pack.failed_reasons[:MAX_REASONS] if (t := _reason_text(r))]
    return base + (" · " + " | ".join(reasons) if reasons else "")


class PreInjectionStage(Stage[PreInjectionResult]):
    name = "stage1_preinjection"

    async def run(self, ctx: RunContext) -> PreInjectionResult:
        s = ctx.deps.settings
        terms = extract_terms(ctx.query)
        pair_src = PairKnowledgeSource(ctx.deps.pair_store, s.PREINJECT_PAIR_TOPK, s.PREINJECT_PAIR_MIN_SIM)
        ret_src = RetrievalKnowledgeSource(ctx.deps.kh, s.PREINJECT_TOKEN_BUDGET, s.PREINJECT_INCONSISTENT_DISCOUNT)

        pair_r, ret_r = await asyncio.gather(
            pair_src.fetch(terms), ret_src.fetch(ctx.query, ctx.user_id, ctx.cookie), return_exceptions=True
        )
        status: dict[str, str] = {}
        pair_items: list[KnowledgeItem] = []
        retrieval_items: list[KnowledgeItem] = []
        matches: list[PairMatch] = []
        if isinstance(pair_r, BaseException):
            status["pair"] = f"failed: {pair_r}"
        else:
            route: PairRoute = pair_r
            pair_items, matches, status["pair"] = route.items, route.matches, "ok"
        if isinstance(ret_r, BaseException):
            status["retrieval"] = f"failed: {ret_r}"
        else:
            pack: RetrievalPack = ret_r
            retrieval_items = pack.items
            status["retrieval"] = describe_pack_status(pack)
        for route, st in status.items():
            if st != "ok":
                log_event("s1.route_degraded", route=route, status=st)

        candidates = merge_routes(pair_items, retrieval_items)
        params = SelectorParams(
            tau=s.PREINJECT_TAU, redundancy=s.PREINJECT_W_REDUNDANCY, conflict=s.PREINJECT_W_CONFLICT,
            synergy=s.PREINJECT_W_SYNERGY,
        )
        out = greedy_select(candidates, s.PREINJECT_TOKEN_BUDGET, params)
        covered = _covered(out.selected, terms)
        return PreInjectionResult(
            injected=out.selected, dropped=out.dropped, trace=out.trace, token_budget=s.PREINJECT_TOKEN_BUDGET,
            tau=s.PREINJECT_TAU, used_tokens=out.used_tokens, total_value=round(sum(t.delta for t in out.trace), 4), candidate_count=len(candidates), route_status=status,
            route_stats={"pair": _route_stat(pair_items), "retrieval": _route_stat(retrieval_items)}, pair_matches=matches,
            covered_terms=covered, uncovered_terms=[t for t in terms if t not in covered],
            pair_hits=sum("pair" in i.routes for i in out.selected),
            retrieval_hits=sum("retrieval" in i.routes for i in out.selected),
        )
