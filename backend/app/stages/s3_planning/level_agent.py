"""入口定级 Agent：LLM 只负责判**每个子目标**的覆盖状态（covered / doubtful / uncovered）。

任务级别（L0 / L1 / L2 = 取最差者）仍由 `levels.assess_level` 确定性汇总。铁律 3：LLM 未配置 / 调用失败 / 输出不可解析 → 返回 None，
调用方回落到规则版；单个子目标的条目非法（未知名字、非法状态、编造资产 id）→ 只有该目标保留规则结果。
铁律 4：LLM 的判定记 rule_id=`llm.goal.<status>`，reason 为 LLM 给出的理由。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from ...errors import UpstreamError
from ...llm_client import LLMClient
from ...prompts import load_prompt
from ...schemas.knowledge import KnowledgeItem, PreInjectionResult
from ...schemas.task import TaskContract, ValueSource
from ...schemas.workflow import GoalStatus, SubGoal
from ..s2_coordination.agent import parse_llm_json
from .levels import required_asset_types

logger = logging.getLogger(__name__)
_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")
SESSION_RULE_ID = "goal.covered.session_confirmed"


def _knowledge_text(items: list[KnowledgeItem]) -> str:
    return "\n".join(
        f"- id={i.id} [{i.kind.value}] {i.title}: {i.content}"
        f"（资产={i.asset_ids or '—'}；一致性={i.consistency}）" for i in items
    ) or "（无）"


def _goals_text(contract: TaskContract) -> str:
    from .levels import metric_names
    lines = [f"- metric: {m}" for m in metric_names(contract)]
    lines += [f"- dimension: {d}" for d in (contract.dimensions.value or [])]
    return "\n".join(lines)


def _conflicts_text(pre: PreInjectionResult | None) -> str:
    return "\n".join(f"- {d.title}（{d.detail}）" for d in (pre.dropped if pre else []) if d.reason == "conflict") or "（无）"


def build_prompt(contract: TaskContract, pre: PreInjectionResult | None, *, confirmed: dict[str, list[str]], require_bi: bool = False) -> str:
    """单遍占位符替换：被插入的内容不会再被当作模板解析，杜绝占位符注入。"""
    caliber = contract.caliber.value if contract.caliber.source == ValueSource.USER and contract.caliber.value else "（未指定）"
    values = {
        "query": contract.query,
        "intent_type": contract.intent.intent_type if contract.intent else "other",
        "caliber": str(caliber),
        "required_types": "BI 资产（bi_dataset）" if required_asset_types(contract, require_bi=require_bi) else "任意数据源资产（BI 资产或明细表均可）",
        "goals": _goals_text(contract),
        "confirmed": json.dumps(confirmed, ensure_ascii=False) if confirmed else "（无）",
        "knowledge": _knowledge_text(list(pre.injected) if pre else []),
        "conflicts": _conflicts_text(pre),
    }
    return _PLACEHOLDER.sub(lambda m: values.get(m.group(1), m.group(0)), load_prompt("level"))


def _valid_assets(raw: Any, allowed: set[str]) -> list[str]:
    ids = [str(a) for a in raw] if isinstance(raw, list) else []
    return list(dict.fromkeys(a for a in ids if a in allowed))


class LevelAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def regrade(self, contract: TaskContract, pre: PreInjectionResult | None, base: list[SubGoal], *,
                      confirmed: dict[str, list[str]], require_bi: bool = False) -> list[SubGoal] | None:
        """返回与 `base` 同序同名的子目标列表；LLM 整体不可用时返回 None。"""
        try:
            data = parse_llm_json(await self.llm.complete(prompt=build_prompt(contract, pre, confirmed=confirmed, require_bi=require_bi), json_mode=True))
        except (UpstreamError, ValueError) as e:
            logger.info("level LLM skipped: %s", e)
            return None
        raw = data.get("goals")
        if not isinstance(raw, list):
            return None
        allowed = {a for it in (pre.injected if pre else []) for a in it.asset_ids}
        by_key = {(str(x.get("kind")), str(x.get("name"))): x for x in raw if isinstance(x, dict)}
        out: list[SubGoal] = []
        for g in base:
            item = by_key.get((g.kind, g.name))
            out.append(self._merge(g, item, allowed))
        return out

    @staticmethod
    def _merge(g: SubGoal, item: dict[str, Any] | None, allowed: set[str]) -> SubGoal:
        if g.rule_id == SESSION_RULE_ID or item is None:        # 会话内已确认是用户事实，不由 LLM 改写
            return g
        try:
            status = GoalStatus(str(item.get("status")))
        except ValueError:
            return g
        assets = _valid_assets(item.get("asset_ids"), allowed)
        if g.kind == "metric" and status != GoalStatus.UNCOVERED and not assets:
            return g                                             # covered / doubtful 必须指向知识里真实存在的资产，否则不可信
        reason = str(item.get("reason") or "").strip() or f"LLM 判定为 {status.value}"
        return SubGoal(kind=g.kind, name=g.name, status=status, rule_id=f"llm.goal.{status.value}", reason=reason, asset_ids=assets)
