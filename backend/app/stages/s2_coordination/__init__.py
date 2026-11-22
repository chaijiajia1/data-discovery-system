"""② 协调：自然语言 + ① 预注入知识 → 意图（Intent）与任务契约（TaskContract）。

流程：Coordination agent 识别意图（domain / sub_domain / intent_type …）并抽取用户明确表达的条件 →
合并用户补充信息（可选） → 提取不到 metric 时用 intent.keywords 兜底（不追问用户） →
其余缺失条件按默认规则补全并记录来源 → 输出 TaskContract（`contract.intent` 即本阶段的意图）。
"""
from __future__ import annotations

from ...schemas.task import SLOT_NAMES, Slot, TaskContract, ValueSource
from ..base import RunContext, Stage
from .agent import CoordinationAgent
from .defaults import DEFAULT_RULES
from .fallback import METRICS_FROM_KEYWORDS_NOTE, METRICS_FROM_KEYWORDS_RULE_ID, metrics_from_keywords


class CoordinationStage(Stage[TaskContract]):
    name = "stage2_coordination"

    async def run(self, ctx: RunContext) -> TaskContract:
        pre = ctx.preinjection
        out = await CoordinationAgent(ctx.deps.llm).run(
            ctx.query, pre, session_id=ctx.session_id, clarifications=ctx.clarifications,
        )
        draft = dict(out.slots)
        # 用户补充信息（可选）优先级最高
        for k, v in ctx.clarifications.items():
            draft[k] = v
        # dim_terms 是「具名维度取值」（如 自然堂），不是维度名：LLM 把已识别的维度名（渠道）也填进来，会触发无效的维值解析，
        # 还会让 ⑤ 把所有没有字段元信息的候选判成「不支持按该取值过滤」
        dim_names = {str(d).strip().casefold() for d in (draft.get("dimensions") or [])}
        out.intent.dim_terms = [t for t in out.intent.dim_terms if t.strip().casefold() not in dim_names]
        # 提取不到 metric：不追问，用 keywords 兜底（先于默认规则，使 caliber 也能按这些词匹配知识）
        keyword_metrics = [] if draft.get("metrics") else metrics_from_keywords(out.intent, draft)
        if keyword_metrics:
            draft["metrics"] = keyword_metrics

        contract = TaskContract(query=ctx.query, intent=out.intent)
        for slot in SLOT_NAMES:
            value = draft.get(slot)
            if slot == "metrics" and keyword_metrics:
                contract.metrics = Slot(value=value, source=ValueSource.INFERRED,
                                        rule_id=METRICS_FROM_KEYWORDS_RULE_ID, note=METRICS_FROM_KEYWORDS_NOTE)
            elif value not in (None, [], ""):
                # 来自用户原话或用户补充 → USER（LLM 抽取同理，仅转写用户表达）
                setattr(contract, slot, Slot(value=value, source=ValueSource.USER))
            elif slot in DEFAULT_RULES:
                rule = DEFAULT_RULES[slot]
                v = rule.value(pre, draft)
                src = ValueSource.KNOWLEDGE if (slot == "caliber" and v) else ValueSource.DEFAULT_RULE
                setattr(contract, slot, Slot(value=v if v is not None else "standard", source=src, rule_id=rule.id, note=rule.note))
        return contract
