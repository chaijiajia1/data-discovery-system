"""失败归因：定位最小可疑步骤，决定回滚点与下一步动作。已验证的上游步骤保留，不重跑。"""
from __future__ import annotations

from ...schemas.execution import FailureAttribution, VerificationReport

MAX_STEP_RETRIES = 2


def attribute(
    *, step_order: list[str], report: VerificationReport, step_failures: dict[str, int], has_fallback_asset: bool,
) -> FailureAttribution:
    failed = [c for c in report.checks if not c.passed and c.step_id]
    # 最小可疑步骤 = 执行顺序中最靠前的失败检查所指步骤（上游错误会污染下游，先修上游）
    failed.sort(key=lambda c: step_order.index(c.step_id) if c.step_id in step_order else 10**6)
    first = failed[0]
    suspect = first.step_id
    retained = step_order[: step_order.index(suspect)] if suspect in step_order else []

    if first.name == "schema_match":
        # 资产本身缺字段，重试无意义
        action = "switch_asset" if has_fallback_asset else "give_up"
    elif step_failures.get(suspect, 0) >= MAX_STEP_RETRIES:
        action = "switch_asset" if has_fallback_asset else "give_up"
    else:
        action = "retry_step"
    return FailureAttribution(
        suspect_step=suspect, reason=f"{first.name}: {first.detail}", rollback_to=suspect,
        retained_steps=[] if action == "switch_asset" else retained, next_action=action,
    )
