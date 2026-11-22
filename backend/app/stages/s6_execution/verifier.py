"""结果核验：每条检查指向最小责任步骤，便于失败归因。"""
from __future__ import annotations

from ...schemas.execution import CheckResult, StepResult, VerificationReport
from ...schemas.recall import CandidateAsset
from ...schemas.task import TaskContract


def verify(steps: dict[str, StepResult], asset: CandidateAsset, contract: TaskContract, metrics: list[str] | None = None) -> VerificationReport:
    """metrics：本资产负责的指标（多主资产时各自只负责一部分）；缺省为契约全部指标。"""
    checks: list[CheckResult] = []
    for sid, r in steps.items():
        if r.status == "failed":
            checks.append(CheckResult(name=f"step_ok:{sid}", passed=False, detail=r.error, step_id=sid))
    if any(not c.passed for c in checks):
        return VerificationReport(checks=checks)   # 有步骤直接失败时，后续内容检查无意义

    fields = steps.get("resolve_fields")
    if fields is not None and isinstance(fields.output, dict):
        have = {f.lower() for f in fields.output.get("fields", [])}
        missing = [m for m in (metrics if metrics is not None else (contract.metrics.value or [])) if str(m).lower() not in have]
        checks.append(CheckResult(name="schema_match", passed=not missing,
                                  detail=f"字段缺失: {missing}" if missing else "指标字段齐全", step_id="resolve_fields"))
    run = steps.get("run_query")
    if run is not None and isinstance(run.output, dict):
        rows = run.output.get("rows") or []
        checks.append(CheckResult(name="non_empty", passed=bool(rows), detail=f"rows={len(rows)}", step_id="run_query"))
    return VerificationReport(checks=checks)
