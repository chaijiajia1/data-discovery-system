"""⑥ 执行与反馈产物。"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .knowledge import VerifiedPair


class StepResult(BaseModel):
    step_id: str                     # resolve_fields | build_query | run_query | ...
    status: Literal["ok", "failed", "rolled_back", "skipped"] = "ok"
    output: Any = None
    error: str = ""
    verified: bool = False           # 该步产出是否已通过核验（回滚时保留已验证的上游）


class CheckResult(BaseModel):
    name: str                        # non_empty | schema_match | metric_range | grain_consistency ...
    passed: bool
    detail: str = ""
    step_id: str | None = None       # 该检查指向的步骤，用于归因


class VerificationReport(BaseModel):
    checks: list[CheckResult] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)


class FailureAttribution(BaseModel):
    suspect_step: str                # 最小可疑步骤
    reason: str
    rollback_to: str                 # 回滚到的步骤（通常 == suspect_step）
    retained_steps: list[str] = Field(default_factory=list)  # 保留的已验证上游步骤
    next_action: Literal["retry_step", "switch_asset", "give_up"] = "retry_step"


class ExecutionAttempt(BaseModel):
    attempt: int
    asset_id: str
    steps: list[StepResult] = Field(default_factory=list)
    verification: VerificationReport = Field(default_factory=VerificationReport)
    attribution: FailureAttribution | None = None


class ExecutionResult(BaseModel):
    status: Literal["success", "failed"] = "failed"
    asset_id: str | None = None
    data_preview: Any = None
    attempts: list[ExecutionAttempt] = Field(default_factory=list)
    verified_pair: VerifiedPair | None = None   # 首个通过核验的 pair（便于展示）
    verified_pairs: list[VerifiedPair] = Field(default_factory=list)   # 全部通过核验 → 回流 ① 的候选
    feedback_written: bool = False
