"""⑥ 执行与反馈：执行 → 核验 → 失败归因最小步骤并回滚（保留已验证上游）→ 通过核验的 pair 回流 ①。"""
from __future__ import annotations

from ...schemas.execution import ExecutionResult
from ..base import RunContext, Stage
from .executor import execute_with_rollback


class ExecutionStage(Stage[ExecutionResult]):
    name = "stage6_execution"

    async def run(self, ctx: RunContext) -> ExecutionResult:
        return await execute_with_rollback(ctx)
