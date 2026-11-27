"""阶段基类与运行上下文。每个阶段：读 RunContext 的上游产物 → 返回本阶段产物（由 pipeline 回填）。"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from ..deps import Deps
from ..schemas.execution import ExecutionResult
from ..schemas.knowledge import PreInjectionResult
from ..schemas.recall import RecallResult
from ..schemas.selection import SelectionResult
from ..schemas.task import TaskContract
from ..schemas.workflow import PlanningResult

T = TypeVar("T")


@dataclass
class RunContext:
    query: str
    deps: Deps
    user_id: str | None = None
    cookie: str | None = None     # SSO cookie，透传给 context-query
    session_id: str = "default"
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    clarifications: dict[str, Any] = field(default_factory=dict)   # 用户主动补充的条件（可选）：slot -> value
    max_parallel: int | None = None        # A_max：前端用户输入的最大并行 Agent 数；缺省取 settings
    confirmed_assets: dict[str, list[str]] = field(default_factory=dict)   # 会话内已确认的资产：指标 -> asset_id 列表

    # 各阶段产物（由 pipeline 依次回填）
    preinjection: PreInjectionResult | None = None
    contract: TaskContract | None = None
    plan: PlanningResult | None = None
    recall: RecallResult | None = None
    selection: SelectionResult | None = None
    execution: ExecutionResult | None = None
    token_budget: int | None = None


class Stage(Generic[T]):
    """阶段协议：`name` 为 SSE 事件名与日志前缀。"""

    name: str = ""

    async def run(self, ctx: RunContext) -> T:  # pragma: no cover - interface
        raise NotImplementedError
