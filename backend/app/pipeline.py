"""六阶段编排。

    ① 知识预注入 → ② 协调 → ③ 工作流规划 → ④ 并行召回 → ⑤ 资产选择 → ⑥ 执行与反馈
         ▲                                                                  │
         └──────────── 已验证 pair 经 pair_store 回流（跨请求闭环）────────────┘

以事件流形式输出（SSE 友好）：stage1..stage6 / done / error。
"""
from __future__ import annotations

import time
from typing import Any, AsyncIterator

from .errors import StageError
from .stages import RunContext, Stage
from .stages.s1_preinjection import PreInjectionStage
from .stages.s2_coordination import CoordinationStage
from .stages.s3_planning import PlanningStage
from .stages.s4_recall import RecallStage
from .stages.s5_selection import SelectionStage
from .stages.s6_execution import ExecutionStage
from .structured_log import log_event, set_run, set_session

# (事件名, ctx 属性名, 阶段)
PIPELINE: list[tuple[str, str, Stage]] = [
    ("stage1", "preinjection", PreInjectionStage()),
    ("stage2", "contract", CoordinationStage()),
    ("stage3", "plan", PlanningStage()),
    ("stage4", "recall", RecallStage()),
    ("stage5", "selection", SelectionStage()),
    ("stage6", "execution", ExecutionStage()),
]


def _dump(obj: Any) -> Any:
    return obj.model_dump(mode="json") if hasattr(obj, "model_dump") else obj


async def stream_pipeline(ctx: RunContext) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    set_session(ctx.session_id)
    set_run(ctx.run_id)
    step_ms: dict[str, int] = {}
    t_all = time.perf_counter()
    for event, attr, stage in PIPELINE:
        t0 = time.perf_counter()
        try:
            result = await stage.run(ctx)
        except StageError as e:
            yield "error", {"stage": e.stage, "message": e.message}
            return
        except Exception as e:  # noqa: BLE001
            log_event("pipeline.unexpected_error", stage=stage.name, error=repr(e))
            yield "error", {"stage": stage.name, "message": str(e)}
            return
        setattr(ctx, attr, result)
        ms = step_ms[event] = int((time.perf_counter() - t0) * 1000)
        yield event, {"name": stage.name, "result": _dump(result), "elapsed_ms": ms}

        # ⑤ 之后：没有可用资产 → 终止，不进入执行
        if event == "stage5" and not ctx.selection.primary:
            yield "done", {"status": "no_asset", "run_id": ctx.run_id, "risk_flags": ctx.selection.risk_flags,
                           "total_ms": int((time.perf_counter() - t_all) * 1000), "step_ms": step_ms}
            return

    ex = ctx.execution
    yield "done", {
        "status": ex.status, "run_id": ctx.run_id, "asset_id": ex.asset_id,
        "closed_loop": {"feedback_written": ex.feedback_written, "pair_id": ex.verified_pair.pair_id if ex.verified_pair else None},
        "total_ms": int((time.perf_counter() - t_all) * 1000), "step_ms": step_ms,
    }
