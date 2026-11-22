"""统一异常：携带阶段名，便于 SSE error 事件上报。"""
from __future__ import annotations


class StageError(Exception):
    """某阶段失败。"""

    def __init__(self, stage: str, message: str):
        super().__init__(f"[{stage}] {message}")
        self.stage = stage
        self.message = message


class ContractValidationError(StageError):
    """任务契约不合法（如 DAG 有环、必填槽位缺失）。"""


class UpstreamError(StageError):
    """外部服务（检索/权限/执行引擎/LLM）调用失败。"""
