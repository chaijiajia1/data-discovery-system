"""外部能力接口。生产实现对接真实的 BI 资产检索 / 知识检索 / 权限 / 取数引擎；
本地开发与测试使用 `fakes.py` 里的内存实现。"""
from __future__ import annotations

from typing import Any, Protocol

from ..schemas.execution import StepResult
from ..schemas.knowledge import KnowledgeItem, RetrievalPack
from ..schemas.recall import CandidateAsset
from ..schemas.task import TaskContract


class BIAssetRetrieval(Protocol):
    """BI 资产检索（bi 通道）：返回已认证的看板 / 数据集 / 图表。`params` 为 intent_to_params 的产物。"""

    async def search(
        self, query: str, *, limit: int = 10, cookie: str | None = None, params: dict[str, Any] | None = None
    ) -> list[CandidateAsset]: ...

    async def get(self, asset_ids: list[str]) -> list[CandidateAsset]:
        """按 id 物化已知资产（L0/L1 子目标跳过召回时使用）。"""
        ...


class LineageRetrieval(Protocol):
    """表与字段血缘检索（graph 通道）：沿知识图谱扩展，覆盖问法与命名脱节的漏召回。"""

    async def search(
        self, query: str, *, limit: int = 10, cookie: str | None = None, params: dict[str, Any] | None = None
    ) -> list[CandidateAsset]: ...


class CaliberRetrieval(Protocol):
    """口径查找（knowledge 通道）：指标定义、业务术语、使用约束。"""

    async def search(
        self, query: str, *, limit: int = 10, cookie: str | None = None, params: dict[str, Any] | None = None
    ) -> list[KnowledgeItem]: ...


class KnowledgeHybridRetrieval(Protocol):
    """知识检索（context-query）。`retrieve` 返回完整 pack（含 partial 信息）；`search` 只要条目。"""

    async def retrieve(
        self, query: str, *, user_id: str | None = None, cookie: str | None = None, max_tokens: int | None = None
    ) -> RetrievalPack: ...

    async def search(self, query: str, *, limit: int = 10) -> list[KnowledgeItem]: ...


class PermissionService(Protocol):
    async def check(self, user_id: str | None, asset: CandidateAsset) -> tuple[bool, str]: ...


class ExecutionEngine(Protocol):
    """取数执行引擎：把执行拆成可单独重跑 / 回滚的最小步骤。"""

    def steps(self) -> list[str]: ...

    async def run_step(
        self, step_id: str, *, asset: CandidateAsset, contract: TaskContract, state: dict[str, Any]
    ) -> StepResult: ...
