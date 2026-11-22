"""④ 的 LLM 闸门：开关 / 整次召回的调用总预算 / 并发上限 / 超时与失败降级 / 调用统计。

所有角色（改写 query、评估结果、自评是否接活、撰写求助问题）都经它调用 LLM，保证：
- 未配置 LLM（NullLLM）或 `RECALL_LLM_ENABLED=false` → `json()` 恒返回 None，调用方回落到规则（铁律 3）；
- 总预算 `RECALL_LLM_MAX_CALLS` 约束的是**整次召回**，不是单个 agent；超预算后的调用直接回落，不抛错；
- 同时在途的调用不超过 `RECALL_LLM_CONCURRENCY`；单次调用超过 `RECALL_LLM_TIMEOUT_SEC` 视为失败；
- 失败（上游错误 / 超时 / 输出不是 JSON 对象）一律返回 None 并计数，不影响其他 agent。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..errors import UpstreamError
from ..llm_client import LLMClient, NullLLM
from ..schemas.collab import LLMUsage
from ..settings import Settings

logger = logging.getLogger(__name__)


class RecallLLM:
    def __init__(self, llm: LLMClient, settings: Settings) -> None:
        self._llm = llm
        self.enabled = bool(settings.RECALL_LLM_ENABLED) and not isinstance(llm, NullLLM)
        self.max_calls = max(0, int(settings.RECALL_LLM_MAX_CALLS))
        self._timeout = float(settings.RECALL_LLM_TIMEOUT_SEC)
        self._sem = asyncio.Semaphore(max(1, int(settings.RECALL_LLM_CONCURRENCY)))
        self._calls = self._failures = self._fallbacks = 0
        self._exhausted = False
        self._by_purpose: dict[str, int] = {}

    @property
    def has_budget(self) -> bool:
        """还能再调用一次吗（补召回后要靠 LLM 评估新增结果，预算用完就不必补召回）。"""
        return self.enabled and self._calls < self.max_calls

    async def json(self, purpose: str, prompt: str) -> dict[str, Any] | None:
        """返回解析后的 JSON 对象；不可用 / 超预算 / 失败返回 None（调用方回落规则）。"""
        if not self.enabled:
            return None
        if self._calls >= self.max_calls:                     # 预算检查与计数之间没有 await：并发下不会超发
            self._exhausted = True
            self._fallbacks += 1
            return None
        self._calls += 1
        self._by_purpose[purpose] = self._by_purpose.get(purpose, 0) + 1
        from ..stages.s2_coordination.agent import parse_llm_json   # 延迟导入：避免 collab ↔ stages 循环依赖
        async with self._sem:
            try:
                raw = await asyncio.wait_for(self._llm.complete(prompt=prompt, json_mode=True), timeout=self._timeout)
                return parse_llm_json(raw)
            except (UpstreamError, ValueError, asyncio.TimeoutError) as e:
                logger.info("recall LLM %s skipped: %s", purpose, e or type(e).__name__)
                self._failures += 1
                self._fallbacks += 1
                return None

    def usage(self) -> LLMUsage:
        return LLMUsage(enabled=self.enabled, calls=self._calls, failures=self._failures, fallbacks=self._fallbacks,
                        max_calls=self.max_calls, budget_exhausted=self._exhausted, by_purpose=dict(self._by_purpose))
