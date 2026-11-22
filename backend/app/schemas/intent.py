"""② 协调 · 意图结构（Intent）。

字段对齐 old-version `prompts/intent_analysis.md` 的输出 schema，并补充可追溯信息（铁律 4）：
`source` / `rule_ids` / `fallback_reason` / `warnings`。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

IntentType = Literal["definition", "data_fetch", "discovery", "analysis", "other"]
IntentSubType = Literal["attribution", "trend", "prediction", "compare"]


class AssetRef(BaseModel):
    """用户明确点名的资产，如 BI 链接 `dataset/list?id=2479` → kind=dataset, id=2479。"""

    kind: str
    id: str


class Intent(BaseModel):
    session_id: str = ""
    query: str
    query_normalized: str | None = None          # 指代消解后的改写；无需消解为 None
    intent_type: IntentType = "other"
    intent_sub_type: IntentSubType | None = None  # 仅 analysis
    asset_refs: list[AssetRef] = Field(default_factory=list)
    domain: str | None = None                    # transaction | community | commercialization | resource_cost | overseas
    sub_domain: list[str] = Field(default_factory=list)   # domain 已识别时非空，且只含该 domain 封闭枚举内的值
    kb_intents: list[str] = Field(default_factory=list)   # 仅 domain=commercialization：商业化知识库粗召意图
    keywords: list[str] = Field(default_factory=list)     # 业务强相关词，供下游召回
    dim_terms: list[str] = Field(default_factory=list)    # keywords 的子集：疑似维度取值的具名词
    time_hint: str | None = None                 # 原文片段（"近 7 天"），不解析为日期
    notes: str | None = None                     # 命中的 Rule / 歧义点

    # ── 可追溯 / 可审计 ──
    source: Literal["llm", "rule"] = "rule"      # 本 intent 由 LLM 还是规则兜底产出
    rule_ids: list[str] = Field(default_factory=list)     # 规则命中 / 兜底填充的依据，如 domain.rule_a.inner_loop
    fallback_reason: str = ""                    # source=rule 时：为什么没走 LLM
    warnings: list[str] = Field(default_factory=list)     # 校验警告（概念冲突 / 非法枚举 / 推断提示），下游可见
