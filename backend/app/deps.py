"""依赖容器：各阶段通过 Deps 取外部能力，便于测试替换。"""
from __future__ import annotations

from dataclasses import dataclass, field

from .llm_client import LLMClient, NullLLM, build_llm
from .services.context_query import ContextQueryClient
from .services.demo_data import DEMO_ASSETS, DEMO_KNOWLEDGE
from .services.fakes import FakeBIRetrieval, FakeEngine, FakeKnowledgeRetrieval, FakeLineageRetrieval, FakePermission
from .services.protocols import (
    BIAssetRetrieval, CaliberRetrieval, ExecutionEngine, KnowledgeHybridRetrieval, LineageRetrieval, PermissionService,
)
from .services.recall_channels import BiChannel, FrameworkKnowledgeChannel, GraphChannel
from .settings import Settings, settings as _default_settings
from .collab.registry import ProfileRegistry, build_profile_registry
from .stores import InMemoryPairStore, VerifiedPairStore, build_pair_store


@dataclass
class Deps:
    pair_store: VerifiedPairStore = field(default_factory=InMemoryPairStore)
    kh: KnowledgeHybridRetrieval = field(default_factory=FakeKnowledgeRetrieval)
    bi: BIAssetRetrieval = field(default_factory=FakeBIRetrieval)               # bi 通道：BI 资产发现
    lineage: LineageRetrieval = field(default_factory=FakeLineageRetrieval)     # graph 通道：表与字段血缘发现
    caliber_kb: CaliberRetrieval = field(default_factory=FakeKnowledgeRetrieval)  # knowledge 通道：口径查找
    permission: PermissionService = field(default_factory=FakePermission)
    engine: ExecutionEngine = field(default_factory=FakeEngine)
    llm: LLMClient = field(default_factory=NullLLM)
    profiles: ProfileRegistry = field(default_factory=ProfileRegistry)          # ④ 角色能力画像注册表
    settings: Settings = field(default_factory=lambda: _default_settings)


def build_default_deps(s: Settings | None = None) -> Deps:
    """默认装配：知识检索走 context-query（可关闭回退 fakes）；④ 的 bi/graph/knowledge 通道由 RECALL_CHANNELS_ENABLED 开启；权限 / 引擎暂用 fakes。"""
    s = s or _default_settings
    if s.DEMO_MODE:          # 外网演示：不管环境变量怎么写，都只用假数据——不连内网、不用 LLM、不带 cookie、不落盘
        s = s.model_copy(update={
            "CONTEXT_QUERY_ENABLED": False, "RECALL_CHANNELS_ENABLED": False, "CONTEXT_QUERY_COOKIE": "", "CONTEXT_QUERY_OWNER": "",
            "LLM_API_KEY": "", "RECALL_LLM_ENABLED": False, "PLANNER_LLM_LEVEL_ENABLED": False, "PLANNER_LLM_EXTRA": 0,
            "ROLE_PROFILE_STORE_BACKEND": "memory",
        })
    kh: KnowledgeHybridRetrieval = FakeKnowledgeRetrieval(DEMO_KNOWLEDGE) if s.DEMO_MODE else FakeKnowledgeRetrieval()
    if s.CONTEXT_QUERY_ENABLED:
        kh = ContextQueryClient(
            base_url=s.CONTEXT_QUERY_BASE_URL, source_types=s.context_source_types(), timeout_sec=s.CONTEXT_QUERY_TIMEOUT_SEC,
            default_cookie=s.CONTEXT_QUERY_COOKIE, default_owner=s.CONTEXT_QUERY_OWNER, agent_id=s.CONTEXT_QUERY_AGENT_ID,
            max_tokens=s.PREINJECT_TOKEN_BUDGET,
        )
    channels: dict = {}
    if s.DEMO_MODE:                    # 演示：三条示例问句专用的假数据（见 services/demo_data.py）
        channels = {"bi": FakeBIRetrieval(DEMO_ASSETS), "lineage": FakeLineageRetrieval(DEMO_ASSETS), "caliber_kb": FakeKnowledgeRetrieval(DEMO_KNOWLEDGE)}
    elif s.RECALL_CHANNELS_ENABLED:      # 默认关闭：真实资产元信息缺分区/粒度，⑤⑥ 仍是占位实现
        channels = {"bi": BiChannel(s), "lineage": GraphChannel(s), "caliber_kb": FrameworkKnowledgeChannel(s)}
    return Deps(
        **channels,
        kh=kh,
        pair_store=build_pair_store(s.PAIR_STORE_BACKEND, s.PAIR_STORE_PATH),
        profiles=build_profile_registry(s.ROLE_PROFILE_STORE_BACKEND, s.ROLE_PROFILE_STORE_PATH, s.COLLAB_QUALITY_ALPHA),
        llm=build_llm(endpoint=s.LLM_ENDPOINT, api_key=s.LLM_API_KEY, model=s.LLM_MODEL, timeout_sec=s.LLM_TIMEOUT_SEC, max_workers=s.LLM_MAX_WORKERS,
                      max_tokens=s.LLM_MAX_TOKENS, auth_header=s.LLM_AUTH_HEADER, json_mode=s.LLM_JSON_MODE),
        settings=s,
    )
