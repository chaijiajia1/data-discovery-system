"""配置：环境变量 + 默认值（pydantic_settings）。"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # HTTP 服务
    HOST: str = "0.0.0.0"
    PORT: int = 8080
    LOG_LEVEL: str = "info"
    # 外网演示模式：只用假数据，不连任何内网上游、不用 LLM、不处理用户 cookie，并屏蔽 cookie / 调试类端点、对较重的接口限流（见 demo.py）
    DEMO_MODE: bool = False
    DEMO_RATE_LIMIT_PER_MIN: float = 30.0   # 每个客户端每分钟允许的 /api/stage* 等请求数
    DEMO_RATE_BURST: int = 15               # 令牌桶容量（允许的突发）
    DEMO_TRUST_PROXY: bool = True           # 前面有可信反向代理（Caddy）时信其 X-Forwarded-For 最右地址；直接暴露端口时必须设 false
    CORS_ORIGINS: str = "*"      # 逗号分隔；前端 HTML 以 file:// 打开时 Origin 为 null，开发期用 *

    # LLM（OpenAI 兼容协议；为空时各阶段退化为规则/启发式实现）
    # LLM 网关（OpenAI 兼容协议，对齐旧版 drr-service-debug：内部网关 runway）。只要配了 LLM_API_KEY 就启用；endpoint / model 有默认值
    LLM_API_KEY: str = ""
    LLM_ENDPOINT: str = "https://runway.devops.rednote.life/v1/chat/completions"
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_MAX_TOKENS: int = 8192
    LLM_AUTH_HEADER: str = "api-key"         # api-key（runway）| authorization（Authorization: Bearer）
    LLM_JSON_MODE: bool = False              # 是否发 response_format=json_object（旧版不发；确认网关支持后再开）
    LLM_TIMEOUT_SEC: float = 60.0
    LLM_MAX_WORKERS: int = 16                # LLM 客户端独立线程池大小（不占用 asyncio 默认线程池，避免与各检索通道抢线程）

    # ① 知识预注入
    PREINJECT_TOKEN_BUDGET: int = 500      # 上下文容量 L：仅作硬上界，防止溢出
    PREINJECT_TAU: float = 0.001            # 停止阈值 τ：最大「边际收益/token」低于它即停
    PREINJECT_PAIR_TOPK: int = 5            # 路 2：最多取几条相似历史 query
    PREINJECT_PAIR_MIN_SIM: float = 0.3     # 路 2：query 相似度下限
    PREINJECT_RETRIEVAL_TOPK: int = 10
    # 块间交互效用 w_ij 的系数（负=冗余/冲突，正=协同）
    PREINJECT_W_REDUNDANCY: float = 1.0
    PREINJECT_W_CONFLICT: float = 1.5
    PREINJECT_W_SYNERGY: float = 0.2
    PREINJECT_INCONSISTENT_DISCOUNT: float = 0.7   # consistencyStatus=possible_inconsistent 时 u_k 乘以该系数
    PREINJECT_VIEWS_ENABLED: bool = True    # 按消费方（② 协调 / ③ 规划 / ④ 各角色 / ⑤ 选择）分发不同的知识视图；关闭 = 所有消费方共用 injected 全量

    # 路 1：context-query（POST /v1/context/retrieve）。鉴权 SSO：优先透传请求 Cookie，其次 CONTEXT_QUERY_COOKIE
    CONTEXT_QUERY_ENABLED: bool = True
    CONTEXT_QUERY_BASE_URL: str = "https://context-query.devops.xiaohongshu.com"
    CONTEXT_QUERY_SOURCE_TYPES: str = "knowledge"   # 逗号分隔：memory,skill,knowledge
    CONTEXT_QUERY_COOKIE: str = ""
    CONTEXT_QUERY_OWNER: str = ""                   # 无登录用户时的服务间 owner
    CONTEXT_QUERY_AGENT_ID: str = ""
    CONTEXT_QUERY_TIMEOUT_SEC: float = 10.0

    # ③ 工作流规划
    MAX_PARALLEL_AGENTS: int = 4            # A_max 缺省值；前端用户输入优先
    PLANNER_NUM_CANDIDATES: int = 3         # K：规则生成的候选图个数（LLM 提案另计，至多 PLANNER_LLM_EXTRA 个）
    PLANNER_LLM_EXTRA: int = 2
    PLANNER_REQUIRE_BI_ASSET: bool = False  # 入口定级：知识必须指明 BI 资产才算「已指明资产」。默认 False：知识指明的任何数据源资产（含明细表）就算
    PLANNER_LLM_LEVEL_ENABLED: bool = True  # 入口定级的子目标覆盖状态由 LLM 判定；未配置 / 失败 / 输出非法时回落规则（levels.py）
    # Score = α·覆盖度 + β·并行度 − γ·Agent 消耗（三项都已归一到 ~[0,1]）：覆盖优先，并行与成本在覆盖相同时分胜负
    PLANNER_W_COVERAGE: float = 0.6         # α
    PLANNER_W_PARALLEL: float = 0.2         # β
    PLANNER_W_COST: float = 0.2             # γ

    # ④ 检索通道（对齐 old-version recall_multi：bi / graph / knowledge）。默认关闭，回退内存 fakes
    RECALL_CHANNELS_ENABLED: bool = False
    RECALL_TIMEOUT_SEC: float = 15.0
    RECALL_BI_BASE_URL: str = "https://aimi.devops.xiaohongshu.com"
    RECALL_GRAPH_BASE_URL: str = "https://dwassetgraph-metric-graph-service.devops.xiaohongshu.com"
    RECALL_COMM_GRAPH_BASE_URL: str = "https://kb-asset-finder-service.devops.xiaohongshu.com"
    RECALL_KB_GRAPH_BASE_URL: str = "https://kb-graph-recall-service.devops.xiaohongshu.com"
    RECALL_KNOWLEDGE_BASE_URL: str = "https://aimi.devops.xiaohongshu.com"

    # ④ 并行召回
    RECALL_ROLE_TIMEOUT_SEC: float = 30.0
    # ④ 角色由 LLM 驱动（角色模板 prompts/roles/）：改写检索 query / 评估结果 / 自评是否接活 / 撰写求助问题。
    # 未配置 LLM、调用失败、超预算 → 回落到规则（铁律 3）。默认关闭，稳定后再打开。
    RECALL_LLM_ENABLED: bool = False
    RECALL_LLM_MAX_CALLS: int = 40          # 整次召回的 LLM 调用总预算（不是单个 agent 的）
    RECALL_LLM_CONCURRENCY: int = 8         # 同时在途的 LLM 调用上限
    RECALL_LLM_TIMEOUT_SEC: float = 30.0    # 单次 LLM 调用超时
    # ④ 召回多少条由 LLM 决定（每个「指标 × 通道」）：范围 [MIN, MAX]，缺省 PREINJECT_RETRIEVAL_TOPK；评估后觉得不够就继续召回，最多补召回 MAX_SUPPLEMENTS 次
    RECALL_MIN_LIMIT: int = 3
    RECALL_MAX_LIMIT: int = 30
    RECALL_MAX_SUPPLEMENTS: int = 2
    # ④ 协作：复用优先 / 定向询问 / 广播 / 画像
    COLLAB_DIRECT_THRESHOLD: float = 0.6    # 路由打分 ≥ 此值且不跨域、画像覆盖 → 定向询问；否则广播
    COLLAB_VOLUNTEER_MIN: float = 0.4       # 广播时角色自评估匹配度 ≥ 此值才揭榜
    COLLAB_ACCEPT_MIN: float = 0.3          # 被询问角色自评估匹配度 < 此值则婉拒（low_fit）
    COLLAB_ADOPT_MIN_CONF: float = 0.3      # 请求方采纳回应所需的证据卡最低置信度
    COLLAB_WAIT_SEC: float = 5.0            # 同一需求别人正在探查时，最多等它的证据卡多久
    COLLAB_QUALITY_ALPHA: float = 0.2       # 画像历史响应质量的 EMA 步长
    # ④ 发现驱动的转交：角色找到东西后主动交给其他角色继续查（口径→BI+血缘；血缘→BI+口径；BI→口径），结果再触发下一跳
    COLLAB_HANDOFF_ENABLED: bool = True
    COLLAB_HANDOFF_MAX_DEPTH: int = 2       # 最多转交几跳（自己的需求为 0 跳）
    COLLAB_HANDOFF_FANOUT: int = 2          # 一张证据卡最多转交几个发现（相关口径 / 资产）
    COLLAB_HANDOFF_MAX_TOTAL: int = 12      # 一次召回里转交的总预算
    ROLE_PROFILE_STORE_BACKEND: str = "json"  # memory | json：角色能力画像注册表
    ROLE_PROFILE_STORE_PATH: Path = _ROOT / "data" / "role_profiles.json"

    # ⑥ 执行与反馈
    EXEC_MAX_ATTEMPTS: int = 3

    # 历史 query 文件（路 2 的数据源，同时是 ⑥→① 的回流点）
    PAIR_STORE_BACKEND: str = "jsonl"  # memory | jsonl
    PAIR_STORE_PATH: Path = _ROOT / "data" / "history_queries.jsonl"

    # 前端页面（后端在 / 托管，同域访问；只暴露这一个 HTML 及其 *_files 资源目录）
    UI_HTML_PATH: Path = _ROOT.parent / "Data Discovery System · Interactive Walkthrough.html"

    # 资源根目录
    PROMPTS_DIR: Path = _ROOT / "prompts"

    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False, env_file=".env", extra="ignore")

    def context_source_types(self) -> list[str]:
        return [t.strip() for t in self.CONTEXT_QUERY_SOURCE_TYPES.split(",") if t.strip()]


settings = Settings()
