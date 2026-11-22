# Data Discovery Backend

资产发现流程后端：把自然语言数据问题，收敛为任务契约，并通过多角色并行探查选出可执行、可核验的数据资产。

```
① 知识预注入 → ② 协调 → ③ 动态工作流规划 → ④ 并行召回 → ⑤ 资产选择 → ⑥ 执行与反馈
     ▲                                                                        │
     └──────────── 通过核验的 查询—资产/知识 pair 回流（闭环）──────────────────┘
```

当前状态：**代码框架**。每个阶段的接口、数据契约、编排、测试已打通；阶段内部为可运行的占位实现（规则 / 启发式 + 内存假服务），后续逐阶段细化。

## 快速启动

```bash
make setup      # python3.11 venv + 依赖
make test       # 全量测试（含 stage3 / 检索通道 / HTTP）
make serve      # http://127.0.0.1:8088
```

```bash
curl -N -X POST localhost:8088/api/discover -H 'content-type: application/json' \
  -d '{"query": "最近30天各渠道的DAU和GMV"}'
```

## API

| Method | Path | 说明 |
|---|---|---|
| GET | `/api/health` | 健康检查 |
| POST | `/api/discover` | **SSE 流**：`stage1`..`stage6` / `done` / `error` |
| POST | `/api/stage2` | 只跑 ① → ②，返回 `preinjection` 与 `contract`（`contract.intent` 即意图） |
| POST | `/api/stage3` | 跑 ① → ② → ③：返回 `contract` 与 `plan`（入口定级 / 覆盖单元 / 候选图与评分）；支持 `run_id` 续跑、`max_parallel`（A_max）、`confirmed_assets` |
| POST | `/api/stage5` | 跑 ① → ⑤：返回 `selection`（primary / alternates / discarded 三档，理由结构化）与 `recall_reused`；带 `run_id` 且同一 query / 补充信息 / 已确认资产时**复用缓存里 ④ 的结果**（含用户确认的图），不重跑 ③④；没有缓存才自己跑 |
| POST | `/api/stage4` | 跑 ① → ④：多 agent 并行检索，返回 `plan` 与 `recall`（候选池 / 证据卡 / 协作轨迹 `collab`，含事件流 `collab.events`）；支持 `run_id` 续跑；带 `graph`（`{nodes:[{id,role,targets}], edges:[{src,dst,kind}]}`）时按这张图运行（每个节点一个 agent 副本），角色未注册或有环返回 400 |
| GET | `/api/profiles` | 角色能力画像（覆盖域 / 资产类型 / 历史响应质量与采纳统计） |
| POST | `/api/plan/evaluate` | 用同一套上下文、可行性约束与公式评估前端自定义的图（需先有 `/api/stage3` 的 `run_id`） |
| POST | `/api/discover/sync` | 非流式，返回全部事件（调试） |
| GET | `/api/roles` | 已注册的探查角色 |
| GET | `/api/pairs` | 已验证 pair（闭环回流内容） |
| GET | `/api/prompts/:name` | prompt 模板原文 |

请求体：`{"query", "session_id?", "user_id?", "clarifications?"}`。
② 不会追问用户：提取不到 metric 时改用 `intent.keywords`（剔除维度词）作为后续查找词，契约里标 `source=inferred`、`rule_id=fallback.metrics.from_keywords`；keywords 也为空则下游回落到整句 query。`clarifications: {"metrics": ["DAU"]}` 仅作为用户主动补充条件的可选入口（优先级最高）。

## 项目结构

```
backend/
├── app/
│   ├── main.py              # FastAPI 入口（SSE）
│   ├── pipeline.py          # 六阶段编排，事件流输出
│   ├── deps.py              # 依赖容器（外部能力注入点）
│   ├── settings.py / structured_log.py / llm_client.py / errors.py / prompts.py / text_utils.py
│   ├── schemas/             # 阶段间数据契约（pydantic）
│   │   ├── knowledge.py     # ① KnowledgeItem / PreInjectionResult / VerifiedPair
│   │   ├── intent.py        # ② Intent（domain / sub_domain / intent_type / keywords …，带 source / rule_ids）
│   │   ├── task.py          # ② TaskContract（每个槽位带 source / rule_id；含 intent）
│   │   ├── workflow.py      # ③ WorkflowGraph（then/dep 边）/ SubGoal·EntryLevel / CoverageUnit / PlanContext / 候选图评分明细
│   │   ├── recall.py        # ④ Evidence / CandidateAsset / Conflict / Blackboard / Ask
│   │   ├── selection.py     # ⑤ 主/佐证资产、排序与拒绝理由
│   │   └── execution.py     # ⑥ 步骤、核验、失败归因、回流 pair
│   ├── stages/
│   │   ├── s1_preinjection/ # pair_source + retrieval_source → token 预算边际收益选择
│   │   ├── s2_coordination/ # agent（query+知识→intent+slots）→ intent_validate → fallback（metric←keywords）→ defaults；taxonomy 为五域词表单一事实源
│   │   ├── s3_planning/     # levels（规则定级 / 汇总）+ level_agent（LLM 判子目标覆盖状态）→ context（R/C₀/R′/D）→ candidates → coverage（覆盖 + ASAP 排程）→ feasibility → scorer → agent（LLM 提案）
│   │   ├── s4_recall/       # scheduler（分层并行）→ merger（去重 + 冲突保留）
│   │   ├── s5_selection/    # rules（核心选用原则 1-8）→ agent（LLM 复核 + 校验 + 硬护栏）→ stage
│   │   └── s6_execution/    # executor（回滚循环）/ verifier / attribution / feedback
│   ├── roles/               # 角色模板（channels / dims / 并发上限；注册表，规划阶段自动可见）
│   ├── collab/              # 黑板（证据卡 / 认领等待 / 候选池）+ 能力画像注册表 + 路由打分 + 协作中枢 hub
│   ├── stores/              # VerifiedPairStore（内存 / JSONL）—— ⑥→① 回流点
│   └── services/            # 外部能力协议 + 内存 fakes（BI 检索 / 知识检索 / 权限 / 执行引擎）
├── prompts/                 # LLM prompt 模板（支持 {{ include: x.md }}）
└── tests/                   # 每阶段一个测试文件 + e2e（含闭环、HTTP/SSE）
```

## 各阶段要点与 TODO

| 阶段 | 已有骨架 | 待细化 |
|---|---|---|
| ① 预注入 | 已实现：见下文「① 知识预注入」 | 历史 query 匹配换向量检索；u_k / w_ij 系数调参；检索块的 asset_ids 抽取 |
| ② 协调 | 见下文「② 协调 · intent」；不追问：缺指标用 keywords 兜底，口径冲突交 ③ 的 caliber_state 走 L1（由口径角色核验）；默认规则带 rule_id | intent 驱动 ③ 角色选择 / ④ 召回范围；token scope 歧义处理 |
| ③ 规划 | 见下文「③ 动态工作流规划」 | 节点工作量由历史耗时校准；LLM 提案质量评估 |
| ④ 召回 | 见下文「④ 并行召回」 | 真实通道的权限 / 分区元信息；画像按域 / 资产类型细分；更多证据卡类型 |
| ⑤ 选择 | 见下文「⑤ 资产选择」 | 真实资产的字段元信息（BI 的 field_meta / sql_rules / cube_rules）接入；scope 词表扩充 |
| ⑥ 执行 | 步骤级回滚（保留已验证上游）、最小步骤归因、换资产兜底、pair 回流 | 真实取数引擎；更多核验规则 |

## ① 知识预注入

```
query ─┬─ 路1 context-query  POST /v1/context/retrieve  ──┐
       └─ 路2 历史 query 文件 data/history_queries.jsonl ─┴→ 按块 id 合并（双路命中 u+0.1）→ 贪心选择 → 注入知识
```

- **路 1**：`services/context_query.py`。SSO 鉴权：优先透传请求 `Cookie`（或 body `cookie`），其次 `CONTEXT_QUERY_COOKIE`；`user_id` 作为 `owner`。`sourceTypes` 由 `CONTEXT_QUERY_SOURCE_TYPES` 控制，`maxTokens` = 容量 L。`partial` / 失败原因记入 `route_status`。
- **路 2**：`data/history_queries.jsonl`，每行一条 `VerifiedPair` = `query` + 用到的知识（`knowledge` 快照）+ 最终选择的 `asset_ids`。按 query term 的 Dice 相似度匹配（≥ `PREINJECT_PAIR_MIN_SIM`），展开为"当时用到的知识"和一条"最终选了哪些资产"的摘要块；相关度 = 相似度 × 历史成功率。⑥ 核验通过后自动追加/更新；也可手工编辑种子。
- **选择**（`stages/s1_preinjection/selector.py`）：
  - `u_k` = 相关度（`possible_inconsistent` 乘 0.7）；`c_k` = token 成本；
  - `w_ij`：同主题内容对不上 → 冲突 `-1.5·min(u)`；文本相似 → 冗余 `-1.0·sim·min(u)`；同资产不同侧面 → 协同 `+0.2·min(u)`；
  - `Δ(k|S) = u_k + Σ_{j∈S} w_jk`，每轮选 `Δ/c` 最大者，最大值 `< τ`（`PREINJECT_TAU`）即停；
  - 容量 `L`（`PREINJECT_TOKEN_BUDGET`）只是硬上界：放不下的块剔除为 `over_budget`，不影响停止条件；
  - 输出 `injected`、`dropped`（含原因）、`trace`（每轮 u / c / Δ / Δ/c / 与已选块的交互）。
- 任一路甚至两路失败只降级（少知识 / 无知识继续），不中断流程。

## 扩展点

- **接外部服务**：实现 `services/protocols.py` 中的协议，在 `deps.build_default_deps` 替换 fakes。
- **新增角色**：在 `roles/` 写 `Role` 子类，加入 `REGISTRY`，并在 `RoleSpec` 声明 `phase / channels（查询的通道）/ dims（能提供的选择维度）/ max_concurrency`。
- **换回流存储**：实现 `VerifiedPairStore` 协议（DB / 向量库），由 `PAIR_STORE_BACKEND` 切换。

## 环境变量

见 `.env.example`。`LLM_*` 全部留空时，各阶段使用规则 / 启发式兜底，服务仍可完整运行。

## ② 协调 · intent

输入：用户提问 + ① 的注入知识（+ 用户主动补充）；输出：`contract.intent`（SSE `stage2` 事件里同样带出）。

```
query + 注入知识 ──▶ CoordinationAgent ──▶ LLM(JSON: intent + slots) ──▶ 归一化 ──▶ validate_intent ──▶ Intent
                          │ LLM 未配置 / 调用失败 / 输出不可解析
                          └──▶ 规则兜底（intent_rules：Rule A–K、触发词、时间/资产解析）
```

- `Intent` 字段对齐 old-version `intent_analysis.md`：`intent_type` / `intent_sub_type` / `domain`（五域）/ `sub_domain`（封闭枚举）/ `kb_intents`（仅商业化）/ `keywords` / `dim_terms` / `time_hint` / `asset_refs` / `notes`；
  另加可追溯字段 `source`（llm | rule）、`rule_ids`、`fallback_reason`、`warnings`。
- `intent_validate.py` 移植自 old-version：枚举校验、domain 概念冲突检测只**警告**不改 domain、str→list 归一化、非法 sub_domain 剔除（保留 warning）；domain 已识别但 sub_domain 为空时由规则补全并记录 `rule_ids`。
- 词表单一事实源是 `taxonomy.py`：prompt 里的枚举/触发词由它渲染，校验与规则兜底也用它，避免 prompt 与代码漂移。
- 预注入知识的用法：LLM 路径整体注入 prompt 做消歧义；规则路径用它补 `keywords`，且仅在 query 自身无业务域信号时才用知识推断 domain（并写入 `warnings`）。

## ③ 动态工作流规划

规划 Agent 把任务契约 + ① 的覆盖情况转成 AOV 图 G=(V,E,A)：顶点是子任务（角色 × 子目标），有向边是依赖（`dep` 等产出 / `then` 仅顺序），每个节点关联一个角色。

**入口定级：整个任务只有一个级别**（`levels.py`）。两步：

① 按子目标（每个指标、每个维度）判**覆盖状态**——它是依据，不是各自的级别（子目标没有 `level` 字段）。**由 LLM 判定**（`level_agent.py`；`prompts/level.md` 用 L0 会话确认 / L1 知识指明资产 / L2 取全文核对 / L3 召回 四档判据，映射为 covered / doubtful / uncovered：L0、L1→covered，L2→doubtful，L3→uncovered；`PLANNER_LLM_LEVEL_ENABLED` 控制），判定记 `rule_id=llm.goal.<status>` 与 LLM 给出的理由；下表是规则版，作兜底：LLM 未配置 / 调用失败 / 输出不可解析 → 全部回落规则；单个子目标条目非法（未知名字、非法状态、`asset_ids` 不在注入知识里导致 covered/doubtful 无资产）→ 只有该目标保留规则结果；会话内已确认的子目标不由 LLM 改写。

| 子目标 | covered（已覆盖） | doubtful（存疑） | uncovered（未覆盖） |
|---|---|---|---|
| 指标 | 会话内已确认资产；或知识指明了数据源资产（BI 资产或明细表都算）且口径一致（用户明确指定口径也算一致） | 知识指明了资产，但口径冲突 / `possible_inconsistent` / 没有口径证据 | 知识没指明任何资产（`PLANNER_REQUIRE_BI_ASSET=true` 时还要求其中有 BI 资产） |
| 维度 | 知识把它绑定到了资产字段 | 只有定义没绑定 | 知识没覆盖 |

② 汇总成唯一的任务级别（取最差者，**始终由代码确定性汇总**，LLM 不直接定级），写进 `plan.level`，并附 `level_rule_id` / `level_reason`（一句话说明）/ `level_basis`（决定该级别的子目标）：

| 级别 | 条件 | 走向 |
|---|---|---|
| L0 | 全部子目标 covered | 不召回、不核验，直接执行（④ 仅物化已知资产，⑤ 的权限硬过滤不跳过） |
| L1 | 没有 uncovered，但有 doubtful | 跳过召回；图由召回角色组成的**核验节点**（每个节点的工作量为 1）构成：口径角色去读知识全文核对口径，其余角色只核验已知资产（seed）的新鲜度 / 维度覆盖；之后进选择 |
| L2 | 有任一子目标 uncovered | 完整召回 + 选择，**对全部口径**——不为已覆盖的口径开例外 |

这样 C₀ 也是全有或全无：L0 / L1 → 全部 口径×通道 都在 C₀；L2 → 没有 C₀。已知资产（`known_assets`）只在 L0 / L1 才物化为候选；L2 时它们只作血缘通道的种子。**明细表（如 `dws.trade_gmv_d`）也是可用的数据源，知识只指到它就算指明了资产**：BI 资产优先只影响 ⑤ 选哪个，不决定要不要召回；以前要求「BI 资产 + 明细表」都有，知识只指到 dws 表就整体判 L2，过严。若你希望必须有 BI 资产才跳过召回，开 `PLANNER_REQUIRE_BI_ASSET=true`（只要求 BI 资产，不再要求两类都有）。

**覆盖度**：覆盖网格 R = 口径 × 通道（bi / graph / knowledge），`Cov = (C₀ ∪ 图命中) 的 (口径, 通道) 对数 / (口径数 × 通道数)`。C₀ 只有 L0 / L1 口径（入口定级本就跳过召回）的全部通道；**L2 口径没有 C₀——不默认知识已命中**，每个 口径×通道 都要真正去查，预注入知识指明的资产只作后续召回（尤其是血缘通道）的种子。口径评估维度 `caliber` 也只在全部口径为 L0 时才由知识满足。每个召回节点查询一个或多个口径（`targets`），经角色的一个或多个通道（`RoleSpec.channels`）；节点命中的对 = targets × channels ∩ R′。覆盖是确定性的，同一对重复命中不增加覆盖，只增加成本。

**评分**：`Score(G) = α·Cov + β·Par − γ·Cost`（三项都已归一；默认 α=0.6, β=0.2, γ=0.2，见 `PLANNER_W_*`）

| 项 | 定义 | 取值 |
|---|---|---|
| `Cov` 覆盖度 | 见上 | 全覆盖 = 1 |
| `Par` 并行度 | `(W − M)/(W − 1)`；W = 总工作量 = Σ 节点工作量，M = 完工时间 = ASAP 排程的关键路径长度 | 全并行 → 接近 1；全串行 → 0；单节点（W=1）记 1；空图记 0 |
| `Cost` Agent 消耗 | `N / (\|R′\| + \|D\|)`；N = Agent 数（节点数），分母 = 最细粒度拆法的 Agent 数 | 越小越好 |

- 节点工作量 = 它处理的「口径 × 通道」对数；不在网格内的节点、以及没有待覆盖对时（L0 / L1）只做核验的节点为 1。所以「一个超大节点」不会被误当成并行：它的时长就是内部要串行处理的对数。
- 批量节点（一个 Agent 带多个口径）与全通道节点 `omni_discovery`（一个 Agent 依次查三个通道）省 Agent、时长变长：Cost 项变好、Par 项变差，划不划算由权重决定。例：3 个口径 × 3 通道，`per_channel`（3 个批量 Agent，各串行处理 3 个口径）= 0.6 + 0.2×0.8 − 0.2×0.5 = 0.66，高于 `fanout`（9 个单对 Agent 全并行）= 0.6 + 0.2 − 0.2 = 0.6。
- 权重调节：加大 β 偏向多 Agent 并行，加大 γ 偏向少 Agent；α 保持最大使覆盖优先。
- 并列取节点度数标准差更小的图（再并列取节点更少、名字字典序）。

**可行性过滤**（硬约束，不进公式；不可行候选保留 `violations` 供展示，但不参选）：峰值并发 ≤ A_max（按 ASAP 排程逐时间片检查，不是只看拓扑层）、各角色同时运行的节点数 ≤ 自身上限、所需评估维度 D 都有召回节点提供、图合法（角色已注册、无环）。覆盖度不是硬约束，而是被最大化的目标。全部不可行时降级选违反最少者并置 `degraded`。

**种子依赖**：血缘通道以已有召回内容（BI 通道的资产）为种子，所以规划图里 `lineage` 节点依赖覆盖其任一口径的 `bi` 节点（dep 边）。图如实包含这条边，批量节点的关键路径才会如实变长（`bi@DAU+GMV`(2) → `lineage@DAU+GMV`(2)，而按指标拆分是 `bi@DAU`(1) → `lineage@DAU`(1)，两条流水线并行）。

**候选生成**（全部生成并评分后由 `pick_top` 取前 `PLANNER_NUM_CANDIDATES` 个，默认 3）：对每个通道独立选择「拆分」（每个待覆盖口径一个节点，口径之间并行）或「批量」（一个节点串行处理全部口径，Agent 更少），3 个通道共 8 种组合，再加两种全通道 Agent（`omni`）方案——`lanes`（全部拆分：每个口径一条并行流水线）、`per_channel`（全部批量）、`split_bi` / `split_graph` / `split_bi_graph` …（只拆分所列通道）、`per_caliber`（每个口径一个全通道 Agent）、`single_agent`（一个全通道 Agent 处理全部口径）。只有一个口径时拆分等于批量，去重后只剩 `lanes` / `per_caliber`。超过 A_max 或角色并发上限的节点用 `then` 边排到后面。

**取 top K 的规则**：第一个始终是分数最高的可行候选；然后保证三种风格各有最优代表——`parallel`（口径、通道各自一个节点）、`omni`（每个口径一个全通道 Agent）、`batched`（存在一个节点处理多个口径）——余下按分数补满。所以按指标拆分、指标之间并行的方案不会被批量方案挤出候选。默认权重下哪种风格排第一取决于 β（并行）与 γ（Agent 成本）的取舍：例如 2 个指标时 `per_caliber` 领先；3 个指标时 `split_bi`（BI 按指标并行、血缘 / 口径批量）领先，调大 `PLANNER_W_PARALLEL` 或调小 `PLANNER_W_COST` 会让 `lanes` 上升。LLM 可选地额外提案（`prompts/planning.md`，至多 `PLANNER_LLM_EXTRA` 个，可带多口径节点），经同样的过滤与评分——省掉预检节点会因 D 无提供者而不可行；LLM 不可用时纯规则，定级 / 过滤 / 评分始终确定性。

**角色模板**

| 角色 | 阶段 | 通道 | 提供的评估维度 / 说明 |
|---|---|---|---|
| `bi_asset_discovery` | 召回 | bi | `metric_match` / `grain_fit` / `freshness`；内联核对分区与粒度；基础通道：已认证看板 / 数据集 / 图表 |
| `lineage_discovery` | 召回 | graph | `metric_match` / `dim_coverage`；内联核对维度覆盖；补充通道：以已有召回内容为种子沿血缘扩展，覆盖问法与命名脱节的漏召回 |
| `caliber_lookup` | 召回 | knowledge | `caliber`；内联佐证核验；指标定义 / 业务术语 / 使用约束 |
| `omni_discovery` | 召回 | bi + graph + knowledge | `metric_match` / `caliber` / `grain_fit` / `freshness` / `dim_coverage`；一个 Agent 查三个通道，三类内联核对都做 |
| `dim_value_resolver` | 召回 | dim_value（不在网格内） | `dim_coverage`；具名维度取值 → 字段；仅在有 `dim_terms` 且非 resource_cost / overseas 域启用 |

**没有单独的「选择类角色」**：原来的 `dimension_scout` / `freshness_precheck` / `cross_validator` 已并入召回角色，由它们按自己提供的评估维度**边召回边核对**（`roles/checks.py`）。`permission` 不再要求节点提供，由 ⑤ 的确定性权限过滤负责。最终选哪个主资产仍由 ⑤ 汇总排序。

## 检索通道（`services/recall_channels.py`）

对齐 old-version `recall_bridge.py` / `recall_multi.py`：

- `intent_to_params(intent)`：intent → 各通道参数（asset_types / top_k / 业务域→知识区域与图谱中文域 / 商业化两段式与 `kb_intents` topK 等）。③ 把它放进 `plan.recall_params`，④ 的角色原样透传；
- `BiChannel`（`/mark/openapi/dataAsset/search` HYBRID）、`GraphChannel`（通用 `/api/recall`；商业化整路换成 kb-asset-finder `/graph/recall`）、`FrameworkKnowledgeChannel`（`framework/retrieve` 老路 + `chunk-retrieve` dibp 新路并行取并集；商业化先粗召筛文件再把 `frameworkNames` 传给精召，粗召失败向上抛、chunk 单通道失败只降级）；
- 只透传 SSO token cookie；默认关闭（`RECALL_CHANNELS_ENABLED=false`，回退内存 fakes），因为真实资产缺分区 / 粒度元信息，⑤⑥ 仍是占位实现。
- 未移植：资源成本 / 国际化域的**本地图谱整路替换**（`graph_local_json`），目前这两域走远程图谱；维值库向量召回（`dim_value_recall`），`dim_value_resolver` 暂用知识检索兜底。

## ④ 并行召回

按 ③ 的工作流图**按依赖尽早开始（ASAP）**并行执行：每个节点在它自己的全部前驱完成后立刻开始，不等同层的其他节点（与 ③ 评分时的排程模型一致）；`A_max` 信号量限制峰值并发，`RECALL_ROLE_TIMEOUT_SEC` 超时、单角色失败隔离且不卡住下游。所有节点共享一个**协作中枢**（黑板 + 画像注册表 + 路由）。节点内分到的每个「指标 × 通道」也并发探查（血缘 / 口径在同一指标的 BI 之后），不再逐个 `await`。

**混合检索**：两个通道，结果归一化进同一个候选池（`services/normalize.py`）

| 通道 | 来源 | 作用 | 进池时 |
|---|---|---|---|
| `bi` | BI 通道：已认证看板 / 数据集 / 指标层 | 真实可用的基础资产 | `channels=["bi"]`，元信息最可信，合并时优先 |
| `knowledge:graph` | 知识通道·图谱路：沿血缘扩展 | 覆盖问法与命名脱节、但有血缘关系的资产 | 分数 ×0.9（补充，略低于 BI 基础） |
| `knowledge:vector` | 知识通道·向量路：知识 chunk | 带出关联的资产（经 BI 取回元信息）、指标、维度实体 | 分数 ×0.9；实体进 `related_entities` |

同一资产被多路命中合并为一条：通道取并集、元信息以 BI 为准并由知识通道补缺口（占位 stub 绝不覆盖真实元信息）、分数取最大；多通道命中融合分 +0.1（封顶 1）。`RecallResult.by_channel` 给出各通道组合的资产数。（③ 里的 `graph` / `knowledge` 两列就是知识通道的图谱路与向量路。）

**agent 副本**：工作流图里的每个节点是一个独立的 agent 实例（`agent_id` = 节点 id，如 `bi_asset_discovery@GMV`），同一角色在图里出现 n 次就有 n 个副本（`replica`/`replicas`），各自独立运行、负载独立（`busy` 按副本计），同层并行。副本之间的沟通：

- **黑板**：同一需求只探查一次——副本 A 认领后，其他副本（含同角色的）等它的证据卡再复用；批量节点（带多个口径）每轮优先做没人认领的口径，同伴做完的直接复用，不干等；
- **定向询问**：路由按**角色**画像打分（画像是角色级的，同角色副本共享，采纳 / 忽略 / 婉拒记在角色上），选出角色后再在其副本里挑一个：持有该需求（已认领 / 已产出 / 目标口径就是它）的优先 → 负载低的 → id；请求方不会问自己，但可以问同角色的另一个副本；
- **婉拒 → 广播**：首选副本婉拒（如 `busy`）后降级为广播，**同角色的其他副本**也会自评估揭榜，揭榜者按 匹配度 → 持有 → 负载 → id 排序；
- **轨迹**：`collab.agents` 给出每个副本的身份、目标口径、自己探查的需求（`produced`）、复用别人的需求（`reused`）、发起 / 响应 / 婉拒的询问数；`asks[]` 里有 `requester_role`、`target_agent`、`served_by`、`served_role`。

**协作协议**（`collab/hub.py`；角色产生需求 `Need(kind, subject, domains, asset_type)` 后依次解决，越靠前越省）

1. **复用优先**：先读黑板，已有可复用的**证据卡**（置信度够、业务域相容）就直接用——不询问、不广播、不重复探查；卡被复用算作其作者被采纳（权重 0.5）。
2. **认领等待**：同一需求别人正在探查 → 等它的证据卡再复用，而不是各自重复探查；认领者失败会 `release` 唤醒等待者；超时（`COLLAB_WAIT_SEC`）则自己探查。
3. **自己探查**：角色有能力（给了 `produce`）→ 认领后探查并发布证据卡。
4. **双通道求助**（角色没有该能力，如血缘节点需要 BI 资产作种子）：对其余在场角色按画像**路由打分**（`collab/router.py`）：
   - 高置信命中某角色画像（打分 ≥ `COLLAB_DIRECT_THRESHOLD`，不跨域，请求的域在其覆盖域内）→ **定向询问**，省去全量广播与无关自评估；
   - 低置信 / 跨域 / 画像未覆盖 → **黑板广播**，各角色自评估、自愿揭榜（匹配度 ≥ `COLLAB_VOLUNTEER_MIN`），匹配度最高的揭榜者回答；
   - 被点名的角色保有**自主权**，可婉拒并说明理由：`out_of_scope`（不服务该类需求）/ `domain_uncovered` / `low_fit` / `busy`（同时响应上限 = `max_concurrency`）；请求方收到婉拒后**降级为广播**，婉拒者不再被重复询问。
5. **采纳反馈**：请求方判断回应是否可用（非空且置信度 ≥ `COLLAB_ADOPT_MIN_CONF`）→ 采纳 / 忽略；连同婉拒一起写回画像。

路由打分 = `(0.40·需求类型 + 0.15·资产类型 + 0.15·业务域 + 0.15·关键词重叠 + 0.15·专才加成) × (0.5 + 0.5·quality)`。`quality` 是历史响应质量：长期被忽略 / 婉拒的角色分数掉到 0.6 以下就不再被定向点名（顺位给下一名，或回退广播）。

**能力画像注册表**（`collab/registry.py`）：每个角色一份轻量摘要——覆盖域、可服务的需求类型、擅长资产类型、关键词、`quality`（EMA，步长 `COLLAB_QUALITY_ALPHA`）与 采纳 / 忽略 / 婉拒 / 被询问 统计。**离线**以 `RoleSpec` 为种子建档（`ensure`，重复调用只刷新静态字段，保留学到的状态）；**在线**每次响应结果 `record` 更新（采纳→目标 1，忽略→0，婉拒→0.35）；可检索（`search(need, roles)`）；持久化到 `data/role_profiles.json`（`ROLE_PROFILE_STORE_BACKEND=json|memory`），文件损坏回退种子，不影响召回。

**角色**：BI 资产发现（`assets`）、表与字段血缘发现（`lineage`，见下）、口径查找（`caliber`，没有口径证据时去问谁承载该指标）、全通道发现（一个 Agent 依次查三个通道）、维值解析。一个节点可带多个口径，节点内各口径并发查询。

**角色由 LLM 驱动（`RECALL_LLM_ENABLED`，默认关闭）**：每个召回角色有自己的角色模板（`prompts/roles/<角色名>.md`，共用 `_protocol.md` 的黑板协作协议），描述角色定义、命中标准与何时婉拒。**路由仍是确定性的画像打分**，LLM 只介入下面这些位置（`app/roles/llm_agent.py`，每个都是**单遍**的一次调用，没有每个 agent 的循环）：

| 位置 | LLM 做什么 | 不可用 / 失败时 |
|---|---|---|
| 节点探查前（`_plan`） | 一个节点**一次**调用，把它分到的全部指标改写成自足的检索 query（必须含指标名），并给出「换角度」的备选说法 `angles` | 直接用指标名查 |
| 节点探查后（`_annotate`） | 一个节点**一次**调用，评估自己探查出的每个候选是否相关；写成证据（`source=llm_filter`，不相关的也保留并说明理由），**不增删候选**；编造的资产 id 丢弃 | 不标注 |
| 结果为空 / 被判定全部不相关 | 用 `angles` 里的说法换个角度再查——走转交机制（新 key、`via`=原需求、`hop+1`），受 `COLLAB_HANDOFF_*` 的深度 / 扇出 / 总预算 / 开关限制 | 不重试 |
| 求助问题（`_ask`） | 角色自己的需求写成自足的问题（含主题；转交来的需求已带 `via` / `hint_assets`，不再花 LLM） | 用模板问句 |
| 被点名 / 揭榜（`_selfeval`），**仅限角色自己的需求** | 硬规则（`out_of_scope` / `domain_uncovered` / `busy`）仍确定性判断、不花 LLM；定向询问由 LLM 判断语义上是否胜任（婉拒必须有理由，`llm_decline：…`）；广播先过画像粗筛（匹配度低于 `COLLAB_ACCEPT_MIN` 不花 LLM），再由 LLM 并发自评、自愿揭榜 | 回落到画像匹配度规则 |

**召回多少条由 LLM 决定，不够再继续召回**：改写 query 时 LLM 给每个「指标 × 通道」一个 `limit`（钳制到 `RECALL_MIN_LIMIT`=3 ~ `RECALL_MAX_LIMIT`=30，缺省 `PREINJECT_RETRIEVAL_TOPK`）；评估时 LLM 对每张卡判断召回的够不够（`coverage`），不够就给出继续召回的条数（必须更大，否则翻倍，不超过上限）和可选的新 query，系统按它补召回，**只把新增资产交给 LLM 评估**。最多补召回 `RECALL_MAX_SUPPLEMENTS`=2 次；补召回没有新增资产、失败、或 LLM 预算用完就停；候选全部不相关的走「换角度重试」而不是多召回。`collab.supplements` 记录补召回的需求 key（不计入 `retrievals`）。

**控制转交的成本**：转交（handoff）是调用量的大头，三处收敛——
- 转交来的需求（`need.via` 非空）已带触发它的发现与 `hint_assets`，**只用规则判断**（硬规则 + 画像匹配度），不花 LLM 自评、不花 LLM 写问题；LLM 自评只给角色自己的需求（定向被点名、广播揭榜）；
- **自己能做的线索自己顺着查**（`delegate`：自己能服务该类需求就直接探查，黑板的认领 / 复用保证不重复；不再询问同类角色）；血缘 / 口径角色需要承载资产的 BI 卡时同理（`_carrier_card`：自己能查 BI 就自己查）。只有自己做不了的才交给别的角色；
- 已在候选池里且分区 / 粒度元信息完整的资产，不再转交给 BI 重新解析（`handoff._resolved`）。
同一 query 的实测（假检索通道、全部接活的脚本化 LLM）：全通道节点 2 个：LLM 调用 17 → 4（询问 12 → 0）；BI + 血缘 + 口径三个节点：LLM 调用 18 → 9。

**成本护栏**：`RECALL_LLM_MAX_CALLS`（默认 40）是**整次召回**的 LLM 调用总预算（不是单个 agent 的），超了之后直接回落规则；`RECALL_LLM_CONCURRENCY`（8）限制同时在途的调用；`RECALL_LLM_TIMEOUT_SEC`（30）超时算失败；LLM 客户端（`llm_client.py`，调用方式对齐旧版 runway 网关）：只要配了 `LLM_API_KEY` 就启用（`LLM_ENDPOINT` / `LLM_MODEL` 默认 runway / gpt-4o-mini），`api-key` 请求头（`LLM_AUTH_HEADER=authorization` 可切 Bearer），`max_tokens` 默认 8192，默认不发 `response_format`（`LLM_JSON_MODE`），日志只记元信息；JSON 被截断时 `parse_llm_json` 保留完整部分；用独立线程池（`LLM_MAX_WORKERS`，16），不与检索通道抢 asyncio 默认线程池。`collab.llm` 给出调用数 / 失败 / 回落 / 预算是否用尽 / 按用途的分布。选择阶段角色（`dimension_scout` / `freshness_precheck` / `cross_validator`）仍是纯规则。

**产出**：`RecallResult.candidates`（统一候选池）、`cards`（证据卡）、`related_entities`、`by_channel`、`collab`（`reuse` / `asks`（路由、回应、揭榜、采纳结果）/ `profile_updates` / `retrievals` / `stats`）。`retrievals` 里同一个 need key 出现多次就是重复探查——协调正常时每个 key 只出现一次。

## ⑤ 资产选择

对齐 old-version `prompts/asset_selection.md`。**按口径（指标）分组**：每个口径从 ④ 的候选池里选出恰好 1 个主资产，其余归 **alternates（备选）** 或 **discarded（不用）**，所有候选必须且只能落在三档之一；同一资产是多个口径的主资产时合并为一条（`covers` 列出负责的口径）。

**⑤ 读 ④，不重复算 ④ 已经做过的事**
- **字段匹配只有一份**（`app/asset_match.py`）：⑤ 的字段命中与 ④ 的维度覆盖核对共用同一个匹配器（字面 > 同义[知识桥接] > 部分[角色一致] > 推断；仅凭名称 / 描述的「推断」不算覆盖证据），不再各写一套子串匹配。
- **粒度冲突不静默丢弃**：④ 合并时 `meta.grain` 只留一个值，冲突在 `Conflict.values` 里。⑤ 在 `grain_hard` 丢弃前先看冲突：冲突值里有满足要求粒度的 → 不丢弃（`grain_conflict`），有不受影响的候选时让位到备选，全都受影响时仍选出主资产并保留 `conflict:` 标记（铁律 5：冲突显式保留）。
- **④ 的 LLM 相关性评估是软信号**（`Evidence.subject` 记它是对着哪个指标判的）：判不相关的候选有别的候选时让位到备选（`upstream_irrelevant`，理由写明「④ 评估为…」）；全都不相关时不让主资产落空，只留降权说明。**不替代硬护栏**：无目标指标字段、粒度粗于要求、主键、scope、权限仍由 ⑤ 的规则判定。
- **新鲜度**：④ 核对过、却没取到最新分区 → `freshness_unknown:{id}` 风险标记，并作为盲区信号出现在理由里；④ 没核对过（如 L0）不臆测。与 ⑤ 原有的 `partition_unknown`（分区字段是否存在，④ 不查）互补。
- **佐证核验**：④ 发现候选之间没有共同字段 → 盲区信号。
- **进 LLM prompt**：每个候选带 `upstream_evidence`（本口径的 ④ 核对：粒度冲突 / `llm_filter` / 新鲜度 / 口径 / 佐证，最多 6 条；不含 FIELD 证据，维度覆盖 ⑤ 自己用同一匹配器算），`prompts/selection.md` 说明「不要重复推断，只是软信号」。
- **权限是 ⑤ 独有**（④ 不做）：无权限降级到备选（`needs_permission`），可申请。

**元信息缺失时不让主资产落空**（真实接口的资产常常没有字段 / chart 严格性等元信息）
- **通道适配器写类型化字段**（`services/recall_channels.py`）：BI 的 chart `核心指标` → `metric_ids`、`核心维度` → `supported_dims`，`热度` → `heat_30d`；图谱的 dataset `tags` → `tags`、chart 的 `metric_ids` / `supported_dimensions` / 是否有取数规则 → `is_strict`。以前只塞进 `meta.extra`，⑤ 读的是类型化字段，真实 chart 因此永远命中不了任何指标。BI 接口没给 `is_strict` 就留空（None），不臆测。
- **未知 ≠ 不支持**：具名维度取值的支撑判断里，没有字段元信息的资产不再被判「不支持按该取值过滤」（`dim_value_unresolved` 如实标注）；有字段元信息的仍必须匹配。
- **放宽兜底**（`relax_if_empty`）：没有任何候选满足全部门槛时，把「只是没过软门槛」的备选（不是硬不合规、没有权限问题、确实命中了口径）放宽为可选，照常按选用原则挑最好的，规则 id `fallback_candidate`、风险标记 `fallback_primary:{id}`、降权说明里写明，不装作它满足了全部条件。有满足全部门槛的候选时不启用；没命中口径的候选永远不会被放宽。
- **dim_terms 不含维度名**：② 会剔除与 `dimensions` 重名的 dim_term（「渠道」是维度，不是维度取值），prompt 也写明。

**核心选用原则**（`stages/s5_selection/rules.py`，每个决定带 `rule_ids`）

| # | 原则 | 实现 |
|---|---|---|
| 1 | primary 恰好 1 个 | 每个口径一个；候选与口径弱相关以上时不允许为空 |
| 2 | chart 严格四条件优先 | `is_strict` + `perm.granted` + `metric_ids` 覆盖口径 + `supported_dims` 覆盖维度（`[]` = 无限制）；任一不满足 → alternates 并写明哪条不满足 |
| 3 | 字段覆盖度 > 资产类型 > 数据层级 > 热度 | 覆盖度 = 命中候选字段的（口径 + 维度）数 ÷ 总数；dataset > hive_table；应用层（`app_` / 宽表汇总聚合）> 明细层（`dws_/dm_/dwd_/ods_` / 明细流水），降级到明细层时写明「应用层 X 无维度，改用明细层 Y」；热度按 1.5 倍分桶，仅显著更高才起作用；趋势类偏好汇总粒度 |
| 4 | OneDash 高权重 | 大盘 / 核心 / 概览 / 整体 / 趋势类 query；覆盖度明显更低（差 > 0.25）或是 cube 时不加权 |
| 5 | cube 表 | 有非 cube 的可用资产 → cube 让位到 alternates；选 cube 必须限定切面（`query_fields` 带切面字段与 `dim_values`），没有切面信息则标 `cube_requires_filter` 风险 |
| 6 | 粒度与 scope | 命中维度的字段必须是主键之一，否则 discarded（「X 粒度为…，无法按 Y 分析」）；候选 scope（名称 / 描述里的直播 / 搜索 / 店播 / 买手 / 点点等）必须 ≥ query 范围，query 限定窄 scope 时 scope 匹配的表优先、全站宽表降级；资产时间粒度粗于要求 → discarded |
| 7 | 用户粒度宽表 | 主键含 uid / user_id / did 且指标是 DAU / UV → 切片字段进 `query_fields`，聚合用求和不得去重计数；切片取值未知标 `slice_values_unknown` |
| 8 | 权限 | 无权限 → alternates（`needs_permission`，可申请），不直接丢弃 |

**理由结构化**（`Reason`）：`conclusion` / `hits`（业务词 → 候选的哪个位置，`matched` 逐字复制候选字段原文含 ★ 与中文别名，`how` = 字面相同 / 同义（知识桥接）/ 部分 / 推断）/ `misses` / `rules` / `downweights`（本可加分被压制）/ `blind_spots`（只能引用后端客观算出的 `signals`，如「中文名覆盖较低: 3/68」「主键信息缺失」）。`knowledge_refs` 是实际参与桥接的知识条目 id，并在 primary 的结论里引用。

**缺失的元信息不臆测**：没有字段元信息的候选（如只有 BI 检索结果）不会被当成「无目标指标字段」丢弃，而是按名称 / 描述 / 标签做「推断」命中（覆盖度只记 0.5，不如有真实字段的资产）并标 `fields_unknown` / `partition_unknown` 风险；命中不到就放 alternates 等待补全元信息。

**具名维度取值**：`intent.dim_terms`（如「自然堂」）需要候选有对应的维度字段——有维值解析结果（④ 的 `dimension` 证据卡）就必须有解析出的字段，并在 `query_fields` 里填 `dim_values`；没有解析结果就要求至少有一个可过滤的维度字段。没有任何候选支持 → `primary=[]`，写明「无候选支持按 X 过滤」。

**LLM 复核**（`agent.py`，`prompts/selection.md`，可选）：每个口径一次调用、口径间并行，规则预判（`hints`，含硬不合规标记）一并交给 LLM。输出经校验：`asset_id` 必须来自候选、primary 至多 1 个、三档补齐且唯一、`query_fields` 字段必须存在、`命中` 必须逐字复述候选原文、`盲区` 只能引用 signals。**硬护栏**：LLM 选出规则判为不合规（无权限 / 时间粒度粗于要求 / 无目标指标字段 / 主键粒度不符 / scope 不足 / 维值无法过滤 / 不满足 chart 四条件）的资产做主资产 → 整个口径回退规则。LLM 未配置 / 失败 / 输出不可解析 → 该口径回退规则，`groups[].mode / fallback_reason` 如实记录；一次选择里口径可以混用（`mode=mixed`）。

**风险标记**（`risk_flags`）：`no_asset_selected`、`metrics_not_covered:[…]`、`no_caliber_evidence:<id>`、`conflict:<id>:<topic>`、`needs_permission:<id>`、`partition_unknown:<id>`、`fields_unknown:<id>`、`cube_requires_filter:<id>`、`slice_values_unknown:<id>`、`dim_term_unsupported:<term>`、`dim_value_unresolved:<term>`、`intent_conflict`。

与 ⑥ 的衔接：⑥ 的兜底资产取自 `alternates` 里有权限的资产。

**血缘发现以已有召回内容为种子**：`lineage_discovery` 先读黑板 / 求助 BI 角色拿到该口径的 `assets` 证据卡（复用优先，不重复探查），种子 = 预注入知识指明的该口径资产（含会话确认的已知资产）在前，BI 通道召回的资产在后（最多 5 个），再沿知识图谱通道找血缘。真实图谱接口 `/api/recall` 没有「种子」参数，所以种子的资产名并入 `keywords` 让图谱做语义 / 血缘匹配（见 `GraphChannel`）；种子写在 `lineage` 证据卡的 `seeds`，血缘召回到的候选带 `meta.extra.lineage_seeds`；没有任何种子时回退按口径名检索，并在卡的 `summary` 里写明「无种子」。

**前端 Roles / Communication 与工作流图保持一致**：第四步的页面不再是脚本演示，而是由工作流图和真实协作轨迹驱动——
- **Roles 面板** = 图里出现的角色种类（首次出现顺序），每张卡标 `×n` 副本数并列出各节点（目标口径 · 第几个/共几个）；运行后显示真实统计（探查数 / 复用数 / 发起 · 响应 · 婉拒的询问数）；
- **Communication 泳道** = 图里的节点，一个节点一条泳道，泳道顺序与图一致；事件取自 `collab.events`：实线 = 询问（箭头 = 请求方 → 被问方）、回答 / 揭榜（被问方 → 请求方）、虚线 = 婉拒（带理由）、点线 = 证据卡复用（作者 → 复用方）、黑板广播（降级广播会注明）、叉 = 无人应答；同一 agent 复用自己的证据卡不算协作，不画；
- **随 ③ 同步预览**：在 Phase 3 切换候选或编辑图（增删节点 / 角色）时，两个面板立即按当前图重画（泳道只有骨架，没有事件）；提交后 Phase 4 以确认的图调用 `/api/stage4`，运行结果替换预览；
- 事件类型：`probe`（自己探查）/ `reuse` / `ask` / `reply` / `decline` / `broadcast` / `volunteer` / `unanswered`，按发生顺序编号 `seq`。

**发现驱动的转交（handoff）**：之前的交互只发生在「某角色自己做不了才求助」（血缘向 BI 要种子）。现在角色产出证据卡后，会按发现物主动把后续线索交给其他角色继续查（`collab/handoff.py`）：

| 发现者 | 找到了什么 | 转交给 |
|---|---|---|
| 口径查找 | 相关口径（口径文档里关联的其他指标，含其关联资产） | **BI**（找它的资产，并解析口径关联的资产元信息）+ **血缘**（以口径关联的资产为种子沿血缘找） |
| 血缘发现 | 沿血缘关联的表 / 数据集 / 图表 | **BI**（按 id 解析元信息 / 找认证数据集）+ **口径查找**（这些资产的口径定义） |
| BI 发现 | 基础资产 | **口径查找**（该资产的口径定义） |

- 转交的结果（新证据卡）会再触发下一跳：口径 → 血缘找到的表 → BI 解析 → 口径…，受 `COLLAB_HANDOFF_MAX_DEPTH`（默认 2 跳）、`COLLAB_HANDOFF_FANOUT`（一张卡最多转交几个发现，默认 2）、`COLLAB_HANDOFF_MAX_TOTAL`（一次召回的总预算，默认 12）限制，`COLLAB_HANDOFF_ENABLED=false` 可关闭；
- 每个后续需求只处理一次（以 need key 去重，黑板上已有证据卡的不再转交）；
- 谁来接：别的 agent 服务该类需求 → 走现有的询问（路由选角色，再在副本里挑持有该需求 / 负载低的那个，被婉拒则广播）；没有别人但自己能做（如单个全通道 agent）→ 自己顺着查；谁都做不了 → 跳过（不制造无人应答的噪音）；
- 转交里的空结果表示「这条线索没东西」，不计入回应方的画像（普通求助的空回应仍照旧降权）；
- 转交链路产生的资产（只经转交找到的）也并入候选池，`found_by` 为产出该证据卡的角色；
- 轨迹：`collab.events` 里转交产生的询问带 `via`（触发它的证据卡 key）和 `hop`（第几跳），`stats.handoffs` 与 `agents[].handoffs` 统计转交次数；前端泳道图用紫色箭头画转交，Roles 卡显示 handed off 计数。

**角色模板存放位置**：角色定义在代码里，不是配置文件。`RoleSpec` 数据类在 `app/roles/base.py`；各角色的模板（名称 / 描述 / 阶段 / 通道 / 能力画像种子 `serves·asset_types·domains·keywords` / `max_concurrency`）与行为（`run` / `produce`）写在 `app/roles/builtin.py` 各角色类的 `spec` 里；`app/roles/__init__.py` 的 `REGISTRY` 是注册表，规划阶段自动可见；`GET /api/roles` 暴露给前端。离线能力画像（含学到的 `quality` 与采纳统计）落在 `data/role_profiles.json`（`ROLE_PROFILE_STORE_BACKEND`），启动时以 `RoleSpec` 为种子建档并保留已学到的状态。前端 `WF_ROLES`（名称 / 图标 / 配色）只是展示用的一份拷贝。

**内联核对的通信**（原选择类角色的逻辑，现在在召回 agent 里，`roles/checks.py`）：每个召回角色探查完，按自己提供的评估维度对「本节点负责的指标在黑板上已有的候选 + 已知资产 seed + 依赖节点的产出」做核对：
- `check` 事件：核对完成后报告一次（`checked N` + 问题数）；`AgentInfo.checked / issues` 统计；
- 发现问题时**转交**给别的 agent 补查（走同一个 `delegate`，同样受开关 / 预算 / 去重限制，事件带 `via`）：
  - 新鲜度（BI / 全通道）：候选缺分区信息或粒度 → 转交 **BI**（按 id 重新解析元信息，`via=freshness:<id>`）；
  - 维度覆盖（血缘 / 全通道）：某个所需维度没有任何候选的字段（物理名或中文别名）覆盖 → 转交**血缘**，以现有候选为种子找带该维度的表（`via=dimension:<维度>`）；
  - 佐证核验（口径 / 全通道）：top 候选之间没有共同字段 → 转交**口径查找**（`via=crosscheck:<id>`）。
- seed（L0 / L1 已知资产）现在注入**所有**节点；L1（`select_only`）时召回角色不再召回，只有口径角色读知识全文，其余只核验 seed。
- 局限：ASAP 调度下核对只能看到核对时黑板上已有的候选，不再像原来的选择节点那样等全部召回节点结束。
