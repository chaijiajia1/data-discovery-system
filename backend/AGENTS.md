# Data Discovery Backend · 开发约定

## 铁律
1. **阶段间只通过 `app/schemas/` 的 pydantic 契约通信**，不要在阶段间传裸 dict。改契约要同步改所有消费方与测试。
2. **阶段内不直接依赖外部系统**：一律经 `Deps`（`services/protocols.py`）注入，保证可用 fakes 单测。
3. **LLM 是可选增强**：任何用 LLM 的地方必须有规则兜底（捕获 `UpstreamError`），`LLM_*` 为空时流程可完整跑通。
4. **契约槽位必须可追溯**：② 补全的每个条件要写 `source` / `rule_id`。
5. **冲突显式保留**：④ 合并不得静默择一；⑥ 回滚只回滚最小可疑步骤，保留已验证上游。
6. **只有通过核验的结果才回流**：`stores` 只在 ⑥ 核验通过后写入新 pair。
7. 严格 TDD：新增/修改逻辑必须配套 `tests/`；运行 `make test`。

## 新增阶段 / 角色
- 阶段：继承 `stages.base.Stage`，在 `pipeline.PIPELINE` 注册事件名与 ctx 属性。
- 角色：`roles/` 下实现 `Role`，注册到 `roles.REGISTRY`。
