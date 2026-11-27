# Data Discovery System

这个交互演示从业务问题出发，展示知识检索、工作流规划、候选召回和资产选择。页面内置三个英文问句，使用合成资产、查询历史和参考答案。

## 运行演示

解压演示源码包后，在包含本文件的目录执行以下命令。需要 Python 3.11，首次安装依赖需要联网。

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r deploy/requirements.lock
cd backend
DEMO_MODE=true PAIR_STORE_BACKEND=jsonl_ro PAIR_STORE_PATH=data/demo_history.jsonl \
  ../.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8080
```

打开 [本地演示](http://127.0.0.1:8080/)，选择示例并点击 **Run**。前两个案例进入规划阶段后，点击 **Submit workflow** 继续。最后在 **Selection** 查看两种方法的结果，展开 **Synthetic reference** 可逐个核对指标与资产。Docker 用户也可以在工程根目录运行 `./deploy/install.sh http`。[部署说明](deploy/README.md)

## 三组对照

两种方法使用同一次运行的候选池。Traditional baseline 按检索分降序排列，平分时按资产 ID 排序。默认 K 等于 Proposed 选出的主资产数。Proposed 逐个指标核对字段、粒度、权限和上游证据。下表按默认工作流和 K 统计，每个分子是选中参考资产的指标数，分母是问题中的指标数。

| 示例 | K | Proposed | Baseline | 差异来自哪里 |
| --- | ---: | ---: | ---: | --- |
| 近 7 天按平台看 DAU 和 CTR | 2 | 2/2 | 1/2 | 两者都找到日活表。Baseline 还选了常被引用的 CTR 月表，Proposed 为 CTR 选了按天、按平台的广告表。 |
| 近 30 天按平台看 CTR | 1 | 1/1 | 0/1 | 按自然月汇总的 CTR 表检索分最高，但无法还原滚动 30 天的窗口。Proposed 选择带点击数和曝光数的日表。 |
| 近 7 天按渠道看 DAU 和 GMV | 2 | 2/2 | 1/2 | Baseline 选到需要财务权限的审计表。Proposed 将其列为需要申请权限的备选，选择可访问的交易日表。 |

这些结果描述三个特意构造的案例。可以在 Baseline 面板调整 K，观察扩大候选范围后参考资产何时进入结果。资产与知识定义在 `backend/app/services/demo_data.py`，六条合成历史在 `backend/data/demo_history.jsonl`，独立的参考答案在 `backend/app/demo.py`。
