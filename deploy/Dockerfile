# 外网演示镜像：只含后端代码 + 提示词 + 英文种子数据 + 前端页面；不含 .env、真实数据、浏览器 cookie 依赖。
# 构建上下文 = 仓库根目录（见 docker-compose.yml）。
# 服务器连不上 Docker Hub / PyPI 时，通过 build args 换成可访问的镜像源（见 README「网络受限」）
ARG PYTHON_IMAGE=python:3.11-slim
ARG PIP_INDEX_URL=https://pypi.org/simple
FROM ${PYTHON_IMAGE}
ARG PIP_INDEX_URL

ENV PIP_INDEX_URL=${PIP_INDEX_URL} PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# browser-cookie3 只用于读本机浏览器 cookie（本地开发的 Sync cookie），演示镜像不需要，去掉可少装一批依赖
COPY backend/requirements.txt /tmp/requirements.txt
RUN grep -vi '^browser-cookie3' /tmp/requirements.txt > /tmp/requirements.demo.txt \
 && pip install -r /tmp/requirements.demo.txt && rm /tmp/requirements*.txt

COPY backend/app ./app
COPY backend/prompts ./prompts
COPY backend/data/demo_history.jsonl ./data/demo_history.jsonl
COPY ["Data Discovery System · Interactive Walkthrough.html", "./ui/index.html"]

# 非 root、只读文件系统运行（compose 里设 read_only）
RUN useradd --system --no-create-home --uid 10001 app
USER 10001

ENV HOST=0.0.0.0 PORT=8080 LOG_LEVEL=info
EXPOSE 8080

# 单进程单 worker：run 缓存在进程内存里，多 worker / 多副本会让 Phase 4 与 Phase 5 的候选池不一致
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--no-server-header"]
