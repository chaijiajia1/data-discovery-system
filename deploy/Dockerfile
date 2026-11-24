# 外网演示镜像：只含后端代码 + 提示词 + 英文种子数据 + 前端页面；不含 .env、真实数据、浏览器 cookie 依赖。
# 构建上下文 = 仓库根目录（见 docker-compose.yml / build-image.sh）。
# 镜像自带演示模式的全部默认环境变量：不挂 demo.env 也能直接 `docker run -p 8080:8080 <image>` 跑起来。
# 服务器连不上 Docker Hub / PyPI 时，通过 build args 换成可访问的镜像源（见 README「网络受限」），
# 或者在能联网的机器上构建好再拷过去（见 README「离线 / 跨机器部署」）。
ARG PYTHON_IMAGE=python:3.11-slim
ARG PIP_INDEX_URL=https://pypi.org/simple
FROM ${PYTHON_IMAGE}
ARG PIP_INDEX_URL

LABEL org.opencontainers.image.title="data-discovery-demo" \
      org.opencontainers.image.description="Data Discovery System walkthrough (demo mode, fake data only)"

ENV PIP_INDEX_URL=${PIP_INDEX_URL} PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# 依赖版本锁在 deploy/requirements.lock（不含 browser-cookie3：它只用于读本机浏览器 cookie，演示镜像不需要）。
# 不同时间 / 不同服务器构建出的镜像因此一致；升级依赖见 README「更新依赖锁」。
COPY deploy/requirements.lock /tmp/requirements.lock
RUN pip install --retries 5 --timeout 60 -r /tmp/requirements.lock && rm /tmp/requirements.lock

COPY backend/app ./app
COPY backend/prompts ./prompts
COPY backend/data/demo_history.jsonl ./data/demo_history.jsonl
COPY ["Data Discovery System · Interactive Walkthrough.html", "./ui/index.html"]

# 非 root、只读文件系统运行（compose 里设 read_only）
RUN useradd --system --no-create-home --uid 10001 app
USER 10001

# 演示模式默认值（demo.env 里的同名变量会覆盖它们）
ENV DEMO_MODE=true \
    PAIR_STORE_BACKEND=jsonl_ro \
    PAIR_STORE_PATH=/app/data/demo_history.jsonl \
    UI_HTML_PATH=/app/ui/index.html \
    HOST=0.0.0.0 PORT=8080 LOG_LEVEL=info
EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=5 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/api/health', timeout=3).status == 200 else 1)"]

# 单进程单 worker：run 缓存在进程内存里，多 worker / 多副本会让 Phase 4 与 Phase 5 的候选池不一致
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--no-server-header"]
