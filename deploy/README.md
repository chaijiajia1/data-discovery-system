# 外网演示部署（假数据，完全公开）

单容器同源提供前端页面 + API；可选在前面放 Caddy（自动 HTTPS）。演示模式（`DEMO_MODE=true`）下：

- 只使用内置假数据，不连内网、不调 LLM、不读/存 cookie、不展示 Cookie 控件；
- `/api/cookie*`、`/api/me`、`/api/pairs`、`/api/prompts*`、`/docs`、`/redoc`、`/openapi.json` 全部 404（应用与 Caddy 各拦一次）；
- `/api/stage*`、`/api/discover`、`/api/plan*` 按客户端 IP 限流（默认每分钟 30 次、突发 15）；问句上限 1000 字符，请求体上限 64KB；
- 页面已去掉遥测脚本；种子历史只读，公开请求产生的配对只存内存，重启即恢复；
- 必须单实例单 worker（run 缓存在进程内存）。

前提：服务器已装 Docker + compose 插件。

## 三种部署方式

| 方式 | 服务器需要 | 命令 |
|---|---|---|
| A. 离线包（推荐，换服务器最省事） | 只要 Docker + compose 插件 | 本机 `./pack.sh` → 拷包 → 服务器 `./install.sh http` |
| B. 服务器上直接构建 | Docker + compose + 能访问 Docker Hub / PyPI（或镜像源）+ 整个仓库 | `cd deploy && ./install.sh http` |
| C. 手动 compose / `docker run` | 同 B | 见下 |

两种访问形态（任选其一，`install.sh` 的参数）：

- `http`：只起 app，`http://<服务器IP>:8080`。没有域名 / 内网 / 先验证时用。**没有反向代理**，所以自动设 `DEMO_TRUST_PROXY=false`（限流只认连接对端地址，`X-Forwarded-For` 不可信）。
- `https`：app + Caddy 自动申请证书，需要域名（A 记录指向本机）并放行 80/443（含 UDP 443）。先在 `.env` 里把 `DOMAIN` 改成真实域名。

### A. 离线包

```bash
# 开发机（有 Docker + buildx，能联网）
cd deploy
./pack.sh                                  # 默认 linux/amd64；ARM 服务器：PLATFORM=linux/arm64 ./pack.sh
scp dist/dd-demo-bundle.tar.gz user@server:~

# 目标服务器
tar xzf dd-demo-bundle.tar.gz && cd dd-demo-bundle
./install.sh http                          # 或 ./install.sh https（先改 .env 里的 DOMAIN）
```

`install.sh` 会：`docker load` 镜像 → 没有 `demo.env` / `.env` 就从示例复制 → `docker compose up -d` → 等健康检查 → 跑 `smoke.sh`。停止：`./install.sh down`。

> **Apple Silicon（M 系列 Mac）注意**：默认构建出的是 arm64 镜像，拷到 x86 服务器会报 `exec format error`。`build-image.sh` 因此显式指定平台（默认 `linux/amd64`），不要绕过它直接 `docker build`。

### B / C. 在服务器上构建 / 手动

```bash
git clone <repo> && cd <repo>/deploy
./install.sh http                          # 等价于下面这条（叠加 docker-compose.build.yml 以便本机构建）
docker compose -f docker-compose.http.yml -f docker-compose.build.yml up -d --build

# 只想看一眼、不用 compose：镜像内置了全部演示默认值
docker build -f deploy/Dockerfile -t dd-demo .        # 在仓库根目录执行
docker run --rm -p 8080:8080 -e DEMO_TRUST_PROXY=false dd-demo
```

更新：`git pull && ./install.sh http`（或重新 `./pack.sh` 拷新包）。日志：`docker compose -f docker-compose.http.yml logs -f app`。

### 配置文件

| 文件 | 作用 |
|---|---|
| `.env`（`.env.example`） | compose 变量：`DOMAIN` / `APP_PORT` / `APP_BIND` / `DD_IMAGE` / 镜像源 |
| `demo.env`（`demo.env.example`） | 传给应用的环境变量（限流等）；镜像里已内置默认值，不改也能跑 |
| `requirements.lock` | 镜像里的依赖版本锁，保证不同服务器、不同时间构建结果一致 |

更新依赖锁（需要 Python 3.11）：

```bash
python3.11 -m venv /tmp/lockvenv && grep -vi '^browser-cookie3' ../backend/requirements.txt > /tmp/req.txt
/tmp/lockvenv/bin/pip install -r /tmp/req.txt && /tmp/lockvenv/bin/pip freeze > requirements.lock
```

## 示例问句与演示数据

三条问句共用合成目录。月度 CTR 归档表和受限财务审计表具有较高的检索相关度，逐指标选择还会检查粒度与权限。各例的默认 K、两种方法的命中情况及操作步骤见[工程首页](../README.md)。

## 验证状态（如实说明）

- 已验证：全量 pytest（含 `tests/test_demo_mode.py`、`tests/test_demo_data.py`）在**仅按 `requirements.lock` 安装的干净 venv** 里通过；
  按镜像的目录布局（只含 app / prompts / demo_history / index.html）和镜像内置的环境变量起 uvicorn，`smoke.sh` 全部通过；脚本 `bash -n` 通过，三个 compose 文件 YAML 可解析。
- **未验证**：Docker 镜像构建、compose 网络 / 只读文件系统、Caddy 证书与代理、`pack.sh` / `install.sh` 的实际执行——开发机没有 Docker。
  首次部署后请看 `install.sh` 末尾的冒烟结果和 `docker compose logs`。若 `read_only` 或 `internal` 网络与你的环境冲突，可先去掉这两项排查。
- 限流依赖 Caddy 写入的 `X-Forwarded-For`（https 模式）；http 模式已自动改为只认连接地址。https 模式下不要把 app 的 8080 直接暴露到公网。

## 网络受限（服务器在中国大陆 / 连不上 Docker Hub、PyPI）

症状：`registry-1.docker.io ... connection refused`（DNS 被污染，解析到非 Docker Hub 地址）。处理：

1. 给 Docker 配镜像加速器（云厂商控制台里的加速地址优先），`/etc/docker/daemon.json`：
   `{ "registry-mirrors": ["https://<加速地址>"] }`，然后 `sudo systemctl restart docker`。
2. 或在 `deploy/.env` 指定镜像源（前缀换成你能访问的）：
   ```
   PYTHON_IMAGE=<镜像站>/library/python:3.11-slim
   CADDY_IMAGE=<镜像站>/library/caddy:2
   PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
   ```
3. 大陆服务器用域名对外提供 80/443 需要 ICP 备案；未备案可改用香港/海外机房。
4. 无证书条件时，`.env` 里设 `DOMAIN=:80` 先走 HTTP 验证服务。
