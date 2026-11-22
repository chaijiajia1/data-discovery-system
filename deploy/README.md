# 外网演示部署（假数据，完全公开）

单容器同源提供前端页面 + API，前面是 Caddy（自动 HTTPS）。演示模式（`DEMO_MODE=true`）下：

- 只使用内置假数据，不连内网、不调 LLM、不读/存 cookie、不展示 Cookie 控件；
- `/api/cookie*`、`/api/me`、`/api/pairs`、`/api/prompts*`、`/docs`、`/redoc`、`/openapi.json` 全部 404（应用与 Caddy 各拦一次）；
- `/api/stage*`、`/api/discover`、`/api/plan*` 按客户端 IP 限流（默认每分钟 30 次、突发 15）；问句上限 1000 字符，请求体上限 64KB；
- 页面已去掉遥测脚本；种子历史只读，公开请求产生的配对只存内存，重启即恢复；
- 必须单实例单 worker（run 缓存在进程内存）。

## 部署步骤

前提：服务器已装 Docker + compose 插件；域名 A 记录指向服务器；放行 80/443（含 UDP 443）。

```bash
cd deploy
cp demo.env.example demo.env          # 无密钥，可直接用；按需改限流
echo "DOMAIN=demo.example.com" > .env # 你的域名
docker compose up -d --build
./smoke.sh https://demo.example.com   # 冒烟：健康检查、页面、屏蔽路径、3 条示例问句、超长输入
```

更新：`git pull && docker compose up -d --build`。日志：`docker compose logs -f app`。

## 示例问句（均已验证每个指标都能选出主资产）

- Last 7 days DAU and CTR by platform
- Last 30 days CTR by platform
- Last 7 days DAU and GMV by channel

## 验证状态（如实说明）

- 已验证：全量 pytest（含 `tests/test_demo_mode.py`）；用与 `demo.env` 相同的环境变量起本地 uvicorn，`smoke.sh` 全部通过。
- **未验证**：Docker 镜像构建、compose 网络/只读文件系统、Caddy 证书与代理配置——开发机没有 Docker。首次部署后请先跑 `smoke.sh`，
  再看 `docker compose logs`。若 `read_only` 或 `internal` 网络与你的环境冲突，可先去掉这两项排查。
- 限流依赖 Caddy 写入的 `X-Forwarded-For`；不要把 app 的 8080 直接暴露到公网。

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
