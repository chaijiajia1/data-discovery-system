#!/usr/bin/env bash
# 在服务器上一键启动。可在两种目录里运行：离线包解压目录（带镜像 tar），或仓库的 deploy/ 目录（本机构建）。
#   ./install.sh http      # 只起 app，http://<IP>:8080（默认）
#   ./install.sh https     # app + Caddy 自动 HTTPS，需要先在 .env 里设 DOMAIN
#   ./install.sh down      # 停止（保留证书卷）
set -euo pipefail
cd "$(dirname "$0")"
MODE="${1:-http}"
need() { command -v "$1" >/dev/null 2>&1 || { echo "missing: $1" >&2; exit 1; }; }
need docker
docker compose version >/dev/null 2>&1 || { echo "docker compose plugin not found (install docker-compose-plugin)" >&2; exit 1; }

case "$MODE" in
  http)  FILE=docker-compose.http.yml ;;
  https) FILE=docker-compose.yml ;;
  down)  for f in docker-compose.http.yml docker-compose.yml; do docker compose -f "$f" down 2>/dev/null || true; done; exit 0 ;;
  *) echo "usage: $0 [http|https|down]" >&2; exit 2 ;;
esac

FILES=(-f "$FILE")
[[ -f demo.env ]] || cp demo.env.example demo.env
[[ -f .env ]] || cp .env.example .env
if [[ "$MODE" == https ]] && { ! grep -Eq '^DOMAIN=[^ #]+' .env || grep -Eq '^DOMAIN=demo\.example\.com' .env; }; then echo "set DOMAIN in .env to your real domain first" >&2; exit 1; fi
set -a; . ./.env; set +a

if [[ -f dd-demo-image.tar.gz ]]; then
  echo "loading image ..."; gunzip -c dd-demo-image.tar.gz | docker load
  BUILD=(--no-build)                                  # 离线包：镜像已经在本机，不构建
elif [[ -f Dockerfile && -f docker-compose.build.yml ]]; then
  FILES+=(-f docker-compose.build.yml); BUILD=(--build)   # 仓库 deploy/ 目录：本机构建
else
  BUILD=(--no-build)
fi

# 另一种模式的旧容器会占用 80/443 / 8080，先停掉
for f in docker-compose.http.yml docker-compose.yml; do [[ "$f" == "$FILE" ]] || docker compose -f "$f" down 2>/dev/null || true; done
docker compose "${FILES[@]}" up -d "${BUILD[@]}"

if [[ "$MODE" == http ]]; then
  URL="http://127.0.0.1:${APP_PORT:-8080}"
  echo -n "waiting for $URL/api/health "
  for _ in $(seq 1 60); do
    if curl -fsS -m 3 "$URL/api/health" >/dev/null 2>&1; then echo " up"; break; fi
    echo -n "."; sleep 2
  done
  curl -fsS -m 3 "$URL/api/health" >/dev/null 2>&1 || { echo; echo "app did not become healthy; docker compose -f $FILE logs app" >&2; exit 1; }
  if command -v python3 >/dev/null && command -v curl >/dev/null; then ./smoke.sh "$URL"; fi
  echo "open: http://<server-ip>:${APP_PORT:-8080}/   (open the port in the cloud firewall / security group)"
else
  echo "started. check: docker compose -f $FILE ps ; ./smoke.sh https://${DOMAIN}"
fi
