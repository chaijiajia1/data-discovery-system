#!/usr/bin/env bash
# 打包成一个可以直接拷到任意服务器的离线包：镜像 + compose + 启动脚本 + 冒烟测试。服务器只需要 Docker（+ compose 插件），不需要源码、不需要联网构建。
#   ./pack.sh                 # → deploy/dist/dd-demo-bundle.tar.gz
#   scp dist/dd-demo-bundle.tar.gz user@server:~ ; ssh user@server
#   tar xzf dd-demo-bundle.tar.gz && cd dd-demo-bundle && ./install.sh http
set -euo pipefail
cd "$(dirname "$0")"
IMAGE="${DD_IMAGE:-dd-demo:latest}"
OUT="dist/dd-demo-bundle"
rm -rf dist && mkdir -p "$OUT"
./build-image.sh
echo "saving image ..."
docker save "$IMAGE" | gzip -9 > "$OUT/dd-demo-image.tar.gz"
cp docker-compose.yml docker-compose.http.yml Caddyfile demo.env.example .env.example smoke.sh install.sh README.md "$OUT/"
tar -C dist -czf dist/dd-demo-bundle.tar.gz dd-demo-bundle
ls -lh dist/dd-demo-bundle.tar.gz
echo "next: scp dist/dd-demo-bundle.tar.gz <server>:~   then on the server:  tar xzf dd-demo-bundle.tar.gz && cd dd-demo-bundle && ./install.sh http"
