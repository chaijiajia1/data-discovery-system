#!/usr/bin/env bash
# 构建演示镜像（在有 Docker 的开发机 / CI 上运行；需要 buildx，Docker Desktop 自带）。
#   ./build-image.sh                    # 构建 linux/amd64（绝大多数云服务器）
#   PLATFORM=linux/arm64 ./build-image.sh   # 目标服务器是 ARM（如 Graviton / 鲲鹏）
# 注意：Apple Silicon 的 Mac 默认构建出 arm64 镜像，拷到 x86 服务器会报 "exec format error"——所以这里显式指定平台。
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE="${DD_IMAGE:-dd-demo:latest}"
PLATFORM="${PLATFORM:-linux/amd64}"
command -v docker >/dev/null || { echo "docker not found" >&2; exit 1; }
docker buildx version >/dev/null 2>&1 || { echo "docker buildx not available (upgrade Docker, or: docker build --platform $PLATFORM ...)" >&2; exit 1; }
echo "building $IMAGE for $PLATFORM ..."
docker buildx build --platform "$PLATFORM" --load -f deploy/Dockerfile -t "$IMAGE" \
  --build-arg "PYTHON_IMAGE=${PYTHON_IMAGE:-python:3.11-slim}" --build-arg "PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.org/simple}" .
echo "built: $IMAGE ($PLATFORM)"
