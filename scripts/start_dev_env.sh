#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

CONTAINER_NAME="efficientsam3-dev"

# Build and start the container in detached mode.
docker compose up -d --build

if ! command -v terminator >/dev/null 2>&1; then
  echo "[WARN] terminator is not installed. Container is running; attach with:"
  echo "  docker exec -it ${CONTAINER_NAME} bash"
  exit 0
fi

# Open Terminator with three tabs:
# 1) interactive shell
# 2) nvidia-smi monitor
# 3) benchmark helper prompt
terminator \
  --new-tab -x bash -lc "docker exec -it ${CONTAINER_NAME} bash" \
  --new-tab -x bash -lc "docker exec -it ${CONTAINER_NAME} bash -lc 'watch -n 1 nvidia-smi'" \
  --new-tab -x bash -lc "docker exec -it ${CONTAINER_NAME} bash -lc 'echo Ready for benchmark; bash'"
