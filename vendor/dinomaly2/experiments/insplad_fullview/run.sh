#!/usr/bin/env bash
set -Eeuo pipefail
EXPERIMENT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY FTP_PROXY http_proxy https_proxy all_proxy ftp_proxy PIP_PROXY
export NO_PROXY='*' no_proxy='*' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TZ=Asia/Shanghai
exec /root/autodl-tmp/envs/dinomaly2/bin/python "$EXPERIMENT_DIR/run.py" "$@"
