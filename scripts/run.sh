#!/usr/bin/env bash
set -Eeuo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY FTP_PROXY http_proxy https_proxy all_proxy ftp_proxy PIP_PROXY
export NO_PROXY='*' no_proxy='*' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TZ=Asia/Shanghai
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
ACTION="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$ACTION" in
  a-train|ncut-evaluate)
    TASK_PYTHON="${PYTHON_BIN:-/root/autodl-tmp/envs/dinomaly2/bin/python}"
    if [[ "$ACTION" == a-train ]]; then ENTRY=train.py; else ENTRY=evaluate.py; fi
    exec "$TASK_PYTHON" -u "$HERE/$ENTRY" "$@" ;;
  costfilter-train|costfilter-evaluate|costfilter-check|costfilter-smoke)
    TASK_PYTHON="${COSTFILTER_PYTHON_BIN:-/root/autodl-tmp/envs/costfilter_ad/bin/python}"
    exec "$TASK_PYTHON" -u "$HERE/costfilter.py" "${ACTION#costfilter-}" "$@" ;;
  check)
    exec "${PYTHON_BIN:-/root/autodl-tmp/envs/dinomaly2/bin/python}" -u "$HERE/check.py" "$@" ;;
  verify)
    exec "${VERIFY_PYTHON_BIN:-python}" "$HERE/verify_results.py" "$@" ;;
  *)
    echo 'Usage: bash scripts/run.sh {check|verify|a-train|ncut-evaluate|costfilter-check|costfilter-smoke|costfilter-train|costfilter-evaluate} [options]'
    [[ "$ACTION" == help ]] ;;
esac
