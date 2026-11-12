#!/usr/bin/env bash
set -euo pipefail
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY PIP_PROXY
export no_proxy='*' NO_PROXY='*' PIP_CONFIG_FILE=/dev/null
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export TMPDIR=/root/autodl-tmp/costfilter-setup/tmp
mkdir -p "$TMPDIR"
export PYTHONDONTWRITEBYTECODE=1
TASK_PYTHON=/root/autodl-tmp/envs/costfilter_ad/bin/python
if [[ ! -x "$TASK_PYTHON" ]]; then
  echo "缺少环境：$TASK_PYTHON" >&2; exit 1
fi
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$TASK_PYTHON" -u "$HERE/run.py" "$@"
