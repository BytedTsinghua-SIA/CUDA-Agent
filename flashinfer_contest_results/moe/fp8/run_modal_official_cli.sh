#!/usr/bin/env bash
# 功能：一键用官方 flashinfer-bench CLI 跑 MoE FP8 Modal B200 benchmark。
# 参数：原样透传给 run_modal_official_cli.py，例如 --probe、--max-workloads 0。
# 默认跑 FlashInfer starter kit timing：全量 workloads + live FlashInfer baseline + 10/50/3 timing，timeout 保留 900。
# 示例：./run_modal_official_cli.sh
# AKO4X 对照使用 run_modal_official_cli_any_solution.py。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODAL_BIN="${MODAL_BIN:-$RESULTS_ROOT/.venv/bin/modal}"

if [[ ! -x "$MODAL_BIN" ]]; then
  if command -v modal >/dev/null 2>&1; then
    MODAL_BIN="$(command -v modal)"
  else
    echo "modal not found. Run: cd $RESULTS_ROOT && python3 -m venv .venv && source .venv/bin/activate && python3 -m pip install -r requirements.txt" >&2
    exit 127
  fi
fi

cd "$SCRIPT_DIR"
mkdir -p logs
LOG_FILE="logs/run_modal_official_cli_$(date +%Y%m%d_%H%M%S).log"
echo "logging to $SCRIPT_DIR/$LOG_FILE"

set +e
"$MODAL_BIN" run run_modal_official_cli.py "$@" 2>&1 | tee "$LOG_FILE"
status=${PIPESTATUS[0]}
set -e
exit "$status"
