#!/usr/bin/env bash
# 功能：一键用官方 flashinfer-bench CLI 跑 GDN decode Modal B200 benchmark。
# 参数：原样透传给 run_modal_official_cli.py，例如 --probe、--max-workloads 0。
# 示例：
# smoke test, 用于快速验证是否能跑通:./run_modal_official_cli.sh --max-workloads 1 --warmup-runs 1 --iterations 10 --num-trials 1
# AKO4X 对照 smoke test: ./run_modal_official_cli.sh --solution ako --max-workloads 1 --warmup-runs 1 --iterations 10 --num-trials 1
# 正式评估，跑所有 workloads，默认使用 FlashInfer starter kit timing，timeout 保留 7200: ./run_modal_official_cli.sh
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
