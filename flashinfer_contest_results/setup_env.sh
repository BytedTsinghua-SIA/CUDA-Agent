#!/usr/bin/env bash
# 功能：从零准备 FlashInfer contest result 复现实验环境。
# 参数：
#   FORCE=1        强制重新从 Hugging Face 拉取缓存文件。
#   HF_TOKEN=...   如 Hugging Face 下载需要鉴权，可传入 token。
# 示例：
#   cd flashinfer_contest_results
#   ./setup_env.sh
#   FORCE=1 ./setup_env.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install -U pip
"$VENV_DIR/bin/python" -m pip install -r "$SCRIPT_DIR/requirements.txt"
"$VENV_DIR/bin/python" "$SCRIPT_DIR/patch_modal_proxy.py"
"$VENV_DIR/bin/python" "$SCRIPT_DIR/setup_env.py"

echo
echo "Environment ready."
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Authenticate Modal with: $VENV_DIR/bin/modal setup"
