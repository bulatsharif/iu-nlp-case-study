#!/usr/bin/env bash
# Run a single config by name. Usage: bash scripts/run_one.sh qwen_eagle3
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <config_name>"
    exit 2
fi

python runner.py --config "$1"
