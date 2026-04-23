#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIGS=(
    qwen_baseline
    qwen_ngram
    qwen_eagle3
    avibe_baseline
    avibe_ngram
    avibe_eagle3
)

for cfg in "${CONFIGS[@]}"; do
    echo "============================================================"
    echo ">>> running $cfg"
    echo "============================================================"
    python runner.py --config "$cfg"
    echo ">>> $cfg done; sleeping 5s to let GPU drain"
    sleep 5
done

echo "============================================================"
echo ">>> all runs done — analyzing"
echo "============================================================"
python analyze.py
