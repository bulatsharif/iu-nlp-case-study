#!/usr/bin/env bash
# Batch-size sweep on Qwen3-8B across {baseline, ngram, eagle3} × {1,2,4,8,16}.
set -euo pipefail

cd "$(dirname "$0")/.."

# Activate project venv if present (matches setup of other scripts).
if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

BATCH_SIZES="${BATCH_SIZES:-1,2,4,8,16}"
METHODS=(baseline ngram eagle3)

for m in "${METHODS[@]}"; do
    echo "============================================================"
    echo ">>> batch sweep: method=$m batch_sizes=$BATCH_SIZES"
    echo "============================================================"
    python runner_batch.py --method "$m" --batch-sizes "$BATCH_SIZES"
    echo ">>> $m done; sleeping 5s to let GPU drain"
    sleep 5
done

echo "============================================================"
echo ">>> all batch runs done — analyzing"
echo "============================================================"
python analyze_batch.py
