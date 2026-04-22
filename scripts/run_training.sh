#!/usr/bin/env bash
# End-to-end training experiment: train EAGLE3 draft → baseline eval → trained eval → analyze.
# Single RTX 5090, MT-Bench only.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "============================================================"
echo ">>> step 1/4: train EAGLE3 draft (Llama-3.2-1B, 20k samples, 1 epoch)"
echo "============================================================"
python train_eagle.py

sleep 5

echo "============================================================"
echo ">>> step 2/4: evaluate baseline Llama-3.2-1B on MT-Bench"
echo "============================================================"
python eval_trained.py --method baseline

sleep 5

echo "============================================================"
echo ">>> step 3/4: evaluate trained EAGLE3 on MT-Bench"
echo "============================================================"
python eval_trained.py --method eagle3

echo "============================================================"
echo ">>> step 4/4: analyze + speedup plot"
echo "============================================================"
python analyze_trained.py
