#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "============================================================"
echo ">>> step 1/5: train EAGLE3 draft (Llama-3.2-1B-Instruct, 80k UltraChat-200k, 1 epoch)"
echo "============================================================"
python train_eagle.py

sleep 5

echo "============================================================"
echo ">>> step 2/5: evaluate baseline Llama-3.2-1B-Instruct on MT-Bench"
echo "============================================================"
python eval_trained.py --method baseline

sleep 5

echo "============================================================"
echo ">>> step 3/5: evaluate Llama-3.2-1B-Instruct + n-gram (PLD) on MT-Bench"
echo "============================================================"
python eval_trained.py --method ngram

sleep 5

echo "============================================================"
echo ">>> step 4/5: evaluate trained EAGLE3 on MT-Bench"
echo "============================================================"
python eval_trained.py --method eagle3

echo "============================================================"
echo ">>> step 5/5: analyze + speedup plot"
echo "============================================================"
python analyze_trained.py
