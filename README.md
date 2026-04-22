# Speculative Decoding Comparison

Per-prompt wall-clock comparison of three decoding methods — vanilla autoregressive, n-gram prompt-lookup (PLD), and EAGLE-3 — against Qwen3-8B on MT-Bench (EN) and A-Vibe on ru-arena-hard (RU). 

## Configs

| Name | Target | Drafter | Dataset |
|---|---|---|---|
| `qwen_baseline` | Qwen3-8B (bf16) | — | MT-Bench |
| `qwen_ngram`    | Qwen3-8B (bf16) | n-gram (PLD, min=2, max=4) | MT-Bench |
| `qwen_eagle3`   | Qwen3-8B (bf16) | `AngelSlim/Qwen3-8B_eagle3` | MT-Bench |
| `avibe_baseline`| A-Vibe (fp16)   | — | ru-arena-hard |
| `avibe_ngram`   | A-Vibe (fp16)   | n-gram (PLD, min=2, max=4) | ru-arena-hard |
| `avibe_eagle3`  | A-Vibe (fp16)   | `AvitoTech/avibe-eagle` | ru-arena-hard |

All: `batch=1`, `temperature=0`, `max_tokens=512`, `min_tokens=128`, `num_speculative_tokens=5`. First 3 prompts per config are warmup and discarded.

## Setup

```bash
pip install -r requirements.txt
```

vLLM must be `>=0.10` built against CUDA 12.8+ for Blackwell (RTX 5090). If flash-attn fails to import on Blackwell, fall back to xFormers:

```bash
export VLLM_ATTENTION_BACKEND=XFORMERS
```

## Run everything

```bash
bash scripts/run_all.sh
```

This runs all six configs sequentially (5 s GPU drain between them) and then invokes `python analyze.py`. A single config:

```bash
bash scripts/run_one.sh qwen_eagle3
```

## Outputs

- `results/<config>.csv` — per-prompt `wall_s`, `out_tokens`, `tok_per_s` 
- `plots/speedup_by_domain.png` — two subplots (Qwen3-8B, A-Vibe), grouped bars of n-gram vs EAGLE-3 speedup vs baseline by category, horizontal line at y=1.0, std error bars.
- `plots/throughput.png` — tokens/sec per config, colored by method family.

`analyze.py` aligns each spec-method row to its baseline by `(target, prompt_idx)` before taking ratios — averaging tok/s first would hide variance.

## Training sub-experiment

See [`training/`](training/) — trains an EAGLE3 draft module on `meta-llama/Llama-3.2-1B` using 20k Daring-Anteater samples for 1 epoch, then benchmarks it on MT-Bench only. Self-contained: `bash training/run_experiment.sh` trains → evaluates baseline → evaluates trained EAGLE3 → plots speedup.
