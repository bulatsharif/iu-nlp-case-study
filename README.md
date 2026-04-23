# Speculative Decoding Case Study

## Speculative Decoding Evaluation

Per-prompt wall-clock comparison of three decoding methods — vanilla autoregressive, n-gram prompt-lookup (PLD), and EAGLE-3 — against Qwen3-8B on MT-Bench (EN) and A-Vibe on ru-arena-hard (RU). 

### Configs

| Name | Target | Drafter | Dataset |
|---|---|---|---|
| `qwen_baseline` | Qwen3-8B (bf16) | — | MT-Bench |
| `qwen_ngram`    | Qwen3-8B (bf16) | n-gram (PLD, min=2, max=4) | MT-Bench |
| `qwen_eagle3`   | Qwen3-8B (bf16) | `AngelSlim/Qwen3-8B_eagle3` | MT-Bench |
| `avibe_baseline`| A-Vibe (fp16)   | — | ru-arena-hard |
| `avibe_ngram`   | A-Vibe (fp16)   | n-gram (PLD, min=2, max=4) | ru-arena-hard |
| `avibe_eagle3`  | A-Vibe (fp16)   | `AvitoTech/avibe-eagle` | ru-arena-hard |

All: `batch=1`, `temperature=0`, `max_tokens=512`, `min_tokens=128`, `num_speculative_tokens=5`. First 3 prompts per config are warmup and discarded.

### Setup

```bash
pip install -r requirements.txt
```

vLLM must be `>=0.10` built against CUDA 12.8+ for Blackwell (RTX 5090). If flash-attn fails to import on Blackwell, fall back to xFormers:

```bash
export VLLM_ATTENTION_BACKEND=XFORMERS
```

### Run everything

```bash
bash scripts/run_all.sh
```

This runs all six configs sequentially (5 s GPU drain between them) and then invokes `python analyze.py`. A single config:

```bash
bash scripts/run_one.sh qwen_eagle3
```

### Outputs

- `results/<config>.csv` — per-prompt `wall_s`, `out_tokens`, `tok_per_s`. One row per prompt, flushed after every generation, so a crash 40 prompts in isn't a total loss.
- `plots/speedup_by_domain.png` — **speedup by prompt category**. MT-Bench tags each prompt with a category (writing, reasoning, coding, math, extraction, stem, humanities, roleplay); ru-arena-hard has an analogous `cluster` field. This plot breaks speedup down along that axis, so you can see where spec decoding actually helps — e.g. EAGLE-3 typically wins on coding/math (predictable tokens) but offers less on creative writing. Two subplots (Qwen3-8B, A-Vibe), grouped bars (n-gram vs EAGLE-3), baseline at y=1.0, std error bars across prompts.
- `plots/throughput.png` — aggregate tokens/sec per config (6 bars), colored by method family. The "how much faster overall" view.

## Training experiment 

A second experiment trains an EAGLE3 draft module on `unsloth/Llama-3.2-1B` (ungated mirror of `meta-llama/Llama-3.2-1B`) using 20k Daring-Anteater samples for 1 epoch, then benchmarks it on MT-Bench only.

```bash
bash scripts/run_training.sh
```

Pipeline: `train_eagle.py` → `eval_trained.py --method baseline` → `eval_trained.py --method eagle3` → `analyze_trained.py`.


Outputs:
- `eagle_out/` — HF artefact from the Trainer (composed model + tokenizer).
- `eagle_hf_ckpt/` — **unified HF checkpoint** consumed by vLLM's `speculative_config`.
- `results/train_loss.json`, `plots/train_loss.png` — per-step training loss.
- `results/llama32_baseline.csv`, `results/llama32_eagle3.csv` — per-prompt timing on MT-Bench.
- `plots/training_speedup.png` — per-category speedup of trained EAGLE3 vs. baseline.
