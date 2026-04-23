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
- `plots/speedup_by_domain.png` — **speedup by MT-Bench category for Qwen3-8B only** (A-Vibe is omitted: ru-arena clusters are not summarized in this figure). n-gram vs EAGLE-3, baseline at 1.0, std error bars. Rebuild with `python analyze.py`.
- `plots/throughput.png` — aggregate tokens/sec per config (6 bars), colored by method family. The "how much faster overall" view.

## Training experiment

Trains an EAGLE-3 draft head for `unsloth/Llama-3.2-1B-Instruct` (ungated mirror of `meta-llama/Llama-3.2-1B-Instruct`) on 80k samples from `HuggingFaceH4/ultrachat_200k` for 1 epoch, then benchmarks it against both the vanilla baseline and vLLM's built-in n-gram prompt-lookup on MT-Bench.

Key choices:

- **Instruct target, not base.** An earlier iteration used the pretrained base model as target; on MT-Bench prompts it produced incoherent continuations, so the draft (trained on chat-style text) had near-zero acceptance and EAGLE-3 ran at 0.75× baseline. Switching to the Instruct variant aligns the target's output distribution with what the draft was trained to predict. SpecForge / the EAGLE-3 paper both use `-Instruct` targets.
- **UltraChat-200k, not Daring-Anteater.** UltraChat-200k is one of the two datasets used in the EAGLE-3 paper (alongside ShareGPT) and is cleaner/larger than what we had. Its `{role, content}` schema is normalized on the fly to the `{from, value}` schema `eagle_utils` expects.
- **Chat template at eval.** `eval_trained.py` wraps every MT-Bench prompt with the Llama-3 chat template before passing it to vLLM so the Instruct model is primed to respond (not continue).

```bash
bash scripts/run_training.sh
```

Pipeline: `train_eagle.py` → `eval_trained.py --method baseline` → `eval_trained.py --method ngram` → `eval_trained.py --method eagle3` → `analyze_trained.py`. Wall time on a single RTX 5090: ~3h for training (80k × 1 epoch at ~7.2 it/s, seq_len 1024) plus ~15 min for the three evals. Well within the 5h budget.

Outputs:

- `eagle_out/` — HF artefact from the Trainer (composed model + tokenizer).
- `eagle_hf_ckpt/` — draft-only checkpoint (`export_speculative_decoding` format) consumed by vLLM's `speculative_config`.
- `results/train_loss.json`, `plots/train_loss.png` — per-step training loss.
- `results/llama32_baseline.csv`, `results/llama32_ngram.csv`, `results/llama32_eagle3.csv` — per-prompt timing on MT-Bench.
- `plots/training_speedup.png` — per-category speedup of n-gram and trained EAGLE-3 vs. baseline.

### Result: spec decoding does not help a 1B target on RTX 5090

| Target | Weights | Baseline tok/s | Per-token budget | n-gram (PLD) | EAGLE-3 |
|---|---|---|---|---|---|
| Qwen3-8B (bf16) | ~16 GB | 95.8 | ~10.4 ms | +43% | +70% |
| **Llama-3.2-1B-Instruct (bf16)** | **~2 GB** | **463.8** | **~2.2 ms** | **−22%** | **−36%** |

The 1B Instruct target already runs at 463 tok/s, i.e. ~2.2 ms per token — essentially the memory-bandwidth lower bound on this hardware (2 GB of weights read per decode step against the 5090's ~1.7 TB/s HBM). Speculative decoding's fixed overhead (draft forward pass + verification scheduling + rejection sampling) is on the order of that same 2 ms, so even **n-gram prompt-lookup — which has no ML draft at all** — runs 22% slower than baseline. Our trained EAGLE-3 draft on top of that overhead loses another 14 points.

Put differently: for the Llama-3.2-1B regime on this GPU, the decoder is already starved for weight bandwidth, not compute — the exact workload where spec decoding normally wins. There is no weight-bandwidth slack left for a draft to reclaim. On the 8B Qwen run, where baseline is ~10× slower per token, spec decoding recovers that slack and gives +43% / +70%.

The training run itself worked correctly — loss dropped from 25 → ~8 within the first few thousand steps and stayed there (modelopt sums TTT losses across ~3 future tokens, so ~8 ≈ 2.7/token). The toy 50-step checkpoint before full training ran at 167 tok/s EAGLE-3 vs. 296 after the full run, so training contributed ~80% more tokens/sec to the draft path — just not enough to cross the 463 tok/s baseline.
