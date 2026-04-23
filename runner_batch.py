"""Batch-size sweep on Qwen3-8B: one LLM load per method, many batch sizes.

For each batch size, split the 80 MT-Bench prompts into contiguous groups of
that size, drop the first group as warmup, and time each remaining group as a
single vLLM generate() call. Throughput is (sum of output tokens in the batch)
/ (wall_s of the batch) — the metric that matters at BS>1.
"""
import argparse
import csv
import time
from pathlib import Path

import torch

import configs
import data_loader

RESULTS_DIR = Path(__file__).parent / "results"
MAX_TOKENS = 512
MIN_TOKENS = 128
TEMPERATURE = 0.0
WARMUP_BATCHES = 1
DEFAULT_BATCH_SIZES = (1, 2, 4, 8, 16)


def build_llm(cfg: configs.ExpConfig, max_num_seqs: int):
    from vllm import LLM

    kwargs = dict(
        model=cfg.target_model,
        dtype=cfg.dtype,
        max_model_len=cfg.max_model_len,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        enforce_eager=cfg.enforce_eager,
        trust_remote_code=True,
        max_num_seqs=max_num_seqs,
    )

    if cfg.method == "ngram":
        kwargs["speculative_config"] = {
            "method": "ngram",
            "num_speculative_tokens": cfg.num_speculative_tokens,
            "prompt_lookup_min": cfg.prompt_lookup_min,
            "prompt_lookup_max": cfg.prompt_lookup_max,
        }
    elif cfg.method == "eagle3":
        kwargs["speculative_config"] = {
            "method": "eagle3",
            "model": cfg.draft_model,
            "num_speculative_tokens": cfg.num_speculative_tokens,
        }

    return LLM(**kwargs)


def gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def run_one_batch_size(llm, sp, prompts, bs: int, method: str, out_path: Path) -> None:
    n_batches = len(prompts) // bs
    if n_batches == 0:
        print(f"  skip bs={bs}: not enough prompts ({len(prompts)})")
        return
    if n_batches <= WARMUP_BATCHES:
        print(f"  skip bs={bs}: only {n_batches} batches, need > warmup ({WARMUP_BATCHES})")
        return

    fields = ["method", "batch_size", "batch_idx", "wall_s", "out_tokens", "tok_per_s"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        measured_tps: list[float] = []
        for b in range(n_batches):
            batch_prompts = [
                prompts[b * bs + j]["prompt"] for j in range(bs)
            ]
            t0 = time.perf_counter()
            outs = llm.generate(batch_prompts, sp, use_tqdm=False)
            wall = time.perf_counter() - t0
            n_tok = sum(len(o.outputs[0].token_ids) for o in outs)
            tps = n_tok / wall if wall > 0 else 0.0

            if b < WARMUP_BATCHES:
                print(f"  [warmup bs={bs} b={b}] wall={wall:.2f}s tok={n_tok} tps={tps:.1f}")
                continue

            writer.writerow({
                "method": method,
                "batch_size": bs,
                "batch_idx": b,
                "wall_s": f"{wall:.4f}",
                "out_tokens": n_tok,
                "tok_per_s": f"{tps:.2f}",
            })
            f.flush()
            measured_tps.append(tps)
            print(f"  [bs={bs:2d} b={b:3d}] wall={wall:7.2f}s tok={n_tok:5d} tps={tps:7.1f}")

    mean_tps = sum(measured_tps) / len(measured_tps) if measured_tps else 0.0
    print(f"  bs={bs}: mean tok/s = {mean_tps:.2f} over {len(measured_tps)} batches → {out_path}")


def run(cfg: configs.ExpConfig, batch_sizes: list[int]) -> None:
    from vllm import SamplingParams

    print(f"=== batch sweep: {cfg.method} on {cfg.target_model} ===")
    print(f"batch_sizes={batch_sizes}  gpu={gpu_name()}")
    prompts = data_loader.load(cfg.dataset)
    print(f"{len(prompts)} prompts from {cfg.dataset}")

    max_bs = max(batch_sizes)
    llm = build_llm(cfg, max_num_seqs=max_bs)
    sp = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        min_tokens=MIN_TOKENS,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for bs in batch_sizes:
        out_path = RESULTS_DIR / f"qwen_batch_{cfg.method}_bs{bs:02d}.csv"
        run_one_batch_size(llm, sp, prompts, bs, cfg.method, out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["baseline", "ngram", "eagle3"])
    ap.add_argument(
        "--batch-sizes",
        default=",".join(str(x) for x in DEFAULT_BATCH_SIZES),
        help="comma-separated list of batch sizes",
    )
    args = ap.parse_args()

    cfg_name = {"baseline": "qwen_baseline", "ngram": "qwen_ngram", "eagle3": "qwen_eagle3"}[args.method]
    cfg = configs.get(cfg_name)

    batch_sizes = sorted({int(x) for x in args.batch_sizes.split(",") if x.strip()})
    run(cfg, batch_sizes)


if __name__ == "__main__":
    main()
