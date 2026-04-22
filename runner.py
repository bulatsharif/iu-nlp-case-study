"""Run one config: load model, iterate prompts, write per-prompt CSV."""
import argparse
import csv
import sys
import time
from pathlib import Path

import torch

import configs
import data_loader

RESULTS_DIR = Path(__file__).parent / "results"
WARMUP = 3
MAX_TOKENS = 512
MIN_TOKENS = 128
TEMPERATURE = 0.0


def build_llm(cfg: configs.ExpConfig):
    from vllm import LLM

    kwargs = dict(
        model=cfg.target_model,
        dtype=cfg.dtype,
        max_model_len=cfg.max_model_len,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        trust_remote_code=True,
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
    # baseline: no speculative_config

    return LLM(**kwargs)


def gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def run(cfg: configs.ExpConfig) -> None:
    from vllm import SamplingParams

    print(f"=== {cfg.name} ===")
    print(f"target={cfg.target_model} dtype={cfg.dtype} method={cfg.method} "
          f"draft={cfg.draft_model} dataset={cfg.dataset}")
    print(f"gpu={gpu_name()}")

    prompts = data_loader.load(cfg.dataset)
    print(f"{len(prompts)} prompts ({WARMUP} warmup + {len(prompts)-WARMUP} measured)")

    llm = build_llm(cfg)
    sp = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        min_tokens=MIN_TOKENS,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{cfg.name}.csv"
    fields = ["config_name", "dataset", "category", "prompt_idx", "wall_s", "out_tokens", "tok_per_s"]

    measured_tps: list[float] = []
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for i, item in enumerate(prompts):
            prompt = item["prompt"]
            category = item["category"]
            t0 = time.perf_counter()
            out = llm.generate([prompt], sp, use_tqdm=False)
            wall = time.perf_counter() - t0
            n_tok = len(out[0].outputs[0].token_ids)
            tps = n_tok / wall if wall > 0 else 0.0

            if i < WARMUP:
                print(f"[warmup {i}] wall={wall:.2f}s tok={n_tok} tps={tps:.1f}")
                continue

            row = {
                "config_name": cfg.name,
                "dataset": cfg.dataset,
                "category": category,
                "prompt_idx": i,
                "wall_s": f"{wall:.4f}",
                "out_tokens": n_tok,
                "tok_per_s": f"{tps:.2f}",
            }
            writer.writerow(row)
            f.flush()
            measured_tps.append(tps)
            print(f"[{i:3d}] {category[:20]:<20} wall={wall:6.2f}s tok={n_tok:4d} tps={tps:6.1f}")

    mean_tps = sum(measured_tps) / len(measured_tps) if measured_tps else 0.0
    print(f"done: {out_path} — mean tok/s = {mean_tps:.2f} over {len(measured_tps)} prompts")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help=f"one of {sorted(configs.CONFIGS)}")
    args = ap.parse_args()
    run(configs.get(args.config))


if __name__ == "__main__":
    main()
