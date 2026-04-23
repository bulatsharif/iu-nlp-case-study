"""Evaluate speculative decoding on MT-Bench against Llama-3.2-1B-Instruct.

Methods:
  baseline — vanilla autoregressive
  ngram    — vLLM built-in n-gram (prompt-lookup, PLD)
  eagle3   — trained EAGLE-3 draft exported to ./eagle_hf_ckpt

The base model is the **Instruct** variant so prompts go through
`tokenizer.apply_chat_template`; feeding raw MT-Bench questions to a
pretrained base model produced incoherent continuations in the previous
iteration of this experiment and tanked the EAGLE-3 acceptance rate.
"""
import argparse
import csv
import time
from pathlib import Path

import torch
import transformers

import data_loader

ROOT = Path(__file__).parent
BASE_MODEL = "unsloth/Llama-3.2-1B-Instruct"
EXPORT_DIR = ROOT / "eagle_hf_ckpt"
RESULTS_DIR = ROOT / "results"

WARMUP = 3
MAX_TOKENS = 512
MIN_TOKENS = 128
NUM_SPEC_TOKENS = 3
PROMPT_LOOKUP_MIN = 2
PROMPT_LOOKUP_MAX = 4


def build_llm(method: str):
    from vllm import LLM
    kwargs = dict(
        model=BASE_MODEL,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
    )
    if method == "eagle3":
        if not EXPORT_DIR.exists():
            raise FileNotFoundError(f"trained draft not found at {EXPORT_DIR}; run train_eagle.py first")
        kwargs["speculative_config"] = {
            "method": "eagle3",
            "model": str(EXPORT_DIR),
            "num_speculative_tokens": NUM_SPEC_TOKENS,
        }
    elif method == "ngram":
        kwargs["speculative_config"] = {
            "method": "ngram",
            "num_speculative_tokens": NUM_SPEC_TOKENS,
            "prompt_lookup_min": PROMPT_LOOKUP_MIN,
            "prompt_lookup_max": PROMPT_LOOKUP_MAX,
        }
    return LLM(**kwargs)


def gpu_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def format_prompts(prompts: list[dict]) -> list[str]:
    """Apply the Llama-3 chat template so the Instruct model is primed to respond.

    Done once up front so the per-prompt wall-clock timer measures only inference.
    """
    tok = transformers.AutoTokenizer.from_pretrained(BASE_MODEL)
    out = []
    for item in prompts:
        text = tok.apply_chat_template(
            [{"role": "user", "content": item["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        out.append(text)
    return out


def run(method: str) -> None:
    from vllm import SamplingParams

    name = f"llama32_{method}"
    print(f"=== {name} on {gpu_name()} ===")
    prompts = data_loader.load_mt_bench()
    formatted = format_prompts(prompts)
    print(f"{len(prompts)} prompts ({WARMUP} warmup + {len(prompts)-WARMUP} measured)")

    llm = build_llm(method)
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, min_tokens=MIN_TOKENS)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{name}.csv"
    fields = ["config_name", "dataset", "category", "prompt_idx", "wall_s", "out_tokens", "tok_per_s"]

    measured: list[float] = []
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, (item, text) in enumerate(zip(prompts, formatted)):
            t0 = time.perf_counter()
            out = llm.generate([text], sp, use_tqdm=False)
            wall = time.perf_counter() - t0
            n_tok = len(out[0].outputs[0].token_ids)
            tps = n_tok / wall if wall > 0 else 0.0

            if i < WARMUP:
                print(f"[warmup {i}] wall={wall:.2f}s tok={n_tok} tps={tps:.1f}")
                continue

            writer.writerow({
                "config_name": name,
                "dataset": "mt_bench",
                "category": item["category"],
                "prompt_idx": i,
                "wall_s": f"{wall:.4f}",
                "out_tokens": n_tok,
                "tok_per_s": f"{tps:.2f}",
            })
            f.flush()
            measured.append(tps)
            print(f"[{i:3d}] {item['category'][:20]:<20} wall={wall:6.2f}s tok={n_tok:4d} tps={tps:6.1f}")

    mean = sum(measured) / len(measured) if measured else 0.0
    print(f"done: {out_path} — mean tok/s = {mean:.2f} over {len(measured)} prompts")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["baseline", "ngram", "eagle3"], required=True)
    args = ap.parse_args()
    run(args.method)


if __name__ == "__main__":
    main()
