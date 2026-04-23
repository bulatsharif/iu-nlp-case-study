"""Compare baseline / n-gram / trained-EAGLE3 on MT-Bench and plot per-category speedup."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"
SPEEDUP_PLOT = PLOTS_DIR / "training_speedup.png"

METHODS = [
    ("ngram", "#ff7f0e", "n-gram (PLD)"),
    ("eagle3", "#d62728", "EAGLE-3 (trained)"),
]


def _load(name: str) -> pd.DataFrame:
    p = RESULTS_DIR / f"llama32_{name}.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def main() -> None:
    base = _load("baseline")
    base_mean = base["tok_per_s"].mean()
    print(f"baseline mean tok/s: {base_mean:.2f} "
          f"(std {base['tok_per_s'].std():.2f}, n={len(base)})")
    print(f"baseline per-token budget: {1000 / base_mean:.2f} ms  "
          f"(memory-bandwidth floor on a 2 GB bf16 model + 5090 HBM ≈ ~1.2 ms)")

    merged_by_method: dict[str, pd.DataFrame] = {}
    for method, _, _ in METHODS:
        try:
            df = _load(method)
        except FileNotFoundError:
            print(f"skipping {method}: csv not found")
            continue
        m = df.merge(
            base[["prompt_idx", "tok_per_s"]],
            on="prompt_idx", suffixes=("_spec", "_base"),
        )
        m["speedup"] = m["tok_per_s_spec"] / m["tok_per_s_base"]
        merged_by_method[method] = m
        print(f"{method:8s} mean tok/s: {df['tok_per_s'].mean():7.2f}  "
              f"speedup: {m['speedup'].mean():.3f} ± {m['speedup'].std():.3f}")

    if not merged_by_method:
        print("no speculative-decoding results to compare")
        return

    losers = [m for m, df in merged_by_method.items() if df["speedup"].mean() < 1.0]
    if losers:
        print(
            "\nNOTE: spec methods below baseline on 1B target. "
            "At ~2 ms/token, spec-decoding overhead (draft forward + verification + "
            "rejection sampling) exceeds the baseline decode budget — even n-gram, "
            "which has no ML draft, loses. This is expected for sub-~5B targets on "
            "an RTX 5090; spec decoding wins on the 8B Qwen run in the main benchmark."
        )

    print("\n=== speedup by category ===")
    # Build a per-category × method table.
    per_cat = pd.DataFrame(index=sorted(next(iter(merged_by_method.values()))["category"].unique()))
    for method, m in merged_by_method.items():
        per_cat[method] = m.groupby("category")["speedup"].mean().round(3)
    print(per_cat.to_string())

    cats = list(per_cat.index)
    x = np.arange(len(cats))
    width = 0.8 / max(len(merged_by_method), 1)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (method, color, label) in enumerate(METHODS):
        if method not in merged_by_method:
            continue
        m = merged_by_method[method]
        means = [m.loc[m.category == c, "speedup"].mean() for c in cats]
        stds = [m.loc[m.category == c, "speedup"].std() for c in cats]
        offset = (i - (len(METHODS) - 1) / 2) * width
        ax.bar(
            x + offset, means, width, yerr=stds, capsize=3,
            color=color, edgecolor="black", linewidth=0.5, label=label,
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("speedup vs baseline")
    ax.set_title(
        "Spec-decoding on MT-Bench — Llama-3.2-1B-Instruct\n"
        "EAGLE-3 draft: 80k UltraChat-200k samples, 1 epoch"
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(SPEEDUP_PLOT, dpi=160)
    print(f"saved {SPEEDUP_PLOT}")


if __name__ == "__main__":
    main()
