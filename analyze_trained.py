"""Compare trained-EAGLE3 vs baseline on MT-Bench and plot per-category speedup."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"
SPEEDUP_PLOT = PLOTS_DIR / "training_speedup.png"


def main() -> None:
    base_p = RESULTS_DIR / "llama32_baseline.csv"
    eagle_p = RESULTS_DIR / "llama32_eagle3.csv"
    if not base_p.exists() or not eagle_p.exists():
        print(f"missing CSV — need both {base_p.name} and {eagle_p.name}")
        return

    base = pd.read_csv(base_p)
    eagle = pd.read_csv(eagle_p)

    merged = eagle.merge(
        base[["prompt_idx", "tok_per_s"]],
        on="prompt_idx", suffixes=("_eagle", "_base"),
    )
    merged["speedup"] = merged["tok_per_s_eagle"] / merged["tok_per_s_base"]

    print(f"baseline mean tok/s: {base['tok_per_s'].mean():.2f} "
          f"(std {base['tok_per_s'].std():.2f}, n={len(base)})")
    print(f"eagle3   mean tok/s: {eagle['tok_per_s'].mean():.2f} "
          f"(std {eagle['tok_per_s'].std():.2f}, n={len(eagle)})")
    print(f"mean per-prompt speedup: "
          f"{merged['speedup'].mean():.3f} ± {merged['speedup'].std():.3f}")

    print("\n=== speedup by category ===")
    cat = merged.groupby("category")["speedup"].agg(["mean", "std", "count"]).round(3)
    print(cat.to_string())

    cats = sorted(merged["category"].unique())
    means = [merged.loc[merged.category == c, "speedup"].mean() for c in cats]
    stds = [merged.loc[merged.category == c, "speedup"].std() for c in cats]

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(cats))
    ax.bar(x, means, yerr=stds, capsize=4, color="#d62728",
           edgecolor="black", linewidth=0.5)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("speedup vs baseline")
    ax.set_title("Trained EAGLE3 on MT-Bench — Llama-3.2-1B, 20k samples, 1 epoch")
    fig.tight_layout()
    fig.savefig(SPEEDUP_PLOT, dpi=160)
    print(f"saved {SPEEDUP_PLOT}")


if __name__ == "__main__":
    main()
