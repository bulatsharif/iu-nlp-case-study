"""Aggregate the six CSVs → summary table + two plots."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"

# (target_family, method) for each config
CONFIG_META = {
    "qwen_baseline":  ("Qwen3-8B", "baseline"),
    "qwen_ngram":     ("Qwen3-8B", "ngram"),
    "qwen_eagle3":    ("Qwen3-8B", "eagle3"),
    "avibe_baseline": ("A-Vibe",   "baseline"),
    "avibe_ngram":    ("A-Vibe",   "ngram"),
    "avibe_eagle3":   ("A-Vibe",   "eagle3"),
}
BASELINE_OF = {"Qwen3-8B": "qwen_baseline", "A-Vibe": "avibe_baseline"}
METHOD_COLORS = {"baseline": "#8a8a8a", "ngram": "#1f77b4", "eagle3": "#d62728"}


def load_all() -> pd.DataFrame:
    frames = []
    for name in CONFIG_META:
        p = RESULTS_DIR / f"{name}.csv"
        if not p.exists():
            print(f"[warn] missing {p}")
            continue
        df = pd.read_csv(p)
        family, method = CONFIG_META[name]
        df["target"] = family
        df["method"] = method
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def per_prompt_speedup(df: pd.DataFrame) -> pd.DataFrame:
    """Align spec method to its baseline by (target, prompt_idx) and take the tok/s ratio."""
    base = (df[df.method == "baseline"]
            .set_index(["target", "prompt_idx"])["tok_per_s"]
            .rename("base_tps"))
    spec = df[df.method != "baseline"].copy()
    spec = spec.join(base, on=["target", "prompt_idx"])
    spec["speedup"] = spec["tok_per_s"] / spec["base_tps"]
    return spec


def print_summary(df: pd.DataFrame, spec: pd.DataFrame) -> None:
    print("\n=== mean tok/s by config ===")
    tps = df.groupby(["target", "method"])["tok_per_s"].agg(["mean", "std", "count"])
    print(tps.round(2).to_string())

    print("\n=== mean speedup vs baseline, by target × method × category ===")
    summ = (spec.groupby(["target", "method", "category"])["speedup"]
                .agg(["mean", "std", "count"]).round(3))
    print(summ.to_string())

    print("\n=== mean speedup vs baseline, by target × method ===")
    overall = (spec.groupby(["target", "method"])["speedup"]
                   .agg(["mean", "std", "count"]).round(3))
    print(overall.to_string())


def plot_speedup_by_domain(spec: pd.DataFrame, out: Path) -> None:
    targets = ["Qwen3-8B", "A-Vibe"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    for ax, target in zip(axes, targets):
        sub = spec[spec.target == target]
        if sub.empty:
            ax.set_title(f"{target}  (no data)")
            continue
        cats = sorted(sub["category"].unique())
        methods = ["ngram", "eagle3"]
        x = np.arange(len(cats))
        width = 0.38

        for i, m in enumerate(methods):
            means, stds = [], []
            for c in cats:
                vals = sub[(sub.method == m) & (sub.category == c)]["speedup"]
                means.append(vals.mean() if len(vals) else np.nan)
                stds.append(vals.std() if len(vals) > 1 else 0.0)
            offset = (i - 0.5) * width
            ax.bar(x + offset, means, width, yerr=stds, capsize=3,
                   label=m, color=METHOD_COLORS[m], edgecolor="black", linewidth=0.5)

        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=30, ha="right", fontsize=9)
        ax.set_title(target, fontsize=12)
        ax.set_ylabel("speedup vs baseline (tok/s ratio)")
        ax.legend(loc="upper left", frameon=False)

    fig.suptitle("Per-domain speculative decoding speedup", fontsize=13)
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


def plot_throughput(df: pd.DataFrame, out: Path) -> None:
    order = ["qwen_baseline", "qwen_ngram", "qwen_eagle3",
             "avibe_baseline", "avibe_ngram", "avibe_eagle3"]
    means, stds, colors, labels = [], [], [], []
    for name in order:
        sub = df[df.config_name == name]["tok_per_s"]
        means.append(sub.mean() if len(sub) else 0.0)
        stds.append(sub.std() if len(sub) > 1 else 0.0)
        _, method = CONFIG_META[name]
        colors.append(METHOD_COLORS[method])
        labels.append(name)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(order))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("tokens / second")
    ax.set_title("End-to-end throughput by config")
    # method legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, edgecolor="black", label=m)
               for m, c in METHOD_COLORS.items()]
    ax.legend(handles=handles, loc="upper right", frameon=False)
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


def main() -> None:
    df = load_all()
    if df.empty:
        print("no CSVs in results/ — run the benchmark first.")
        return
    spec = per_prompt_speedup(df)
    print_summary(df, spec)
    plot_speedup_by_domain(spec, PLOTS_DIR / "speedup_by_domain.png")
    plot_throughput(df, PLOTS_DIR / "throughput.png")


if __name__ == "__main__":
    main()
