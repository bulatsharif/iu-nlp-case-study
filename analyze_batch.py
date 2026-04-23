"""Aggregate qwen_batch_*.csv → summary table + speedup-vs-batch-size plot."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"

METHOD_COLORS = {"baseline": "#8a8a8a", "ngram": "#1f77b4", "eagle3": "#d62728"}
METHODS = ["baseline", "ngram", "eagle3"]


def load_all() -> pd.DataFrame:
    frames = []
    for p in sorted(RESULTS_DIR.glob("qwen_batch_*.csv")):
        df = pd.read_csv(p)
        if df.empty:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    agg = (df.groupby(["method", "batch_size"])["tok_per_s"]
             .agg(mean_tps="mean", std_tps="std", n="count")
             .reset_index())
    base = (agg[agg.method == "baseline"]
              .set_index("batch_size")["mean_tps"]
              .rename("baseline_tps"))
    agg = agg.join(base, on="batch_size")
    agg["speedup"] = agg["mean_tps"] / agg["baseline_tps"]
    return agg


def plot_speedup(summary: pd.DataFrame, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
    bs_ticks = sorted(summary["batch_size"].unique())

    # left: absolute throughput (aggregate tok/s across all sequences in the batch)
    for m in METHODS:
        sub = summary[summary.method == m].sort_values("batch_size")
        ax1.errorbar(
            sub["batch_size"], sub["mean_tps"], yerr=sub["std_tps"].fillna(0.0),
            marker="o", capsize=3, color=METHOD_COLORS[m], label=m,
            linewidth=1.8, markersize=7,
        )
        for _, row in sub.iterrows():
            ax1.annotate(
                f"{row['mean_tps']:.0f}",
                xy=(row["batch_size"], row["mean_tps"]),
                xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=8, color=METHOD_COLORS[m],
            )
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(bs_ticks)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.set_xlabel("batch size (concurrent sequences)")
    ax1.set_ylabel("throughput (tokens / second, aggregate)")
    ax1.set_title("Absolute throughput")
    ax1.legend(frameon=False, loc="upper left")
    ax1.grid(alpha=0.3)

    # right: speedup vs baseline at the SAME batch size
    for m in ["ngram", "eagle3"]:
        sub = summary[summary.method == m].sort_values("batch_size")
        ax2.plot(
            sub["batch_size"], sub["speedup"],
            marker="o", color=METHOD_COLORS[m], label=m,
            linewidth=1.8, markersize=7,
        )
        for _, row in sub.iterrows():
            ax2.annotate(
                f"{row['speedup']:.2f}×",
                xy=(row["batch_size"], row["speedup"]),
                xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=8, color=METHOD_COLORS[m],
            )
    ax2.axhline(1.0, color="black", linewidth=0.9, linestyle="--", label="baseline (1.0×)")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(bs_ticks)
    ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax2.set_xlabel("batch size (concurrent sequences)")
    ax2.set_ylabel("speedup vs baseline (tok/s ratio)")
    ax2.set_title("Speculative speedup vs batch size")
    ax2.legend(frameon=False, loc="best")
    ax2.grid(alpha=0.3)

    fig.suptitle(
        "Qwen3-8B — speculative decoding across batch sizes (MT-Bench, RTX 5090)",
        fontsize=13,
    )
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


def main() -> None:
    df = load_all()
    if df.empty:
        print("no qwen_batch_*.csv in results/ — run scripts/run_batch.sh first.")
        return

    summary = summarize(df)
    print("\n=== mean tok/s and speedup by method × batch size ===")
    print(summary.round(3).to_string(index=False))

    plot_speedup(summary, PLOTS_DIR / "batch_size_speedup.png")


if __name__ == "__main__":
    main()
