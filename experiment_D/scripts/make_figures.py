# Developer: hubo
# Date: 2026-05-12

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Experiment D figures.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--strata", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def bar(df, y, path, title, ylabel):
    import matplotlib.pyplot as plt

    plot_df = df.sort_values("group")
    plt.figure(figsize=(8, 4))
    plt.bar(plot_df["group"], plot_df[y].fillna(0))
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def scatter(df, path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    for group, part in df.groupby("group"):
        plt.scatter(part["token_cost"], part["F"], label=group)
    plt.xlabel("Token cost")
    plt.ylabel("F")
    plt.title("Token Cost vs Faithfulness")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    import pandas as pd

    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.summary)
    strata = pd.read_csv(args.strata)
    bar(df, "U", out_dir / "figure_mitigation_tradeoff.png", "Mitigation Utility by Strategy", "U")
    bar(df, "Se", out_dir / "figure_se_reduction_by_strategy.png", "Se by Strategy", "Se")
    bar(df, "verbalization_recall", out_dir / "figure_verbalization_by_strategy.png", "Verbalization Recall by Strategy", "Recall")
    bar(df, "over_confession_rate", out_dir / "figure_over_confession_by_strategy.png", "Over-confession by Strategy", "Rate")
    scatter(df, out_dir / "figure_token_cost_vs_faithfulness.png")
    if not strata.empty:
        hard = strata[strata["difficulty_bucket"] == "hard_high"]
        if not hard.empty:
            bar(hard.groupby("group", as_index=False).mean(numeric_only=True), "F", out_dir / "figure_hard_sample_f_by_strategy.png", "Hard Sample F by Strategy", "F")
    print(f"[figures-D] wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
