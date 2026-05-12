# Developer: hubo
# Date: 2026-05-12

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Experiment C figures.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--monitor-summary", required=True)
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


def monitor_line(df, path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    for group, part in df.groupby("group"):
        part = part.sort_values("monitor_input")
        plt.plot(part["monitor_input"], part["monitor_recall"], marker="o", label=group)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Monitor recall")
    plt.title("Monitor Recall by Input")
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
    mon = pd.read_csv(args.monitor_summary)
    bar(df, "hack_rate", out_dir / "figure_hack_rate_by_group.png", "Hack Rate by Group", "Hack rate")
    bar(df, "reward_score_gap", out_dir / "figure_reward_gap_by_group.png", "Reward Score Gap by Group", "Monitor - groundtruth")
    bar(df, "Se", out_dir / "figure_se_by_group.png", "Se by Group", "Se")
    bar(df, "cot_concealment_rate", out_dir / "figure_concealment_by_group.png", "CoT Concealment by Group", "Concealment rate")
    bar(df, "behavioral_failure_rate", out_dir / "figure_behavioral_failure_by_group.png", "Behavioral Failure by Group", "Failure rate")
    monitor_line(mon, out_dir / "figure_monitor_recall_by_input.png")
    print(f"[figures-C] wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
