# Developer: hubo
# Date: 2026-05-12

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Experiment B figures.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--cue-type-summary", default=None)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def bar(df, y, path, title, ylabel):
    import matplotlib.pyplot as plt

    plot_df = df.sort_values("group")
    labels = plot_df["group"].tolist()
    values = plot_df[y].fillna(0).tolist()
    plt.figure(figsize=(8, 4))
    plt.bar(labels, values)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def cue_type_heatmap(path_csv, out_path):
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(path_csv)
    if df.empty:
        return
    pivot = df.pivot_table(index="group", columns="cue_type", values="Se", aggfunc="mean").fillna(0)
    plt.figure(figsize=(9, 4.5))
    plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(label="Se")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title("Se by Cue Type")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    import pandas as pd

    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.summary)
    bar(df, "cue_following_rate", out_dir / "figure_cue_following_by_group.png", "Cue Following by Group", "Cue-following rate")
    bar(df, "verbalization_recall", out_dir / "figure_verbalization_recall_by_group.png", "Verbalization Recall by Group", "Verbalization recall")
    bar(df, "Se", out_dir / "figure_se_by_reward_group.png", "Se by Reward Group", "Se")
    bar(df, "rationalization_rate", out_dir / "figure_rationalization_by_group.png", "Rationalization by Group", "Rationalization rate")
    bar(df, "accuracy_drop", out_dir / "figure_accuracy_drop_by_group.png", "Accuracy Drop by Group", "Accuracy original - Accuracy cue")
    if args.cue_type_summary:
        cue_type_heatmap(args.cue_type_summary, out_dir / "figure_se_by_cue_type.png")
    print(f"[figures-B] wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
