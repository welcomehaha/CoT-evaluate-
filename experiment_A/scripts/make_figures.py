# Developer: hubo
# Date: 2026-05-12

from __future__ import annotations

import argparse
from pathlib import Path


LAMBDA_L = {
    "Baseline": 0.0,
    "L-P": 0.5,
    "F-P": 0.0,
    "L-P+F-P": 0.5,
    "LDR": 0.15,
    "LDR+Confession": 0.15,
    "Dynamic-Penalty": 0.25,
    "A0_base_sft": 0.0,
    "A1_low_length": 0.1,
    "A2_high_length": 0.5,
    "A3_fluency": 0.1,
    "A4_logic_density": 0.1,
    "A5_mixed": 0.15,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Experiment A figures from summary.csv.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def load_summary(path: str):
    import pandas as pd

    df = pd.read_csv(path)
    df["lambda_L"] = df["group"].map(LAMBDA_L)
    return df


def save_line(df, x, y, path, title, ylabel):
    import matplotlib.pyplot as plt

    plot_df = df.dropna(subset=[x]).sort_values(x)
    plt.figure(figsize=(6, 4))
    for prompt_version, part in plot_df.groupby("prompt_version"):
        plt.plot(part[x], part[y], marker="o", label=prompt_version)
    plt.xlabel("lambda_L")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_density_scatter(df, path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    for group, part in df.groupby("group"):
        plt.scatter(part["D"], part["F"], label=group)
    plt.xlabel("D")
    plt.ylabel("F")
    plt.title("Logic Density vs Faithfulness")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_radar(df, path):
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = ["accuracy", "D", "F", "Se", "S"]
    radar_df = df.copy()
    radar_df["token_efficiency"] = 1.0 / (1.0 + radar_df["token_cost"])
    metrics = ["accuracy", "token_efficiency", "D", "F", "Se", "S"]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)
    for group, part in radar_df.groupby("group"):
        row = part.mean(numeric_only=True)
        values = [float(row.get(metric, 0.0)) for metric in metrics]
        values += values[:1]
        ax.plot(angles, values, label=group)
        ax.fill(angles, values, alpha=0.05)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_title("Black-box/reference radar" if len(radar_df["group"].unique()) > 2 else "Group radar")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_summary(args.summary)
    save_line(df, "lambda_L", "token_cost", out_dir / "lambda_length_vs_token_cost.png", "Length Penalty vs Token Cost", "Token Cost")
    save_line(df, "lambda_L", "F", out_dir / "lambda_length_vs_faithfulness.png", "Length Penalty vs Faithfulness", "F")
    save_line(df, "lambda_L", "Se", out_dir / "lambda_length_vs_secrecy.png", "Length Penalty vs Secrecy", "Se")
    save_density_scatter(df, out_dir / "density_vs_faithfulness.png")
    save_radar(df, out_dir / "blackbox_radar.png")
    print(f"[figures] wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
