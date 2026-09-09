from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("results/grid.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("results/summary"))
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    metrics = ["recall_op", "precision_op", "map50", "map50_95",
               "recall_sweep", "fwd_median_ms"]
    group_cols = ["model", "split", "capture_tag", "imgsz"]
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = group["seed"].nunique()
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            mean = float(np.mean(values)) if len(values) else np.nan
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            critical = float(t.ppf(0.975, len(values) - 1)) if len(values) > 1 else 0.0
            half = critical * std / np.sqrt(len(values)) if len(values) > 1 else 0.0
            row[f"{metric}_mean"] = round(mean, 6) if not np.isnan(mean) else None
            row[f"{metric}_std"] = round(std, 6)
            row[f"{metric}_ci95_half"] = round(half, 6)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.outdir / "summary_by_seed.csv", index=False)

    val = df[df["split"] == "val"] if "split" in df else df
    native = val[val["capture_tag"] == "native"]
    if native.empty:
        native = val
    comparison = (native.groupby(["model", "seed"], as_index=False)
                  [["map50", "map50_95", "recall_op", "precision_op", "fwd_median_ms"]].mean())
    comparison.to_csv(args.outdir / "model_comparison_by_seed.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model, group in comparison.groupby("model"):
        means = group.groupby("model")["map50_95"].mean().iloc[0]
        std = group["map50_95"].std(ddof=1) if len(group) > 1 else 0
        ax.errorbar(model, means, yerr=std, fmt="o", capsize=4, label=model)
    ax.set_ylabel("mAP50-95 médio no val")
    ax.set_title("Comparação dos modelos: média e desvio-padrão entre seeds")
    ax.legend(frameon=False)
    fig.savefig(args.outdir / "model_comparison.png")
    plt.close(fig)
    print(f"Resumo salvo em {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())