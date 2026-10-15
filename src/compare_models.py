from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASELINE = "yolo11s"


def welch(a: np.ndarray, b: np.ndarray, level: float) -> tuple[float, float, float, float]:
    """Diferença a-b, IC bilateral no nível dado e p bilateral de Welch."""
    diff = a.mean() - b.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)) if se > 0 else len(a) + len(b) - 2
    half = stats.t.ppf(0.5 + level / 2, df) * se
    p = 2 * stats.t.sf(abs(diff) / se, df) if se > 0 else (0.0 if diff else 1.0)
    return diff, diff - half, diff + half, p


def holm(pvals: dict[str, float]) -> dict[str, float]:
    order = sorted(pvals, key=pvals.get)
    adj, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * pvals[k]))
        adj[k] = running
    return adj


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=Path("results/family/grid_family.csv"))
    ap.add_argument("--delta", type=float, default=0.02)
    ap.add_argument("--primary-imgsz", type=int, default=1280)
    ap.add_argument("--expected-seeds", type=int, default=4, help="2 apenas no ensaio curto")
    args = ap.parse_args()

    g = pd.read_csv(args.csv)
    if ((g["conf_operating"] != 0.25) | (g["match_iou_operating"] != 0.5)).any():
        raise ValueError("grade fora de conf=0.25/IoU=0.50")
    g = g[g["capture_tag"] == "native"]
    pc = pd.read_csv(args.csv.with_name(f"{args.csv.stem}_per_class.csv"))
    worst = (pc[(pc["metric"] == "recall") & (pc["capture_tag"] == "native")]
             .groupby(["model", "seed", "imgsz"])["value"].min().rename("worst_class_recall"))
    g = g.join(worst, on=["model", "seed", "imgsz"])
    if (g.groupby(["model", "imgsz"])["seed"].nunique() != args.expected_seeds).any():
        raise ValueError(f"toda célula modelo x imgsz precisa de {args.expected_seeds} seeds")

    def ci(x: pd.Series) -> str:
        h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
        return f"{x.mean():.3f} ± {h:.3f}"

    table = (g.groupby(["imgsz", "model"])
             .agg(recall_op=("recall_op", ci), recall_micro=("recall_micro_op", ci),
                  precision_op=("precision_op", ci), map50_95=("map50_95", ci),
                  worst_class=("worst_class_recall", ci), fwd_ms=("fwd_median_ms", "median"),
                  gflops=("gflops", "first"), params_M=("params_M", "first"))
             .reset_index())
    out_dir = args.csv.parent
    table.to_csv(out_dir / "model_table.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n=== Média ± IC95% entre 4 seeds (val, native, conf=0.25)\n")
    print(table.to_string(index=False))

    tests = []
    for imgsz, sub in g.groupby("imgsz"):
        base = sub.loc[sub["model"] == BASELINE, "recall_op"].to_numpy()
        rows = {}
        for model, s in sub.groupby("model"):
            if model == BASELINE:
                continue
            x = s["recall_op"].to_numpy()
            d, lo95, hi95, p = welch(x, base, 0.95)
            _, lo90, hi90, _ = welch(x, base, 0.90)
            rows[model] = dict(imgsz=imgsz, model=model, reference=model.startswith("rtdetr"),
                               diff_vs_yolo11s=round(d, 4), ci95_lo=round(lo95, 4), ci95_hi=round(hi95, 4),
                               ci90_lo=round(lo90, 4), ci90_hi=round(hi90, 4), p_welch=p)
        adj = holm({m: r["p_welch"] for m, r in rows.items() if not r["reference"]})
        for m, r in rows.items():
            r["p_holm"] = adj.get(m, np.nan)
            significant = r["p_holm"] < 0.05 if not r["reference"] else r["ci95_lo"] > 0 or r["ci95_hi"] < 0
            if significant and r["diff_vs_yolo11s"] > 0:
                r["verdict"] = "superior ao YOLO11s"
            elif significant:
                r["verdict"] = "inferior ao YOLO11s"
            elif -args.delta < r["ci90_lo"] and r["ci90_hi"] < args.delta:
                r["verdict"] = f"equivalente (±{args.delta:.2f})"
            else:
                r["verdict"] = "inconclusivo"
            tests.append(r)
    tests = pd.DataFrame(tests)
    tests.to_csv(out_dir / "model_tests.csv", index=False)
    print(f"\n=== Diferença de recall_op vs YOLO11s (Welch; Holm entre candidatos; TOST delta={args.delta})\n")
    print(tests.drop(columns=["p_welch"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    prim = tests[(tests["imgsz"] == args.primary_imgsz) & ~tests["reference"]]
    better = prim[prim["verdict"] == "superior ao YOLO11s"]
    print(f"\n=== Regra pré-registrada (desfecho primário: native x {args.primary_imgsz})")
    if better.empty:
        equiv = prim[prim["verdict"].str.startswith("equivalente")]["model"].tolist()
        print("Nenhum candidato supera o YOLO11s: YOLO11s mantido"
              + (f"; equivalentes: {', '.join(equiv)} (desempate por latência na tabela)." if equiv else "."))
    else:
        print(f"Superam o YOLO11s: {', '.join(better['model'])}. A regra escolhe o de maior média.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
