#!/usr/bin/env python3
"""
plot_results.py — lê results/grid.csv e produz as figuras da dissertação.

Inclui o modelo de latência fim-a-fim drone -> edge, que é onde os dois
eixos do experimento se encontram:

    T_total = T_overhead + T_encode + T_transmissao + T_inferencia

    T_encode        = k * megapixels                (custo no drone)
    T_transmissao   = payload_kbit / banda_kbit_s   (o link é o gargalo real)
    T_inferencia    = pre + forward + pos           (custo no edge)

O ponto da tese: baixar a resolução de captura reduz T_transmissao de forma
QUADRÁTICA (área) e o recall de forma aproximadamente sigmoide. Existe,
portanto, um joelho — a menor resolução cujo recall ainda está acima do
mínimo aceitável. É esse joelho que a Fig. pareto identifica, e ele MUDA
conforme a banda disponível do enlace.

Uso:
    python src/plot_results.py --csv results/grid.csv --config configs/experiment.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ==========================================================================
# Modelo de latência fim-a-fim
# ==========================================================================
def add_latency_columns(df: pd.DataFrame, link: dict) -> pd.DataFrame:
    df = df.copy()
    mpix = (df["mean_w"] * df["mean_h"]) / 1e6
    df["encode_ms"] = link["encode_ms_per_mpixel"] * mpix
    df["compute_ms"] = df["pre_ms"] + df["fwd_median_ms"] + df["post_ms"]

    for bw in link["bandwidths_mbps"]:
        # kB * 8 = kbit ; kbit / (Mbit/s) = ms   (as constantes 1000 se cancelam)
        tx = df["mean_kb"] * 8.0 / bw
        df[f"tx_ms@{bw}"] = tx
        df[f"e2e_ms@{bw}"] = link["overhead_ms"] + df["encode_ms"] + tx + df["compute_ms"]
    return df


# ==========================================================================
def order_caps(df: pd.DataFrame) -> list[str]:
    """Ordena as tags de captura pela altura real, 'native' por último."""
    d = df.groupby("capture_tag")["mean_h"].max().sort_values()
    return list(d.index)


def fig_recall_vs_capture(df: pd.DataFrame, out: Path, metric: str) -> None:
    """A figura que responde 'qual a menor resolução aceitável?'"""
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for imgsz, g in df.groupby("imgsz"):
        g = g.sort_values("mean_h")
        ax.plot(g["mean_h"], g[metric], marker="o", lw=1.8, label=f"imgsz={imgsz}")
        for _, r in g.iterrows():
            ax.annotate(r["capture_tag"], (r["mean_h"], r[metric]),
                        textcoords="offset points", xytext=(0, 7),
                        fontsize=7, ha="center", alpha=0.6)
    ax.set_xlabel("Altura da imagem capturada/transmitida (px)")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title("Perda de informação na captura × capacidade de detecção")
    ax.legend(title="Entrada da rede", frameon=False, fontsize=8)
    fig.savefig(out)
    plt.close(fig)


def fig_heatmap(df: pd.DataFrame, out: Path, metric: str) -> None:
    """Desacopla os dois fatores: quem manda, o pixel capturado ou o imgsz?"""
    caps = order_caps(df)
    piv = (df.pivot_table(index="capture_tag", columns="imgsz", values=metric, aggfunc="mean")
             .reindex(caps))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(piv.columns)), piv.columns)
    ax.set_yticks(range(len(piv.index)), piv.index)
    ax.set_xlabel("imgsz da inferência (custo computacional)")
    ax.set_ylabel("Resolução de captura (informação disponível)")
    ax.set_title(f"{metric} — grade fatorial")
    ax.grid(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=8, color="w" if v < np.nanmean(piv.values) else "k")
    fig.colorbar(im, ax=ax, shrink=0.85, label=metric)
    fig.savefig(out)
    plt.close(fig)


def fig_tri_axis(df: pd.DataFrame, out: Path, metric: str) -> None:
    """Resolução × Tempo de inferência × Recall no mesmo par de eixos."""
    caps = order_caps(df)
    sub = df[df["capture_tag"] == caps[-1]].sort_values("imgsz")
    if sub.empty:
        return

    fig, ax1 = plt.subplots(figsize=(7.2, 4.4))
    ax2 = ax1.twinx()
    ax2.grid(False)

    for model, g in sub.groupby("model"):
        g = g.sort_values("imgsz")
        ax1.plot(g["imgsz"], g[metric], marker="o", lw=2, label=f"{model} — {metric}")
        ax2.plot(g["imgsz"], g["compute_ms"], marker="s", ls="--", lw=1.5,
                 alpha=0.75, label=f"{model} — latência")

    ax1.set_xlabel("Resolução de entrada da rede (imgsz, px)")
    ax1.set_ylabel(metric.replace("_", " ").title(), color="tab:blue")
    ax2.set_ylabel("Tempo de inferência por quadro (ms)", color="tab:red")
    ax1.set_title("Acurácia × custo computacional")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="center right")
    fig.savefig(out)
    plt.close(fig)


def fig_payload(df: pd.DataFrame, out: Path, link: dict) -> None:
    """Quanto custa, em bytes e em milissegundos de rádio, cada resolução."""
    g = (df.groupby(["capture_tag", "mean_h"], as_index=False)["mean_kb"]
           .mean().sort_values("mean_h"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    a1.bar(g["capture_tag"], g["mean_kb"], color="#4C72B0")
    a1.set_ylabel("Payload médio por quadro (kB)")
    a1.set_xlabel("Resolução de captura")
    a1.set_title("Custo de transmissão em bytes")

    for bw in link["bandwidths_mbps"]:
        a2.plot(g["capture_tag"], g["mean_kb"] * 8.0 / bw, marker="o", label=f"{bw} Mbit/s")
    a2.set_ylabel("Tempo de transmissão (ms)")
    a2.set_xlabel("Resolução de captura")
    a2.set_yscale("log")
    a2.set_title("Custo de transmissão no enlace")
    a2.legend(title="Banda do link", frameon=False, fontsize=8)
    fig.savefig(out)
    plt.close(fig)


def fig_pareto(df: pd.DataFrame, out: Path, link: dict, metric: str) -> None:
    """Latência fim-a-fim × recall. É aqui que se escolhe a configuração."""
    bws = link["bandwidths_mbps"]
    n = len(bws)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.9 * nrow), squeeze=False)

    caps = order_caps(df)
    cmap = plt.get_cmap("plasma")
    colors = {t: cmap(i / max(len(caps) - 1, 1)) for i, t in enumerate(caps)}

    for k, bw in enumerate(bws):
        ax = axes[k // ncol][k % ncol]
        col = f"e2e_ms@{bw}"
        for tag in caps:
            g = df[df["capture_tag"] == tag]
            ax.scatter(g[col], g[metric], s=46, color=colors[tag], label=tag,
                       edgecolor="k", linewidth=0.4, zorder=3)

        # fronteira de Pareto: menor latência para cada nível de recall
        pts = df[[col, metric]].dropna().sort_values(col).values
        front, best = [], -np.inf
        for x, y in pts:
            if y > best:
                best = y
                front.append((x, y))
        if front:
            fx, fy = zip(*front)
            ax.step(fx, fy, where="post", color="k", lw=1.2, alpha=0.55, zorder=2)

        ax.set_xscale("log")
        ax.set_xlabel("Latência fim-a-fim drone→alerta (ms, log)")
        ax.set_ylabel(metric)
        ax.set_title(f"Enlace {bw} Mbit/s")
        if k == 0:
            ax.legend(title="Captura", frameon=False, fontsize=7, ncol=2)

    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    fig.suptitle("Fronteira de Pareto: quanto recall se compra com cada milissegundo")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ==========================================================================
def knee_table(df: pd.DataFrame, link: dict, metric: str, floor: float) -> pd.DataFrame:
    """Menor payload que ainda atinge recall >= floor, por banda de link."""
    ok = df[df[metric] >= floor]
    out = []
    for bw in link["bandwidths_mbps"]:
        col = f"e2e_ms@{bw}"
        if ok.empty:
            out.append({"bandwidth_mbps": bw, "config": "nenhuma atinge o piso"})
            continue
        best = ok.loc[ok[col].idxmin()]
        out.append({
            "bandwidth_mbps": bw,
            "capture": best["capture_tag"],
            "imgsz": int(best["imgsz"]),
            "model": best["model"],
            metric: round(float(best[metric]), 4),
            "payload_kB": round(float(best["mean_kb"]), 1),
            "e2e_ms": round(float(best[col]), 1),
        })
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=Path("results/grid.csv"))
    ap.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    ap.add_argument("--outdir", type=Path, default=Path("results/figs"))
    ap.add_argument("--metric", default="recall_op",
                    choices=["recall_op", "recall_sweep", "map50", "map50_95"],
                    help="recall_op = ponto de operação real do sistema de alerta")
    ap.add_argument("--recall-floor", type=float, default=0.60,
                    help="recall mínimo aceitável para o alerta")
    ap.add_argument("--latency-csv", type=Path, default=None,
                    help="saída da grade com latência; padrão deriva de --csv")
    ap.add_argument("--knee-csv", type=Path, default=None,
                    help="saída da configuração ótima; padrão deriva de --csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    link = cfg["link"]

    df = pd.read_csv(args.csv)
    df = add_latency_columns(df, link)
    args.outdir.mkdir(parents=True, exist_ok=True)

    m = args.metric
    fig_recall_vs_capture(df, args.outdir / f"01_{m}_vs_captura.png", m)
    fig_heatmap(df, args.outdir / f"02_heatmap_{m}.png", m)
    fig_tri_axis(df, args.outdir / f"03_acuracia_vs_latencia.png", m)
    fig_payload(df, args.outdir / "04_payload_transmissao.png", link)
    fig_pareto(df, args.outdir / f"05_pareto_{m}.png", link, m)

    stem_suffix = "" if args.csv.stem == "grid" else f"_{args.csv.stem}"
    latency_csv = args.latency_csv or args.csv.with_name(f"grid_com_latencia{stem_suffix}.csv")
    knee_csv = args.knee_csv or args.csv.with_name(f"config_otima_por_banda{stem_suffix}.csv")
    latency_csv.parent.mkdir(parents=True, exist_ok=True)
    knee_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(latency_csv, index=False)

    kt = knee_table(df, link, m, args.recall_floor)
    kt.to_csv(knee_csv, index=False)

    print(f"\nFiguras em {args.outdir}\n")
    print(f"Configuração de menor latência com {m} >= {args.recall_floor}:\n")
    print(kt.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
