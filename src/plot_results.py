#!/usr/bin/env python3
"""
plot_results.py — lê results/grid.csv e produz as figuras da dissertação.

Inclui o modelo de latência fim-a-fim drone -> edge, que é onde os dois
eixos do experimento se encontram:

    T_total = T_overhead + T_encode + T_transmissao + T_inferencia

    T_encode        = k * megapixels                (custo no drone)
    T_transmissao   = payload_kbit / banda_kbit_s   (o link é o gargalo real)
    T_inferencia    = pre + forward + pos           (custo no edge)

O experimento testa, sem assumir previamente a forma da curva, como payload,
latência e recall mudam com captura, entrada da rede e modelo. A seleção usa
uma restrição de recall pré-declarada; a linha de Pareto não deve ser chamada
de detector automático de "joelho".

Uso:
    python src/plot_results.py --csv results/grid.csv --config configs/experiment.yaml
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import t

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
    if "mean_pixels" not in df:
        raise ValueError("CSV sem mean_pixels; regenere os dados com o pipeline atual")
    mpix = df["mean_pixels"] / 1e6
    df["encode_ms"] = link["encode_ms_per_mpixel"] * mpix
    required_op = {"pre_op_ms", "post_op_ms"}
    if not required_op.issubset(df.columns):
        raise ValueError("CSV antigo sem tempos operacionais; execute novamente benchmark.py")
    df["compute_ms"] = df["pre_op_ms"] + df["fwd_median_ms"] + df["post_op_ms"]
    df["compute_p95_ms"] = df["pre_op_ms"] + df["fwd_p95_ms"] + df["post_op_ms"]

    for bw in link["bandwidths_mbps"]:
        # kB * 8 = kbit ; kbit / (Mbit/s) = ms   (as constantes 1000 se cancelam)
        tx = df["mean_kb"] * 8.0 / bw
        df[f"tx_ms@{bw}"] = tx
        df[f"e2e_ms@{bw}"] = link["overhead_ms"] + df["encode_ms"] + tx + df["compute_ms"]
        df[f"e2e_p95_ms@{bw}"] = link["overhead_ms"] + df["encode_ms"] + tx + df["compute_p95_ms"]
    return df


def aggregate_seeds(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """One scientific point per configuration, with uncertainty across seeds."""
    required = {"seed", "model", "split", "capture_tag", "imgsz", "precision_mode"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV sem colunas da grade multi-seed: {sorted(missing)}")
    group_cols = ["model", "split", "capture_tag", "capture_h", "imgsz", "precision_mode"]
    rows = []
    numeric = df.select_dtypes(include=[np.number]).columns
    for keys, group in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        for column in numeric:
            if column != "seed":
                row[column] = float(group[column].mean())
        n = group["seed"].nunique()
        row["n_seeds"] = int(n)
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            critical = float(t.ppf(0.975, len(values) - 1)) if len(values) > 1 else 0.0
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95_half"] = critical * std / np.sqrt(len(values)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


# ==========================================================================
def order_caps(df: pd.DataFrame) -> list[str]:
    """Ordena as tags de captura pela altura real, 'native' por último."""
    d = df.groupby("capture_tag")["mean_h"].max().sort_values()
    return list(d.index)


def fig_recall_vs_capture(df: pd.DataFrame, out: Path, metric: str) -> None:
    """A figura que responde 'qual a menor resolução aceitável?'"""
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for (model, imgsz), g in df.groupby(["model", "imgsz"]):
        g = g.sort_values("mean_h")
        ax.errorbar(g["mean_h"], g[metric], yerr=g.get(f"{metric}_ci95_half"),
                    marker="o", lw=1.5, capsize=2,
                    label=f"{model}, imgsz={int(imgsz)}")
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
    models = list(df["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(6.0 * len(models), 4.2), squeeze=False)
    vmin, vmax = df[metric].min(), df[metric].max()
    im = None
    for ax, model in zip(axes[0], models):
        piv = (df[df["model"] == model]
               .pivot_table(index="capture_tag", columns="imgsz", values=metric, aggfunc="mean")
               .reindex(caps))
        im = ax.imshow(piv.values, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(piv.columns)), [int(x) for x in piv.columns])
        ax.set_yticks(range(len(piv.index)), piv.index)
        ax.set_xlabel("imgsz da inferência")
        ax.set_ylabel("Captura")
        ax.set_title(model)
        ax.grid(False)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                            color="w" if v < (vmin + vmax) / 2 else "k")
    fig.suptitle(f"{metric} — grade fatorial, média entre seeds")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label=metric)
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
    markers = {m: marker for m, marker in zip(df["model"].unique(), ["o", "s", "^"]) }

    for k, bw in enumerate(bws):
        ax = axes[k // ncol][k % ncol]
        col = f"e2e_ms@{bw}"
        for (tag, model), g in df.groupby(["capture_tag", "model"]):
            ax.scatter(g[col], g[metric], s=46, color=colors[tag], marker=markers[model],
                       edgecolor="k", linewidth=0.4, zorder=3)
            ci_col = f"{metric}_ci95_half"
            if ci_col in g:
                ax.errorbar(g[col], g[metric], yerr=g[ci_col], fmt="none",
                            ecolor=colors[tag], alpha=0.45, capsize=2, zorder=1)

        # fronteira de Pareto: menor latência para cada nível de recall
        pts = (df[[col, metric]].dropna().groupby(col, as_index=False)[metric].max()
               .sort_values(col).values)
        front, best = [], -np.inf
        for x, y in pts:
            if y > best:
                best = y
                front.append((x, y))
        if front:
            fx, fy = zip(*front)
            ax.step(fx, fy, where="post", color="k", lw=1.2, alpha=0.55, zorder=2)

        ax.set_xscale("log")
        ax.set_xlabel("Latência fim-a-fim estimada drone→alerta (ms, log)")
        ax.set_ylabel(metric)
        ax.set_title(f"Enlace {bw} Mbit/s")
        if k == 0:
            from matplotlib.lines import Line2D
            handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[tag],
                              markeredgecolor="k", label=tag) for tag in caps]
            handles += [Line2D([0], [0], marker=marker, color="k", linestyle="none", label=model)
                        for model, marker in markers.items()]
            ax.legend(handles=handles, title="Captura / modelo", frameon=False, fontsize=7, ncol=2)

    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    fig.suptitle("Fronteira de Pareto: quanto recall se compra com cada milissegundo")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ==========================================================================
def knee_table(df: pd.DataFrame, link: dict, metric: str, floor: float,
               selection_rule: str) -> pd.DataFrame:
    """Lowest mean latency satisfying a predeclared recall constraint."""
    score = df[metric]
    if selection_rule == "ci95":
        score = score - df[f"{metric}_ci95_half"]
    ok = df.assign(selection_score=score)[score >= floor]
    out = []
    for bw in link["bandwidths_mbps"]:
        col = f"e2e_ms@{bw}"
        if ok.empty:
            out.append({"bandwidth_mbps": bw, "status": "nenhuma atinge o piso"})
            continue
        best = ok.loc[ok[col].idxmin()]
        out.append({
            "bandwidth_mbps": bw,
            "capture": best["capture_tag"],
            "imgsz": int(best["imgsz"]),
            "model": best["model"],
            "precision_mode": best["precision_mode"],
            "jpeg_quality": (None if pd.isna(best.get("jpeg_quality"))
                             else int(best["jpeg_quality"])),
            "n_seeds": int(best["n_seeds"]),
            metric: round(float(best[metric]), 4),
            "selection_score": round(float(best["selection_score"]), 4),
            "selection_rule": selection_rule,
            "payload_kB": round(float(best["mean_kb"]), 1),
            "estimated_e2e_ms": round(float(best[col]), 1),
        })
    return pd.DataFrame(out)


def write_frozen_manifest(path: Path, table: pd.DataFrame, source_csv: Path,
                          cfg: dict, metric: str, floor: float, rule: str) -> None:
    valid = table[table.get("status", pd.Series(index=table.index, dtype=object)).isna()]
    configurations = []
    seen = set()
    for _, row in valid.iterrows():
        key = (row["model"], row["capture"], int(row["imgsz"]))
        if key not in seen:
            seen.add(key)
            configurations.append({"model": key[0], "capture": key[1], "imgsz": key[2]})
    digest = hashlib.sha256(source_csv.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1, "selected_on": "val", "source_csv": str(source_csv),
        "source_sha256": digest, "metric": metric, "floor": floor,
        "experiment_config_sha256": hashlib.sha256(
            yaml.safe_dump(cfg, sort_keys=True).encode()
        ).hexdigest(),
        "selection_rule": rule, "conf_operating": cfg["eval"]["conf_operating"],
        "match_iou_operating": cfg["eval"].get("match_iou_operating", 0.5),
        "precision_modes": sorted(valid["precision_mode"].dropna().unique().tolist()),
        "jpeg_quality": cfg["transmission"].get("jpeg_quality"),
        "configurations": configurations,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=Path("results/grid.csv"))
    ap.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    ap.add_argument("--outdir", type=Path, default=Path("results/figs"))
    ap.add_argument("--metric", default=None,
                    choices=["recall_op", "recall_sweep", "map50", "map50_95"],
                    help="recall_op = ponto de operação real do sistema de alerta")
    ap.add_argument("--recall-floor", type=float, default=None,
                    help="recall mínimo aceitável para o alerta")
    ap.add_argument("--latency-csv", type=Path, default=None,
                    help="saída da grade com latência; padrão deriva de --csv")
    ap.add_argument("--knee-csv", type=Path, default=None,
                    help="saída da configuração ótima; padrão deriva de --csv")
    ap.add_argument("--selection-rule", choices=["ci95", "mean"], default=None,
                    help="ci95 exige que o limite inferior do IC95 atinja o piso")
    ap.add_argument("--frozen-manifest", type=Path, default=Path("results/frozen_configs.yaml"))
    ap.add_argument("--expected-seeds", type=int, default=None,
                    help="falha se uma configuração não tiver este número de seeds")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    link = cfg["link"]
    selection = cfg.get("selection", {})
    metric = args.metric or selection.get("metric", "recall_op")
    recall_floor = (args.recall_floor if args.recall_floor is not None
                    else float(selection.get("recall_floor", 0.60)))
    selection_rule = args.selection_rule or selection.get("rule", "ci95")
    if metric not in {"recall_op", "recall_sweep", "map50", "map50_95"}:
        raise ValueError(f"Métrica de seleção inválida: {metric}")
    if selection_rule not in {"ci95", "mean"}:
        raise ValueError(f"Regra de seleção inválida: {selection_rule}")

    raw = pd.read_csv(args.csv)
    if set(raw["split"].unique()) != {"val"}:
        raise ValueError("Seleção/Pareto só pode ser gerada a partir do split val")
    raw = add_latency_columns(raw, link)
    latency_metrics = [c for c in raw.columns if c.startswith("e2e_")]
    df = aggregate_seeds(raw, [metric, "precision_op", "map50", "map50_95", *latency_metrics])
    expected_seeds = args.expected_seeds or len(cfg.get("seeds", []))
    if expected_seeds and not (df["n_seeds"] == expected_seeds).all():
        incomplete = df[df["n_seeds"] != expected_seeds]
        raise ValueError(
            f"Grade incompleta: {len(incomplete)} configurações não possuem {expected_seeds} seeds"
        )
    args.outdir.mkdir(parents=True, exist_ok=True)

    m = metric
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

    kt = knee_table(df, link, m, recall_floor, selection_rule)
    kt.to_csv(knee_csv, index=False)
    write_frozen_manifest(args.frozen_manifest, kt, args.csv, cfg, m,
                          recall_floor, selection_rule)

    print(f"\nFiguras em {args.outdir}\n")
    print(f"Configuração de menor latência com {m} >= {recall_floor}:\n")
    print(kt.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
