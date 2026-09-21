from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml


# --------------------------------------------------------------------------
def pick_device(cfg_device: str | int | None) -> str:
    if cfg_device is not None and str(cfg_device) != "auto":
        return str(cfg_device)
    return "0" if torch.cuda.is_available() else "cpu"


_HUB_PREFIXES = ("yolo", "rtdetr", "sam", "fastsam", "mobile_sam")


def resolve_weights(spec: str) -> str | None:
    """Aceita caminho em disco OU nome curto que o Ultralytics baixa sozinho.

    'runs/train/.../best.pt' -> precisa existir.
    'yolo11n.pt'             -> nome do hub, é baixado na primeira execução.
    """
    p = Path(spec)
    if p.exists():
        return str(p)
    is_bare_name = p.parent in (Path("."), Path(""))
    if is_bare_name and p.suffix == ".pt" and p.name.lower().startswith(_HUB_PREFIXES):
        return p.name
    return None


def device_banner(device: str) -> dict:
    cpu_name = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if not cpu_name and cpuinfo.exists():
        for line in cpuinfo.read_text(errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                cpu_name = line.split(":", 1)[-1].strip()
                break
    info = {
        "device": device,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "host": platform.node(),
        "cpu": cpu_name or "desconhecido",
        "platform": platform.platform(),
    }
    if device != "cpu" and torch.cuda.is_available():
        idx = int(device)
        info["gpu_name"] = torch.cuda.get_device_name(idx)
        cc = torch.cuda.get_device_capability(idx)
        info["compute_capability"] = f"{cc[0]}.{cc[1]}"
        # sm_120 = Blackwell (RTX 50xx). Se aparecer erro de "no kernel image",
        # o torch foi compilado sem sm_120 -> imagem Docker errada.
        if cc[0] >= 12:
            info["arch"] = "blackwell(sm_120)"
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return info


# --------------------------------------------------------------------------
@torch.inference_mode()
def time_forward(model, imgsz: int, device: str, warmup: int, iters: int,
                 half: bool) -> dict:
    """Latência do forward puro, batch=1. Sem IO, sem NMS, sem letterbox."""
    net = model.model
    net = net.half() if half else net.float()
    net.eval().to(device if device == "cpu" else f"cuda:{device}")

    dev = torch.device("cpu" if device == "cpu" else f"cuda:{device}")
    x = torch.randn(1, 3, imgsz, imgsz, device=dev,
                    dtype=torch.half if half else torch.float32)

    sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else (lambda: None)

    for _ in range(warmup):
        net(x)
    sync()

    samples = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        net(x)
        sync()
        samples.append((time.perf_counter() - t0) * 1000.0)

    samples.sort()
    return {
        "fwd_mean_ms": round(statistics.fmean(samples), 3),
        "fwd_median_ms": round(statistics.median(samples), 3),
        "fwd_p95_ms": round(samples[int(0.95 * len(samples)) - 1], 3),
        "fwd_std_ms": round(statistics.pstdev(samples), 3),
    }


def model_complexity(model, imgsz: int) -> dict:
    """Params e GFLOPs — o que de fato prediz o custo na NPU."""
    try:
        from ultralytics.utils.torch_utils import get_flops
        n_params = sum(p.numel() for p in model.model.parameters())
        return {"params_M": round(n_params / 1e6, 3),
                "gflops": round(float(get_flops(model.model, imgsz)), 2)}
    except Exception:
        return {"params_M": None, "gflops": None}


# --------------------------------------------------------------------------
def _metric_array(value) -> list[float]:
    if value is None:
        return []
    return [float(x) for x in np.asarray(value).reshape(-1)]


def _exact_operating_metrics(matrix: np.ndarray) -> dict:
    """Precision/recall at the requested confidence and IoU, not max-F1."""
    matrix = np.asarray(matrix, dtype=float)
    nc = matrix.shape[0] - 1
    tp = np.diag(matrix[:nc, :nc])
    predicted = matrix[:nc, :].sum(axis=1)
    targets = matrix[:, :nc].sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros(nc), where=predicted > 0)
    recall = np.divide(tp, targets, out=np.zeros(nc), where=targets > 0)
    evaluated = targets > 0
    return {
        "precision": float(np.mean(precision[evaluated])) if np.any(evaluated) else 0.0,
        "recall": float(np.mean(recall[evaluated])) if np.any(evaluated) else 0.0,
        "precision_micro": float(tp.sum() / predicted.sum()) if predicted.sum() else 0.0,
        "recall_micro": float(tp.sum() / targets.sum()) if targets.sum() else 0.0,
        "class_precision": precision.tolist(),
        "class_recall": recall.tolist(),
        "tp": int(tp.sum()),
        "fp": int(predicted.sum() - tp.sum()),
        "fn": int(targets.sum() - tp.sum()),
    }


def _operating_validator(match_iou: float):
    """Build a validator that records a confusion matrix at an explicit IoU."""
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils.metrics import ConfusionMatrix

    class OperatingPointValidator(DetectionValidator):
        def init_metrics(self, model) -> None:
            super().init_metrics(model)
            self.operating_matrix = ConfusionMatrix(names=model.names)

        def update_metrics(self, preds, batch) -> None:
            super().update_metrics(preds, batch)
            for si, pred in enumerate(preds):
                pbatch = self._prepare_batch(si, batch)
                predn = self._prepare_pred(pred.copy())
                self.operating_matrix.process_batch(
                    predn, pbatch, conf=float(self.args.conf), iou_thres=match_iou
                )

        def finalize_metrics(self) -> None:
            super().finalize_metrics()
            self.metrics.operating_matrix = self.operating_matrix.matrix.copy()

    return OperatingPointValidator


def run_val(model, data_yaml: Path, imgsz: int, conf: float, cfg_eval: dict,
            device: str, half: bool, split: str, plot_dir: Path | None = None,
            make_plots: bool = False, exact_operating: bool = False) -> dict:
    """Uma passada de validação. Retorna métricas + speed do Ultralytics."""
    val_args = dict(
        data=str(data_yaml),
        imgsz=imgsz,
        batch=cfg_eval["batch"],
        conf=conf,
        iou=cfg_eval["iou_nms"],
        max_det=cfg_eval["max_det"],
        device=device,
        half=half,
            # Cada YAML derivado aponta seu único conjunto avaliado para val.
            split="val",
        rect=False,        # rect=True muda o shape por batch e polui o timing
        plots=make_plots,
        verbose=False,
        save_json=False,
        project=str(plot_dir.parent) if plot_dir else None,
        name=plot_dir.name if plot_dir else None,
    )
    if exact_operating:
        validator = _operating_validator(float(cfg_eval.get("match_iou_operating", 0.5)))
        r = model.val(validator=validator, **val_args)
    else:
        r = model.val(**val_args)
    b = r.box
    class_indices = [int(x) for x in np.asarray(getattr(b, "ap_class_index", []))]
    recalls = _metric_array(getattr(b, "r", None))
    precisions = _metric_array(getattr(b, "p", None))
    ap50 = _metric_array(getattr(b, "ap50", None))
    ap5095 = _metric_array(getattr(b, "ap", None))
    result = {
        "recall": round(float(b.mr), 5),
        "precision": round(float(b.mp), 5),
        "map50": round(float(b.map50), 5),
        "map50_95": round(float(b.map), 5),
        "pre_ms": round(float(r.speed["preprocess"]), 3),
        "inf_ms": round(float(r.speed["inference"]), 3),
        "post_ms": round(float(r.speed["postprocess"]), 3),
        "class_indices": class_indices,
        "class_recall": recalls,
        "class_precision": precisions,
        "class_ap50": ap50,
        "class_ap50_95": ap5095,
        "plots_dir": str(plot_dir) if plot_dir else None,
    }
    if exact_operating:
        if not hasattr(r, "operating_matrix"):
            raise RuntimeError("Ultralytics não devolveu a matriz operacional exata")
        op = _exact_operating_metrics(r.operating_matrix)
        result.update({
            "recall": round(op["recall"], 5),
            "precision": round(op["precision"], 5),
            "recall_micro": round(op["recall_micro"], 5),
            "precision_micro": round(op["precision_micro"], 5),
            "class_recall": op["class_recall"],
            "class_precision": op["class_precision"],
            "tp": op["tp"], "fp": op["fp"], "fn": op["fn"],
        })
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_provenance() -> tuple[str, bool | None]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip())
        return commit, dirty
    except Exception:
        return "desconhecido", None


def read_frozen_configs(path: Path, cfg: dict) -> set[tuple[str, str, int]]:
    manifest = yaml.safe_load(path.read_text()) or {}
    if manifest.get("schema_version") != 1:
        raise ValueError("Versão desconhecida do manifesto congelado")
    cfg_digest = hashlib.sha256(yaml.safe_dump(cfg, sort_keys=True).encode()).hexdigest()
    if manifest.get("experiment_config_sha256") != cfg_digest:
        raise ValueError("configs/experiment.yaml mudou depois do congelamento")
    expected_conf = float(cfg["eval"]["conf_operating"])
    if float(manifest.get("conf_operating", -1)) != expected_conf:
        raise ValueError("Threshold do manifesto congelado difere de eval.conf_operating")
    if manifest.get("selected_on") != "val":
        raise ValueError("O manifesto de teste precisa ter sido selecionado exclusivamente em val")
    expected_iou = float(cfg["eval"].get("match_iou_operating", 0.5))
    if float(manifest.get("match_iou_operating", -1)) != expected_iou:
        raise ValueError("IoU operacional do manifesto difere da configuração atual")
    expected_precision = "fp16" if cfg["eval"].get("half", False) else "fp32"
    if manifest.get("precision_modes") != [expected_precision]:
        raise ValueError("Precisão numérica do manifesto difere da configuração atual")
    if manifest.get("jpeg_quality") != cfg["transmission"].get("jpeg_quality"):
        raise ValueError("Qualidade JPEG do manifesto difere da configuração atual")
    source = Path(manifest.get("source_csv", ""))
    if not source.is_file() or file_sha256(source) != manifest.get("source_sha256"):
        raise ValueError("A grade de validação que originou o manifesto está ausente ou mudou")
    combos = {
        (str(item["model"]), str(item["capture"]), int(item["imgsz"]))
        for item in manifest.get("configurations", [])
    }
    if not combos:
        raise ValueError("Manifesto congelado não contém configurações elegíveis")
    return combos


def read_manifest(split_dir: Path) -> dict:
    """Bytes médios por imagem — entra no modelo de transmissão."""
    m = split_dir / "manifest.csv"
    if not m.exists():
        return {"mean_kb": None, "mean_w": None, "mean_h": None,
                "mean_pixels": None, "n_images": None}
    import pandas as pd
    df = pd.read_csv(m)
    return {
        # Decimal kB keeps the radio formula kB*8/Mbit_s = ms exact.
        "mean_kb": round(df["bytes"].mean() / 1000, 3),
        "mean_w": int(df["width"].mean()),
        "mean_h": int(df["height"].mean()),
        "mean_pixels": float(df["pixels"].mean()),
        "n_images": len(df),
    }


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    ap.add_argument("--out", type=Path, default=Path("results/grid.csv"))
    ap.add_argument("--models", nargs="*", default=None, help="filtra por nome")
    ap.add_argument("--sizes", nargs="*", type=int, default=None, help="filtra imgsz")
    ap.add_argument("--tags", nargs="*", default=None, help="filtra resoluções de captura")
    ap.add_argument("--device", default=None)
    ap.add_argument("--split", default=None, choices=["val", "test"],
                    help="split avaliado; por padrão usa dataset.split")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--weights-root", type=Path, default=None,
                    help="raiz dos checkpoints: runs/train/model_visdrone_seed_N/weights/best.pt")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--half", action="store_true", help="força FP16")
    ap.add_argument("--frozen-config", type=Path, default=None,
                    help="manifesto gerado em val; obrigatório para avaliar test")
    ap.add_argument("--smoke", action="store_true",
                    help="1 modelo x 2 imgsz x 2 resoluções, iterações reduzidas")
    args = ap.parse_args()

    import ultralytics
    from ultralytics import YOLO  # import tardio: acelera --help

    cfg = yaml.safe_load(args.config.read_text())
    ds, ev, tm = cfg["dataset"], cfg["eval"], cfg["timing"]
    split = args.split or ds["split"]
    if split == "test" and args.frozen_config is None:
        raise ValueError("Avaliar test exige --frozen-config selecionado em val")
    frozen = read_frozen_configs(args.frozen_config, cfg) if args.frozen_config else None

    device = pick_device(args.device or cfg.get("device"))
    meta = device_banner(device)
    meta["ultralytics"] = ultralytics.__version__
    meta["python"] = platform.python_version()
    half = args.half or ev.get("half", False)

    derived_root = Path(ds["derived_root"])
    quality = cfg["transmission"].get("jpeg_quality")
    qsuf = "lossless" if quality is None else f"q{quality}"

    models = cfg["models"]
    seeds = args.seeds if args.seeds is not None else cfg.get("seeds", [0])
    if args.weights_root is None and len(seeds) > 1 and not args.smoke:
        raise ValueError("Múltiplas seeds exigem --weights-root; repetir pesos do hub cria pseudorreplicação")
    sizes = cfg["inference_sizes"]
    caps = cfg["capture_resolutions"]

    if args.models:
        models = [m for m in models if m["name"] in args.models]
    if args.sizes:
        sizes = [s for s in sizes if s in args.sizes]
    if args.tags:
        caps = [c for c in caps if c["tag"] in args.tags]
    if args.smoke:
        models, sizes, seeds = models[:1], sizes[:2], seeds[:1]
        caps = caps[:2]
        tm = {"warmup_iters": 5, "timed_iters": 20}

    for s in sizes:
        if s % 32:
            raise ValueError(f"imgsz={s} não é múltiplo de 32; o stride do YOLO exige.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = None
    rows = []

    total = len(models) * len(seeds) * len(caps) * len(sizes)
    done = 0
    class_rows = []
    timing_cache = {}
    config_hash = file_sha256(args.config)
    git_commit, git_dirty = git_provenance()

    for mspec in models:
        for seed in seeds:
            if args.weights_root:
                run_name = f"{mspec['name']}_visdrone_seed_{seed}" + ("_smoke" if args.smoke else "")
                candidate = args.weights_root / run_name / "weights" / "best.pt"
                wpath = str(candidate) if candidate.exists() else None
            else:
                wpath = resolve_weights(mspec["weights"])
            if wpath is None:
                print(f"[AVISO] pesos ausentes para {mspec['name']} seed={seed}; pulando")
                continue
            print(f"\n=== carregando {mspec['name']} seed={seed} <- {wpath}")
            model = YOLO(wpath)
            weights_hash = file_sha256(Path(wpath)) if Path(wpath).is_file() else None

            for cap in caps:
                tag = cap["tag"]
                split_dir = derived_root / f"{ds['name']}_{split}_{tag}_{qsuf}"
                data_yaml = derived_root / f"{ds['name']}_{split}_{tag}_{qsuf}.yaml"
                if not data_yaml.exists():
                    print(f"[AVISO] rode downsample.py antes; falta {data_yaml}")
                    continue
                man = read_manifest(split_dir)

                for imgsz in sizes:
                    if frozen is not None and (mspec["name"], tag, imgsz) not in frozen:
                        continue
                    done += 1
                    print(f"\n[{done}/{total}] {mspec['name']} seed={seed} | captura={tag} | imgsz={imgsz}")

                    timing_key = (mspec["name"], seed, imgsz, half)
                    if timing_key not in timing_cache:
                        timing_cache[timing_key] = (
                            time_forward(model, imgsz, device,
                                         tm["warmup_iters"], tm["timed_iters"], half),
                            model_complexity(model, imgsz),
                        )
                    lat, cx = timing_cache[timing_key]

                    plot_dir = Path("results/pr") / split / f"{mspec['name']}_seed_{seed}_{tag}_{imgsz}"
                    sweep = run_val(model, data_yaml, imgsz, ev["conf_sweep"], ev, device, half,
                                    split, plot_dir, not args.no_plots)
                    oper = run_val(model, data_yaml, imgsz, ev["conf_operating"], ev, device, half,
                                   split, exact_operating=True)

                    row = {
                        "model": mspec["name"],
                        "seed": seed,
                        "split": split,
                        "weights": wpath,
                        "capture_tag": tag,
                        "capture_h": cap.get("height"),
                        "mean_w": man["mean_w"],
                        "mean_h": man["mean_h"],
                        "mean_pixels": man["mean_pixels"],
                        "mean_kb": man["mean_kb"],
                        "n_images": man["n_images"],
                        "imgsz": imgsz,
                        "precision_mode": "fp16" if half else "fp32",
                        "jpeg_quality": quality,
                        "conf_operating": ev["conf_operating"],
                        "match_iou_operating": ev.get("match_iou_operating", 0.5),
                        "recall_op": oper["recall"],
                        "precision_op": oper["precision"],
                        "recall_micro_op": oper["recall_micro"],
                        "precision_micro_op": oper["precision_micro"],
                        "tp_op": oper["tp"], "fp_op": oper["fp"], "fn_op": oper["fn"],
                        "recall_sweep": sweep["recall"],
                        "precision_sweep": sweep["precision"],
                        "map50": sweep["map50"],
                        "map50_95": sweep["map50_95"],
                        "fwd_median_ms": lat["fwd_median_ms"],
                        "fwd_p95_ms": lat["fwd_p95_ms"],
                        "fwd_std_ms": lat["fwd_std_ms"],
                        "pre_ms": sweep["pre_ms"],
                        "inf_ms": sweep["inf_ms"],
                        "post_ms": sweep["post_ms"],
                        "pre_op_ms": oper["pre_ms"],
                        "inf_op_ms": oper["inf_ms"],
                        "post_op_ms": oper["post_ms"],
                        "params_M": cx["params_M"],
                        "gflops": cx["gflops"],
                        "device": meta.get("gpu_name", device),
                        "cpu": meta["cpu"],
                        "platform": meta["platform"],
                        "torch": meta["torch"],
                        "ultralytics": meta["ultralytics"],
                        "python": meta["python"],
                        "cuda_runtime": meta.get("cuda_runtime"),
                        "config_sha256": config_hash,
                        "weights_sha256": weights_hash,
                        "git_commit": git_commit,
                        "git_dirty": git_dirty,
                        "pr_plot_dir": sweep["plots_dir"],
                    }
                    rows.append(row)

                    names_cfg = yaml.safe_load(data_yaml.read_text()).get("names", [])
                    metric_values = (
                        ("recall", range(len(oper["class_recall"])), oper["class_recall"]),
                        ("precision", range(len(oper["class_precision"])), oper["class_precision"]),
                        ("map50", sweep["class_indices"], sweep["class_ap50"]),
                        ("map50_95", sweep["class_indices"], sweep["class_ap50_95"]),
                    )
                    for metric_name, indices, values in metric_values:
                        for class_index, value in zip(indices, values):
                            if np.isnan(value):
                                continue
                            class_rows.append({
                                "model": mspec["name"], "seed": seed, "split": split,
                                "capture_tag": tag, "imgsz": imgsz,
                                "class_id": class_index,
                                "class_name": names_cfg[class_index] if class_index < len(names_cfg) else str(class_index),
                                "metric": metric_name, "value": round(value, 6),
                            })

                    # Grava a cada célula: a grade completa é longa.
                    fieldnames = list(row)
                    with open(args.out, "w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=fieldnames)
                        w.writeheader()
                        w.writerows(rows)

                    print(f"      recall@op={row['recall_op']:.3f}  "
                          f"mAP50={row['map50']:.3f}  "
                          f"fwd={row['fwd_median_ms']:.2f}ms  "
                          f"payload={row['mean_kb']} kB")

    if not rows:
        print("\nERRO: nenhuma célula foi avaliada — nada a salvar.\n"
              "Causas prováveis:\n"
              "  * pesos não encontrados (veja os avisos acima)\n"
              "  * splits degradados ausentes -> rode downsample.py antes")
        return 1

    class_out = args.out.with_name(f"{args.out.stem}_per_class.csv")
    with open(class_out, "w", newline="") as f:
        fields = ["model", "seed", "split", "capture_tag", "imgsz", "class_id", "class_name", "metric", "value"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(class_rows)

    print(f"\nGrade salva em {args.out}  ({len(rows)} linhas)")
    print(f"Métricas por classe em {class_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
