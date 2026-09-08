from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
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
    info = {
        "device": device,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "host": platform.node(),
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
def run_val(model, data_yaml: Path, imgsz: int, conf: float, cfg_eval: dict,
            device: str, half: bool) -> dict:
    """Uma passada de validação. Retorna métricas + speed do Ultralytics."""
    r = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        batch=cfg_eval["batch"],
        conf=conf,
        iou=cfg_eval["iou_nms"],
        max_det=cfg_eval["max_det"],
        device=device,
        half=half,
        split="val",
        rect=False,        # rect=True muda o shape por batch e polui o timing
        plots=False,
        verbose=False,
        save_json=False,
    )
    b = r.box
    return {
        "recall": round(float(b.mr), 5),
        "precision": round(float(b.mp), 5),
        "map50": round(float(b.map50), 5),
        "map50_95": round(float(b.map), 5),
        "pre_ms": round(float(r.speed["preprocess"]), 3),
        "inf_ms": round(float(r.speed["inference"]), 3),
        "post_ms": round(float(r.speed["postprocess"]), 3),
    }


def read_manifest(split_dir: Path) -> dict:
    """Bytes médios por imagem — entra no modelo de transmissão."""
    m = split_dir / "manifest.csv"
    if not m.exists():
        return {"mean_kb": None, "mean_w": None, "mean_h": None, "n_images": None}
    import pandas as pd
    df = pd.read_csv(m)
    return {
        "mean_kb": round(df["bytes"].mean() / 1024, 2),
        "mean_w": int(df["width"].mean()),
        "mean_h": int(df["height"].mean()),
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
    ap.add_argument("--half", action="store_true", help="força FP16")
    ap.add_argument("--smoke", action="store_true",
                    help="1 modelo x 2 imgsz x 2 resoluções, iterações reduzidas")
    args = ap.parse_args()

    from ultralytics import YOLO  # import tardio: acelera --help

    cfg = yaml.safe_load(args.config.read_text())
    ds, ev, tm = cfg["dataset"], cfg["eval"], cfg["timing"]

    device = pick_device(args.device or cfg.get("device"))
    meta = device_banner(device)
    half = args.half or ev.get("half", False)

    derived_root = Path(ds["derived_root"])
    quality = cfg["transmission"].get("jpeg_quality")
    qsuf = "lossless" if quality is None else f"q{quality}"

    models = cfg["models"]
    sizes = cfg["inference_sizes"]
    caps = cfg["capture_resolutions"]

    if args.models:
        models = [m for m in models if m["name"] in args.models]
    if args.sizes:
        sizes = [s for s in sizes if s in args.sizes]
    if args.tags:
        caps = [c for c in caps if c["tag"] in args.tags]
    if args.smoke:
        models, sizes = models[:1], sizes[:2]
        caps = caps[:2]
        tm = {"warmup_iters": 5, "timed_iters": 20, "sync_cuda": True}

    for s in sizes:
        if s % 32:
            raise ValueError(f"imgsz={s} não é múltiplo de 32; o stride do YOLO exige.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = None
    rows = []

    total = len(models) * len(caps) * len(sizes)
    done = 0

    for mspec in models:
        wpath = resolve_weights(mspec["weights"])
        if wpath is None:
            print(f"[AVISO] pesos ausentes, pulando: {mspec['weights']}\n"
                  f"        (treine primeiro, ou use um nome do hub como 'yolo11n.pt')")
            continue
        print(f"\n=== carregando {mspec['name']} <- {wpath}")
        model = YOLO(wpath)

        for cap in caps:
            tag = cap["tag"]
            split_dir = derived_root / f"{ds['name']}_{ds['split']}_{tag}_{qsuf}"
            data_yaml = derived_root / f"{ds['name']}_{ds['split']}_{tag}_{qsuf}.yaml"
            if not data_yaml.exists():
                print(f"[AVISO] rode downsample.py antes; falta {data_yaml}")
                continue
            man = read_manifest(split_dir)

            for imgsz in sizes:
                done += 1
                print(f"\n[{done}/{total}] {mspec['name']} | captura={tag} | imgsz={imgsz}")

                lat = time_forward(model, imgsz, device,
                                   tm["warmup_iters"], tm["timed_iters"], half)
                cx = model_complexity(model, imgsz)

                sweep = run_val(model, data_yaml, imgsz, ev["conf_sweep"], ev, device, half)
                oper = run_val(model, data_yaml, imgsz, ev["conf_operating"], ev, device, half)

                row = {
                    "model": mspec["name"],
                    "weights": wpath,
                    "capture_tag": tag,
                    "capture_h": cap.get("height"),
                    "mean_w": man["mean_w"],
                    "mean_h": man["mean_h"],
                    "mean_kb": man["mean_kb"],
                    "n_images": man["n_images"],
                    "imgsz": imgsz,
                    "precision_mode": "fp16" if half else "fp32",
                    "jpeg_quality": quality,
                    # ponto de operação real (o que importa para o alerta)
                    "recall_op": oper["recall"],
                    "precision_op": oper["precision"],
                    "map50_op": oper["map50"],
                    # varredura completa (comparável com papers)
                    "recall_sweep": sweep["recall"],
                    "precision_sweep": sweep["precision"],
                    "map50": sweep["map50"],
                    "map50_95": sweep["map50_95"],
                    # latência
                    "fwd_median_ms": lat["fwd_median_ms"],
                    "fwd_p95_ms": lat["fwd_p95_ms"],
                    "fwd_std_ms": lat["fwd_std_ms"],
                    "pre_ms": sweep["pre_ms"],
                    "inf_ms": sweep["inf_ms"],
                    "post_ms": sweep["post_ms"],
                    "params_M": cx["params_M"],
                    "gflops": cx["gflops"],
                    "device": meta.get("gpu_name", device),
                    "torch": meta["torch"],
                }
                rows.append(row)

                # grava a cada célula: 40 combinações demoram, não perca tudo
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

    print(f"\nGrade salva em {args.out}  ({len(rows)} linhas)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
    