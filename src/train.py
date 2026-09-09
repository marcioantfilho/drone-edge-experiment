from __future__ import annotations

import argparse
import csv
import datetime
import subprocess
import sys
from pathlib import Path

import yaml

LOG_FIELDS = [
    "timestamp", "model", "seed", "pretrained_weights", "dataset", "smoke",
    "imgsz", "batch", "epochs_requested", "epochs_completed",
    "device", "git_commit", "git_dirty",
    "precision", "recall", "map50", "map50_95",
    "run_dir", "best_weights",
]


# --------------------------------------------------------------------------
def git_provenance() -> dict:
    """Commit e estado do worktree no momento do treino.

    Se o worktree estiver sujo, o commit sozinho não basta para reproduzir
    o checkpoint -- é preciso saber disso na hora de escrever a dissertação.
    """
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    commit = run("rev-parse", "HEAD")
    dirty = run("status", "--porcelain")
    return {
        "git_commit": commit or "desconhecido",
        "git_dirty": bool(dirty) if dirty is not None else None,
    }


def pick_device(cfg_device) -> str:
    import torch
    if cfg_device is not None and str(cfg_device) != "auto":
        return str(cfg_device)
    return "0" if torch.cuda.is_available() else "cpu"


def resolve_weights(spec: str) -> str:
    """Caminho local, se existir; senão assume nome do hub Ultralytics (baixa sozinho)."""
    p = Path(spec)
    return str(p) if p.exists() else spec


def append_log(log_path: Path, row: dict) -> None:
    """Acrescenta uma linha; nunca sobrescreve treinos anteriores."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(row)


# --------------------------------------------------------------------------
def train_one(mspec: dict, cfg: dict, args: argparse.Namespace, prov: dict, seed: int) -> dict:
    from ultralytics import YOLO

    tr = dict(cfg["train"])
    smoke_over = cfg.get("smoke", {}) if args.smoke else {}
    tr.update(smoke_over)

    device = args.device or tr.pop("device", None)
    device = pick_device(device)

    weights = resolve_weights(mspec["weights"])
    name = f"{mspec['name']}_visdrone_seed_{seed}" + ("_smoke" if args.smoke else "")

    print(f"\n=== treinando {mspec['name']} <- {weights}  "
          f"(device={device}, epochs={tr['epochs']}, imgsz={tr['imgsz']}, smoke={args.smoke})")

    model = YOLO(weights)
    results = model.train(
        data=cfg["dataset"]["base_yaml"],
        imgsz=tr["imgsz"],
        epochs=tr["epochs"],
        batch=tr["batch"],
        workers=tr["workers"],
        amp=tr.get("amp", True),
        cache=tr.get("cache", "disk"),
        fraction=tr.get("fraction", 1.0),
        max_det=tr.get("max_det", 300),
        device=device,
        seed=seed,
        project=str(Path("runs/train").resolve()),
        name=name,
        exist_ok=True,
    )

    box = results.box
    epochs_completed = getattr(getattr(model, "trainer", None), "epoch", None)
    epochs_completed = (epochs_completed + 1) if epochs_completed is not None else tr["epochs"]

    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"

    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": mspec["name"],
        "seed": seed,
        "pretrained_weights": weights,
        "dataset": cfg["dataset"]["base_yaml"],
        "smoke": args.smoke,
        "imgsz": tr["imgsz"],
        "batch": tr["batch"],
        "epochs_requested": tr["epochs"],
        "epochs_completed": epochs_completed,
        "device": device,
        "git_commit": prov["git_commit"],
        "git_dirty": prov["git_dirty"],
        "precision": round(float(box.mp), 5),
        "recall": round(float(box.mr), 5),
        "map50": round(float(box.map50), 5),
        "map50_95": round(float(box.map), 5),
        "run_dir": str(run_dir),
        "best_weights": str(best) if best.exists() else None,
    }

    print(f"    -> mAP50={row['map50']:.4f}  mAP50-95={row['map50_95']:.4f}  "
          f"P={row['precision']:.4f}  R={row['recall']:.4f}")
    print(f"    -> pesos em {row['best_weights']}")
    return row


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    ap.add_argument("--log", type=Path, default=Path("results/train_log.csv"))
    ap.add_argument("--models", nargs="*", default=None, help="filtra por nome")
    ap.add_argument("--device", default=None, help="sobrescreve o device do config")
    ap.add_argument("--seeds", nargs="*", type=int, default=None,
                    help="sobrescreve as seeds do YAML")
    ap.add_argument("--smoke", action="store_true",
                    help="2 épocas, poucas imagens (fração do split) -- valida o pipeline no PC local")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    models = cfg["models"]
    if args.models:
        models = [m for m in models if m["name"] in args.models]
    if not models:
        print("Nenhum modelo selecionado (confira --models contra configs/train.yaml).")
        return 1

    seeds = args.seeds if args.seeds else cfg.get("seeds", [cfg.get("seed", 0)])
    if args.smoke:
        seeds = cfg.get("smoke", {}).get("seeds", [seeds[0]])

    prov = git_provenance()
    if prov["git_dirty"]:
        print("[AVISO] worktree com mudanças não commitadas -- o commit registrado "
              "no log NAO reproduz exatamente este checkpoint.")

    for mspec in models:
        for seed in seeds:
            row = train_one(mspec, cfg, args, prov, seed)
            append_log(args.log, row)

    print(f"\nProveniência registrada em {args.log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
