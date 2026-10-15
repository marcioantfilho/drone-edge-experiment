"""Salva predições e ground truth para análise offline, pelo mesmo caminho do benchmark.

Usa a mesma subclasse de DetectionValidator do benchmark.py (mesmo imgsz, rect=False,
iou_nms=0.7, max_det=1000), mas com conf=0.001. Para cada imagem grava as caixas no
espaço letterbox usado pela matriz de confusão do protocolo e também convertidas para
pixels da imagem original. Na mesma passada calcula o recall_op em conf=0.25/IoU=0.50,
que serve de alvo para verificar o matcher offline.

Funciona com .pt no servidor e com .engine (Jetson) ou *_rknn_model (RK3588) via --weights.
"""
from __future__ import annotations

import argparse
import csv
import platform
from pathlib import Path

import numpy as np
import yaml

from benchmark import (PROTOCOL_CONF_OPERATING, PROTOCOL_MATCH_IOU, _exact_operating_metrics,
                       check_operating_point, file_sha256, git_provenance, pick_device)

DUMP_CONF = 0.001


def _np(x, dtype=np.float32) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype)


def _dump_validator():
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils import ops
    from ultralytics.utils.metrics import ConfusionMatrix

    class DumpValidator(DetectionValidator):
        def init_metrics(self, model) -> None:
            super().init_metrics(model)
            self.operating_matrix = ConfusionMatrix(names=model.names)
            self.dump = {"images": [], "pred": [], "gt": []}

        def update_metrics(self, preds, batch) -> None:
            super().update_metrics(preds, batch)
            for si, pred in enumerate(preds):
                pbatch = self._prepare_batch(si, batch)
                predn = self._prepare_pred(pred.copy())
                self.operating_matrix.process_batch(
                    predn, pbatch, conf=PROTOCOL_CONF_OPERATING, iou_thres=PROTOCOL_MATCH_IOU
                )
                imgsz = tuple(int(v) for v in batch["img"].shape[2:])
                ori_shape = tuple(int(v) for v in batch["ori_shape"][si])
                ratio_pad = batch["ratio_pad"][si]

                def to_orig(boxes):
                    if len(boxes) == 0:
                        return np.zeros((0, 4), np.float32)
                    return _np(ops.scale_boxes(imgsz, boxes.clone(), ori_shape, ratio_pad=ratio_pad))

                idx = len(self.dump["images"])
                self.dump["images"].append({
                    "im_file": str(batch["im_file"][si]),
                    "ori_h": ori_shape[0], "ori_w": ori_shape[1],
                    "gain": float(_np(ratio_pad[0]).reshape(-1)[0]),
                    "pad_w": float(_np(ratio_pad[1]).reshape(-1)[0]),
                    "pad_h": float(_np(ratio_pad[1]).reshape(-1)[1]),
                })
                self.dump["pred"].append((
                    idx, _np(predn["bboxes"]).reshape(-1, 4), to_orig(predn["bboxes"]),
                    _np(predn["conf"]).reshape(-1), _np(predn["cls"], np.int16).reshape(-1),
                ))
                self.dump["gt"].append((
                    idx, _np(pbatch["bboxes"]).reshape(-1, 4), to_orig(pbatch["bboxes"]),
                    _np(pbatch["cls"], np.int16).reshape(-1),
                ))

        def finalize_metrics(self) -> None:
            super().finalize_metrics()
            self.metrics.operating_matrix = self.operating_matrix.matrix.copy()
            self.metrics.dump = self.dump

    return DumpValidator


def _stack(rows: list, n_box_cols: int) -> dict:
    """rows: (idx, xyxy, xyxy_orig, [conf], cls) por imagem -> arrays concatenados."""
    img = np.concatenate([np.full(len(r[-1]), r[0], np.int32) for r in rows]) if rows else np.zeros(0, np.int32)
    out = {"img": img,
           "xyxy": np.concatenate([r[1] for r in rows]) if rows else np.zeros((0, 4), np.float32),
           "xyxy_orig": np.concatenate([r[2] for r in rows]) if rows else np.zeros((0, 4), np.float32),
           "cls": np.concatenate([r[-1] for r in rows]) if rows else np.zeros(0, np.int16)}
    if n_box_cols == 5:
        out["conf"] = np.concatenate([r[3] for r in rows]) if rows else np.zeros(0, np.float32)
    return out


def dump_one(weights: str, data_yaml: Path, imgsz: int, ev: dict, device: str, out_file: Path) -> dict:
    from ultralytics import YOLO

    model = YOLO(weights, task="detect")
    r = model.val(
        validator=_dump_validator(), data=str(data_yaml), imgsz=imgsz, batch=ev["batch"],
        conf=DUMP_CONF, iou=ev["iou_nms"], max_det=ev["max_det"], device=device,
        half=ev.get("half", False), split="val", rect=False, plots=False, verbose=False,
        save_json=False,
    )
    dump = r.dump
    pred = _stack(dump["pred"], 5)
    gt = _stack(dump["gt"], 4)
    imgs = dump["images"]
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_file,
        image=np.array([Path(i["im_file"]).name for i in imgs]),
        ori_hw=np.array([[i["ori_h"], i["ori_w"]] for i in imgs], np.int32),
        gain=np.array([i["gain"] for i in imgs], np.float32),
        pad_wh=np.array([[i["pad_w"], i["pad_h"]] for i in imgs], np.float32),
        **{f"pred_{k}": v for k, v in pred.items()},
        **{f"gt_{k}": v for k, v in gt.items()},
    )
    op = _exact_operating_metrics(r.operating_matrix)
    return {
        "n_images": len(imgs), "n_pred": len(pred["img"]), "n_gt": len(gt["img"]),
        "recall_op_inrun": round(op["recall"], 5), "precision_op_inrun": round(op["precision"], 5),
        "recall_micro_op_inrun": round(op["recall_micro"], 5),
        "tp_op": op["tp"], "fp_op": op["fp"], "fn_op": op["fn"],
        "map50": round(float(r.box.map50), 5), "map50_95": round(float(r.box.map), 5),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    ap.add_argument("--weights-root", type=Path, default=Path("runs/train"))
    ap.add_argument("--weights", default=None,
                    help="arquivo/diretório de pesos explícito (.pt, .engine, *_rknn_model); exige uma seed")
    ap.add_argument("--label", default=None, help="nome do modelo na saída; padrão = --models[0]")
    ap.add_argument("--models", nargs="*", default=["yolo11s"])
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--tags", nargs="*", default=["native"])
    ap.add_argument("--sizes", nargs="*", type=int, default=[640, 960, 1280])
    ap.add_argument("--derived-root", type=Path, default=None, help="sobrescreve dataset.derived_root")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("results/predictions"))
    args = ap.parse_args()

    import ultralytics

    cfg = yaml.safe_load(args.config.read_text())
    ds, ev = cfg["dataset"], cfg["eval"]
    check_operating_point(ev)
    if ds["split"] != "val":
        raise ValueError("save_predictions só roda em val; o test-dev não é tocado")
    seeds = args.seeds if args.seeds is not None else cfg["seeds"]
    if args.weights and len(seeds) != 1:
        raise ValueError("--weights explícito exige exatamente uma --seeds")
    derived_root = args.derived_root or Path(ds["derived_root"])
    quality = cfg["transmission"].get("jpeg_quality")
    qsuf = "lossless" if quality is None else f"q{quality}"
    device = pick_device(args.device or cfg.get("device"))
    git_commit, git_dirty = git_provenance()

    index = args.out_dir / "index.csv"
    write_header = not index.exists()
    for model_name in ([args.label or args.models[0]] if args.weights else args.models):
        for seed in seeds:
            weights = args.weights or str(args.weights_root / f"{model_name}_visdrone_seed_{seed}" / "weights" / "best.pt")
            if not Path(weights).exists():
                print(f"[AVISO] pesos ausentes: {weights}; pulando")
                continue
            wfile = Path(weights)
            whash = file_sha256(wfile) if wfile.is_file() else None
            for tag in args.tags:
                data_yaml = derived_root / f"{ds['name']}_val_{tag}_{qsuf}.yaml"
                if not data_yaml.exists():
                    raise FileNotFoundError(f"falta {data_yaml}; rode downsample.py antes")
                for imgsz in args.sizes:
                    stem = f"{model_name}_seed{seed}_{tag}_{imgsz}_{wfile.suffix.lstrip('.') or 'dir'}"
                    out_file = args.out_dir / f"{stem}.npz"
                    if out_file.exists():
                        print(f"[SKIP] {out_file} já existe")
                        continue
                    print(f"\n=== {stem}")
                    stats = dump_one(weights, data_yaml, imgsz, ev, device, out_file)
                    row = {"model": model_name, "seed": seed, "capture_tag": tag, "imgsz": imgsz,
                           "weights": weights, "weights_sha256": whash, "format": wfile.suffix or "dir",
                           "conf_dump": DUMP_CONF, "conf_operating": PROTOCOL_CONF_OPERATING,
                           "match_iou_operating": PROTOCOL_MATCH_IOU, **stats,
                           "file": str(out_file), "file_sha256": file_sha256(out_file),
                           "ultralytics": ultralytics.__version__, "host": platform.node(),
                           "git_commit": git_commit, "git_dirty": git_dirty}
                    with index.open("a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=list(row))
                        if write_header:
                            w.writeheader()
                            write_header = False
                        w.writerow(row)
                    print(f"    recall_op@0.25 (mesma passada) = {stats['recall_op_inrun']:.4f}  "
                          f"pred={stats['n_pred']}  gt={stats['n_gt']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
