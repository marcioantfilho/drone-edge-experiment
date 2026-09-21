from __future__ import annotations
 
import argparse
import shutil
import sys
from pathlib import Path
 
import cv2
import yaml
from tqdm import tqdm
 
# índice YOLO = category - 1
NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]
 
IMG_EXT = {".jpg", ".jpeg", ".png"}
EXPECTED_IMAGES = {"train": 6471, "val": 548, "test": 1610}
 
 
def find_subdir(root: Path, name: str) -> Path:
    """O zip às vezes extrai com uma pasta a mais no meio."""
    direct = root / name
    if direct.is_dir():
        return direct
    hits = [p for p in root.rglob(name) if p.is_dir()]
    if not hits:
        raise FileNotFoundError(f"Não achei '{name}/' dentro de {root}")
    return hits[0]
 
 
def convert_one(ann: Path, w: int, h: int) -> list[str]:
    out = []
    for line_no, line in enumerate(ann.read_text().splitlines(), start=1):
        line = line.strip().rstrip(",")
        if not line:
            continue
        parts = line.split(",")
        if len(parts) != 8:
            raise ValueError(f"{ann}:{line_no}: esperado formato VisDrone de 8 colunas")
        try:
            x, y, bw, bh = (float(parts[i]) for i in range(4))
            score = int(float(parts[4]))
            cat = int(float(parts[5]))
        except ValueError as exc:
            raise ValueError(f"{ann}:{line_no}: valor numérico inválido") from exc
 
        if score == 0 or cat in (0, 11):
            continue
        if bw <= 0 or bh <= 0:
            continue
 
        # clipa na moldura antes de normalizar; anotação estourada existe
        x0, y0 = max(x, 0.0), max(y, 0.0)
        x1, y1 = min(x + bw, float(w)), min(y + bh, float(h))
        if x1 <= x0 or y1 <= y0:
            continue
 
        cx = ((x0 + x1) / 2) / w
        cy = ((y0 + y1) / 2) / h
        nw = (x1 - x0) / w
        nh = (y1 - y0) / h
        out.append(f"{cat - 1} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return out


def update_dataset_yaml(yaml_path: Path, dst: Path, split: str) -> dict:
    """Update one split without ever changing another existing split."""
    cfg = (yaml.safe_load(yaml_path.read_text()) or {}) if yaml_path.exists() else {}
    cfg.update({"path": str(dst.resolve()), "nc": len(NAMES), "names": NAMES})
    cfg[split] = f"images/{split}"
    cfg.setdefault("train", f"images/{split}")
    cfg.setdefault("val", f"images/{split}")
    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    return cfg
 
 
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True,
                    help="pasta extraída, ex: data/raw/VisDrone2019-DET-val")
    ap.add_argument("--dst", type=Path, default=Path("data/datasets/VisDrone"))
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--allow-nonstandard-count", action="store_true",
                    help="somente depuração; não use no experimento científico")
    args = ap.parse_args()
 
    src_img = find_subdir(args.src, "images")
    src_ann = find_subdir(args.src, "annotations")
 
    dst_img = args.dst / "images" / args.split
    dst_lbl = args.dst / "labels" / args.split
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)
 
    images = sorted(p for p in src_img.iterdir() if p.suffix.lower() in IMG_EXT)
    if not images:
        print(f"Nenhuma imagem em {src_img}", file=sys.stderr)
        return 1
    if not args.allow_nonstandard_count and len(images) != EXPECTED_IMAGES[args.split]:
        raise ValueError(
            f"Split {args.split} possui {len(images)} imagens; esperado "
            f"{EXPECTED_IMAGES[args.split]}. Download/extracão pode estar incompleto."
        )
 
    n_boxes = n_empty = n_missing = 0
 
    for img in tqdm(images, desc=f"convertendo {args.split}", ncols=80):
        im = cv2.imread(str(img), cv2.IMREAD_COLOR)
        if im is None:
            raise ValueError(f"Imagem ilegível: {img}")
        h, w = im.shape[:2]
 
        ann = src_ann / f"{img.stem}.txt"
        if not ann.exists():
            raise FileNotFoundError(f"Anotação ausente: {ann}")
        lines = convert_one(ann, w, h)
        if not lines:
            n_empty += 1
        n_boxes += len(lines)
 
        (dst_lbl / f"{img.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
 
        target = dst_img / img.name
        if not target.exists():
            shutil.copy2(img, target)
 
    # Preserve the other splits.  The previous implementation rebuilt ``val``
    # from the split currently being converted, so converting test-dev could
    # silently turn the test set into validation data.
    yaml_path = args.dst.parent / "VisDrone.yaml"
    # Keep the original eight-column annotations for the official VisDrone
    # evaluator and for stratification by truncation/occlusion.  YOLO labels
    # alone cannot reproduce the benchmark treatment of ignored regions.
    raw_ann_dir = args.dst / "annotations" / args.split
    raw_ann_dir.mkdir(parents=True, exist_ok=True)
    for img in images:
        ann = src_ann / f"{img.stem}.txt"
        if ann.exists():
            shutil.copy2(ann, raw_ann_dir / ann.name)

    expected_stems = {p.stem for p in images}
    for directory, extensions in ((dst_img, IMG_EXT), (dst_lbl, {".txt"}),
                                  (raw_ann_dir, {".txt"})):
        actual = {p.stem for p in directory.iterdir() if p.suffix.lower() in extensions}
        if actual != expected_stems:
            raise ValueError(
                f"Conteúdo residual/incompleto em {directory}: "
                f"faltam {len(expected_stems - actual)}, sobram {len(actual - expected_stems)}"
            )

    update_dataset_yaml(yaml_path, args.dst, args.split)
 
    print(f"\n{len(images)} imagens | {n_boxes} caixas | "
          f"{n_empty} sem objeto | {n_missing} sem anotação")
    print(f"imagens -> {dst_img}")
    print(f"labels  -> {dst_lbl}")
    print(f"anotacoes VisDrone originais -> {raw_ann_dir}")
    print(f"yaml    -> {yaml_path}")
    return 0
 
 
if __name__ == "__main__":
    sys.exit(main())
