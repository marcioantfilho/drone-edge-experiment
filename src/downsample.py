#!/usr/bin/env python3
"""
downsample.py — gera as versões degradadas do split de avaliação.

Simula o que o DRONE realmente captura e transmite: a imagem é reduzida
com filtro de área (equivalente ao decimador de um sensor/encoder) e,
opcionalmente, comprimida em JPEG. Nada é reamostrado de volta para cima:
a perda de informação tem que chegar no modelo como perda de informação.

Ponto importante e que economiza muito trabalho:
    os rótulos YOLO são NORMALIZADOS (cx, cy, w, h em [0,1]).
    Reduzir a imagem NÃO muda o rótulo. Os .txt são reaproveitados por
    hardlink — zero reprocessamento, zero risco de erro de conversão.

Saída, por resolução:
    <derived_root>/<dataset>_<split>_<tag>_q<qualidade>/
        images/*.jpg
        labels/*.txt          (hardlink para os originais)
    e um manifest.csv com os bytes de cada imagem, que alimenta o
    modelo de latência de transmissão em latency_model.py.

Uso:
    python src/downsample.py --config configs/experiment.yaml
    python src/downsample.py --config configs/experiment.yaml --smoke 20
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

INTERP = {
    "area": cv2.INTER_AREA,        # correto para reduzir
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "nearest": cv2.INTER_NEAREST,  # só para ablação "pior caso"
}


@dataclass(frozen=True)
class Variant:
    tag: str
    height: int | None
    quality: int | None
    interp: str

    @property
    def dirname_suffix(self) -> str:
        q = "lossless" if self.quality is None else f"q{self.quality}"
        return f"{self.tag}_{q}"


# --------------------------------------------------------------------------
# Descoberta do split a partir do data.yaml do Ultralytics
# --------------------------------------------------------------------------
def resolve_split_dir(base_yaml: Path, split: str) -> tuple[Path, Path]:
    """Retorna (images_dir, labels_dir) do split pedido."""
    with open(base_yaml) as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg.get("path", base_yaml.parent))
    if not root.is_absolute():
        root = (base_yaml.parent / root).resolve()

    rel = cfg.get(split)
    if rel is None:
        raise KeyError(f"split '{split}' não existe em {base_yaml}. Chaves: {list(cfg)}")
    if isinstance(rel, list):
        rel = rel[0]

    images_dir = (root / rel).resolve()
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Diretório de imagens não encontrado: {images_dir}")

    # convenção Ultralytics: .../images/... -> .../labels/...
    labels_dir = Path(str(images_dir).replace("/images", "/labels", 1))
    if not labels_dir.is_dir():
        raise FileNotFoundError(
            f"Diretório de labels não encontrado: {labels_dir}\n"
            "Confira se o dataset já foi convertido para o formato YOLO."
        )
    return images_dir, labels_dir


def list_images(images_dir: Path, limit: int | None = None) -> list[Path]:
    files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    if not files:
        raise FileNotFoundError(f"Nenhuma imagem em {images_dir}")
    return files[:limit] if limit else files


# --------------------------------------------------------------------------
# Processamento de uma imagem
# --------------------------------------------------------------------------
def process_one(args) -> tuple[str, int, int, int, int] | None:
    """Retorna (stem, w_out, h_out, bytes_out, n_labels) ou None em caso de falha."""
    src_img, src_lbl, out_img_dir, out_lbl_dir, variant = args

    img = cv2.imread(str(src_img), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h0, w0 = img.shape[:2]

    if variant.height is None or variant.height >= h0:
        # 'native', ou alvo maior que o original: não faz upsample.
        # Fazer upsample inventaria informação e contaminaria a curva de recall.
        out = img
    else:
        scale = variant.height / h0
        w1 = max(32, int(round(w0 * scale)))
        h1 = variant.height
        out = cv2.resize(img, (w1, h1), interpolation=INTERP[variant.interp])

    if variant.quality is None:
        dst_img = out_img_dir / f"{src_img.stem}.png"
        cv2.imwrite(str(dst_img), out, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    else:
        dst_img = out_img_dir / f"{src_img.stem}.jpg"
        cv2.imwrite(str(dst_img), out, [cv2.IMWRITE_JPEG_QUALITY, variant.quality])

    # rótulo: hardlink (mesmo inode, ocupa 0 byte extra)
    dst_lbl = out_lbl_dir / f"{src_img.stem}.txt"
    n_labels = 0
    if src_lbl.exists():
        if not dst_lbl.exists():
            try:
                os.link(src_lbl, dst_lbl)
            except OSError:
                shutil.copyfile(src_lbl, dst_lbl)
        with open(src_lbl) as f:
            n_labels = sum(1 for line in f if line.strip())
    else:
        dst_lbl.touch()  # imagem de fundo, sem objetos — é válido no YOLO

    hh, ww = out.shape[:2]
    return (src_img.stem, ww, hh, dst_img.stat().st_size, n_labels)


# --------------------------------------------------------------------------
def build_variant(
    images: list[Path],
    labels_dir: Path,
    out_root: Path,
    variant: Variant,
    workers: int,
) -> Path:
    out_dir = out_root
    img_dir = out_dir / "images"
    lbl_dir = out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        (p, labels_dir / f"{p.stem}.txt", img_dir, lbl_dir, variant)
        for p in images
    ]

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(process_one, j) for j in jobs]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=variant.dirname_suffix, ncols=80):
            r = fut.result()
            if r:
                rows.append(r)

    manifest = out_dir / "manifest.csv"
    with open(manifest, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stem", "width", "height", "bytes", "n_labels"])
        w.writerows(sorted(rows))

    total_mb = sum(r[3] for r in rows) / 1e6
    print(f"  -> {len(rows)} imagens | {total_mb:.1f} MB | média {total_mb*1000/max(len(rows),1):.1f} kB/img")
    return out_dir


def write_data_yaml(base_yaml: Path, out_dir: Path, dst_yaml: Path) -> None:
    """data.yaml apontando train E val para o split degradado.

    O Ultralytics exige a chave 'train'; ela não é usada em model.val(),
    mas se faltar, o carregador quebra.
    """
    with open(base_yaml) as f:
        base = yaml.safe_load(f)

    cfg = {
        "path": str(out_dir.resolve()),
        "train": "images",
        "val": "images",
        "names": base["names"],
    }
    if "nc" in base:
        cfg["nc"] = base["nc"]

    with open(dst_yaml, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    ap.add_argument("--smoke", type=int, default=None,
                    help="processa só N imagens (teste rápido na máquina local)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--force", action="store_true", help="regera mesmo se já existir")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    ds = cfg["dataset"]
    tx = cfg["transmission"]

    base_yaml = Path(ds["base_yaml"])
    split = ds["split"]
    derived_root = Path(ds["derived_root"])
    derived_root.mkdir(parents=True, exist_ok=True)

    images_dir, labels_dir = resolve_split_dir(base_yaml, split)
    images = list_images(images_dir, args.smoke)
    print(f"Base: {images_dir}  ({len(images)} imagens)\n")

    quality = tx.get("jpeg_quality")
    interp = tx.get("interpolation", "area")

    produced = []
    for spec in cfg["capture_resolutions"]:
        v = Variant(tag=spec["tag"], height=spec.get("height"), quality=quality, interp=interp)
        out_dir = derived_root / f"{ds['name']}_{split}_{v.dirname_suffix}"
        yaml_path = derived_root / f"{ds['name']}_{split}_{v.dirname_suffix}.yaml"

        if out_dir.exists() and not args.force:
            n = len(list((out_dir / "images").glob("*"))) if (out_dir / "images").exists() else 0
            if n == len(images):
                print(f"[skip] {out_dir.name} já completo ({n} imagens)")
                write_data_yaml(base_yaml, out_dir, yaml_path)
                produced.append((v.tag, yaml_path))
                continue

        print(f"[gerar] {out_dir.name}")
        build_variant(images, labels_dir, out_dir, v, args.workers)
        write_data_yaml(base_yaml, out_dir, yaml_path)
        produced.append((v.tag, yaml_path))

    print("\nSplits prontos:")
    for tag, y in produced:
        print(f"  {tag:>8s}  {y}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
