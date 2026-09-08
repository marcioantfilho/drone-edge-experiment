#!/usr/bin/env python3
"""
visdrone_to_coco_classes.py — remapeia os rótulos VisDrone para índices COCO.

Para que serve
--------------
Um checkpoint COCO pré-treinado (yolo11n.pt) prevê os 80 índices do COCO.
Os rótulos VisDrone usam outra numeração. Avaliar um contra o outro sem
remapear pune o modelo duas vezes por acerto: falso positivo na classe que
ele emitiu e falso negativo na classe esperada.

Com o remapeamento você obtém um BASELINE HONESTO sem treinar nada — o
suficiente para a curva de resolução ter forma legível e o pipeline ser
validado com números que significam alguma coisa.

Mapeamento
----------
    VisDrone            -> COCO
    0 pedestrian        -> 0  person
    1 people            -> 0  person
    2 bicycle           -> 1  bicycle
    3 car               -> 2  car
    4 van               -> 2  car        (COCO não tem 'van')
    5 truck             -> 7  truck
    6 tricycle          -> descartado    (sem equivalente)
    7 awning-tricycle   -> descartado    (sem equivalente)
    8 bus               -> 5  bus
    9 motor             -> 3  motorcycle

Limitações a declarar no texto
------------------------------
  * 'van' vira 'car': o COCO não distingue. Infla a classe car.
  * tricycle/awning-tricycle somem: ~7% das instâncias do VisDrone.
    O recall reportado é sobre as classes REMANESCENTES, não sobre o
    VisDrone inteiro. Não compare este número com papers.
  * COCO foi treinado em fotos ao nível do olho. O recall ainda será baixo
    para objetos minúsculos — mas por gap de domínio, que é real, e não
    por índice trocado, que é artefato.

Uso:
    python scripts/visdrone_to_coco_classes.py \
        --src /data/datasets/VisDrone --split val \
        --dst /data/datasets/VisDroneCOCO
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

# VisDrone -> COCO ; None = descartar
VISDRONE_TO_COCO: dict[int, int | None] = {
    0: 0,    # pedestrian      -> person
    1: 0,    # people          -> person
    2: 1,    # bicycle         -> bicycle
    3: 2,    # car             -> car
    4: 2,    # van             -> car
    5: 7,    # truck           -> truck
    6: None,  # tricycle
    7: None,  # awning-tricycle
    8: 5,    # bus             -> bus
    9: 3,    # motor           -> motorcycle
}

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

IMG_EXT = {".jpg", ".jpeg", ".png"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("/data/datasets/VisDrone"))
    ap.add_argument("--dst", type=Path, default=Path("/data/datasets/VisDroneCOCO"))
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    src_img = args.src / "images" / args.split
    src_lbl = args.src / "labels" / args.split
    for d in (src_img, src_lbl):
        if not d.is_dir():
            print(f"Não encontrei {d}", file=sys.stderr)
            return 1

    dst_img = args.dst / "images" / args.split
    dst_lbl = args.dst / "labels" / args.split
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    kept = dropped = 0
    per_class: dict[int, int] = {}

    images = sorted(p for p in src_img.iterdir() if p.suffix.lower() in IMG_EXT)
    for img in tqdm(images, desc="remapeando", ncols=80):
        # imagem por hardlink: mesmo inode, 0 byte extra
        target = dst_img / img.name
        if not target.exists():
            try:
                os.link(img, target)
            except OSError:
                shutil.copy2(img, target)

        out = []
        lbl = src_lbl / f"{img.stem}.txt"
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                new = VISDRONE_TO_COCO.get(int(parts[0]))
                if new is None:
                    dropped += 1
                    continue
                kept += 1
                per_class[new] = per_class.get(new, 0) + 1
                out.append(f"{new} {' '.join(parts[1:])}")

        (dst_lbl / f"{img.stem}.txt").write_text(
            "\n".join(out) + ("\n" if out else "")
        )

    yaml_path = args.dst.parent / "VisDroneCOCO.yaml"
    yaml_path.write_text(yaml.safe_dump({
        "path": str(args.dst.resolve()),
        "train": f"images/{args.split}",
        "val": f"images/{args.split}",
        "nc": 80,
        "names": COCO_NAMES,
    }, sort_keys=False, allow_unicode=True))

    total = kept + dropped
    print(f"\n{len(images)} imagens")
    print(f"{kept} caixas mantidas | {dropped} descartadas "
          f"({100*dropped/max(total,1):.1f}% — tricycle/awning-tricycle)")
    print("\nDistribuição após o remapeamento:")
    for c, n in sorted(per_class.items(), key=lambda kv: -kv[1]):
        print(f"  {COCO_NAMES[c]:<12s} {n:6d}")
    print(f"\nyaml -> {yaml_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
    