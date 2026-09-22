from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

EXPECTED_COUNTS = {"train": 6471, "val": 548, "test": 1610}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_dataset(yaml_path: Path) -> tuple[dict, Path]:
    cfg = yaml.safe_load(yaml_path.read_text()) or {}
    root = Path(cfg.get("path", yaml_path.parent))
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    return cfg, root


def find_content_duplicates(hashes_by_split: dict[str, dict[str, str]]) -> tuple[dict[str, list[tuple[str, str]]], dict[str, list[tuple[str, str]]]]:
    """Return exact-content duplicates within and across dataset splits."""
    occurrences: dict[str, list[tuple[str, str]]] = {}
    for split, image_hashes in hashes_by_split.items():
        for name, digest in image_hashes.items():
            occurrences.setdefault(digest, []).append((split, name))

    within_split: dict[str, list[tuple[str, str]]] = {}
    across_splits: dict[str, list[tuple[str, str]]] = {}
    for digest, locations in occurrences.items():
        locations.sort()
        if len({split for split, _ in locations}) > 1:
            across_splits[digest] = locations
        elif len(locations) > 1:
            within_split[digest] = locations
    return within_split, across_splits


def duplicate_details(duplicates: dict[str, list[tuple[str, str]]]) -> list[list[str]]:
    """Produce stable, JSON-friendly duplicate locations."""
    return [[f"{split}/{name}" for split, name in locations] for locations in duplicates.values()]


def audit_split(cfg: dict, root: Path, split: str) -> dict:
    images_dir = (root / cfg[split]).resolve()
    if images_dir.parent.name != "images":
        raise ValueError(f"{split}: caminho nao segue .../images/<split>: {images_dir}")
    labels_dir = images_dir.parent.parent / "labels" / images_dir.name
    annotations_dir = root / "annotations" / split
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    expected = EXPECTED_COUNTS[split]
    if len(images) != expected:
        raise ValueError(f"{split}: {len(images)} imagens; esperado {expected}")

    content_hashes = {}
    labels_digest = hashlib.sha256()
    n_boxes = 0
    nc = int(cfg.get("nc", len(cfg["names"])))
    for image in images:
        label = labels_dir / f"{image.stem}.txt"
        original = annotations_dir / f"{image.stem}.txt"
        if not label.is_file() or not original.is_file():
            raise FileNotFoundError(f"{split}/{image.stem}: label YOLO ou anotacao original ausente")
        content_hashes[image.name] = sha256(image)
        raw_label = label.read_bytes()
        labels_digest.update(image.stem.encode())
        labels_digest.update(raw_label)
        for line_no, line in enumerate(raw_label.decode().splitlines(), 1):
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"{label}:{line_no}: label YOLO nao possui 5 colunas")
            cls = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            if not 0 <= cls < nc or not all(0.0 <= x <= 1.0 for x in coords):
                raise ValueError(f"{label}:{line_no}: classe/coordenadas fora do dominio")
            if coords[2] <= 0 or coords[3] <= 0:
                raise ValueError(f"{label}:{line_no}: caixa sem area")
            n_boxes += 1
        for line_no, line in enumerate(original.read_text().splitlines(), 1):
            if line.strip() and len(line.strip().rstrip(",").split(",")) != 8:
                raise ValueError(f"{original}:{line_no}: anotacao VisDrone invalida")

    return {"images": len(images), "boxes": n_boxes, "labels_sha256": labels_digest.hexdigest(), "image_hashes": content_hashes}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", type=Path, default=Path("/data/datasets/VisDrone.yaml"))
    parser.add_argument("--splits", nargs="+", choices=list(EXPECTED_COUNTS), default=["train", "val"])
    parser.add_argument("--out", type=Path, default=Path("results/dataset_audit.json"))
    parser.add_argument("--fail-on-intra-split-duplicates", action="store_true", help="treat exact duplicates within a split as an error after reporting them")
    args = parser.parse_args()
    cfg, root = resolve_dataset(args.yaml)
    missing = [split for split in args.splits if split not in cfg]
    if missing:
        raise KeyError(f"splits ausentes no YAML: {missing}")

    report = {"dataset_yaml": str(args.yaml), "root": str(root), "splits": {}}
    hashes_by_split: dict[str, dict[str, str]] = {}
    for split in args.splits:
        result = audit_split(cfg, root, split)
        hashes_by_split[split] = result.pop("image_hashes")
        report["splits"][split] = result

    within_split, across_splits = find_content_duplicates(hashes_by_split)
    if across_splits:
        locations = next(iter(across_splits.values()))
        raise ValueError("vazamento entre splits: " + " duplica ".join(f"{split}/{name}" for split, name in locations) + " por conteudo")

    intra_split_details = duplicate_details(within_split)
    report["unique_image_hashes"] = sum(len(hashes) for hashes in hashes_by_split.values()) - sum(len(locations) - 1 for locations in within_split.values())
    report["intra_split_duplicate_groups"] = intra_split_details
    report["intra_split_duplicate_images"] = sum(len(group) for group in intra_split_details)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if intra_split_details:
        message = f"[AVISO] {len(intra_split_details)} grupo(s) de imagens identicas no mesmo split registrado(s) em {args.out}; isto nao e vazamento entre splits."
        if args.fail_on_intra_split_duplicates:
            raise ValueError(message)
        print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
