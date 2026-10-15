from __future__ import annotations

import argparse
import hashlib
import random
from collections import defaultdict
from pathlib import Path

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def train_images(base_yaml: Path) -> list[Path]:
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(str(base_yaml))
    root = Path(data["train"])
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def stratified_sample(images: list[Path], n: int, seed: int) -> list[Path]:
    by_seq: dict[str, list[Path]] = defaultdict(list)
    for p in images:
        by_seq[p.name.split("_")[0]].append(p)
    rng = random.Random(seed)
    pools = [rng.sample(v, len(v)) for _, v in sorted(by_seq.items())]
    rng.shuffle(pools)
    chosen: list[Path] = []
    while len(chosen) < n and any(pools):
        for pool in pools:
            if pool and len(chosen) < n:
                chosen.append(pool.pop())
    return sorted(chosen)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-yaml", type=Path, default=Path("/data/datasets/VisDrone.yaml"))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("results/quant"))
    args = ap.parse_args()

    images = train_images(args.base_yaml)
    chosen = stratified_sample(images, args.n, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    listing = args.out_dir / "calib_train.txt"
    listing.write_text("".join(f"{p}\n" for p in chosen))
    names = yaml.safe_load(args.base_yaml.read_text())["names"]
    (args.out_dir / "calib_train.yaml").write_text(yaml.safe_dump({
        "train": str(listing.resolve()), "val": str(listing.resolve()), "names": names,
    }, sort_keys=False, allow_unicode=True))
    digest = hashlib.sha256(listing.read_bytes()).hexdigest()
    (args.out_dir / "calib_train.sha256").write_text(f"{digest}  calib_train.txt\n")
    n_seq = len({p.name.split('_')[0] for p in chosen})
    print(f"{len(chosen)} imagens de train de {n_seq} sequências -> {listing} (sha256 {digest[:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
