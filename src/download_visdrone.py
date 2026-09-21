from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

# Official VisDrone-DET Google Drive archives linked by VisDrone/VisDrone-Dataset.
DATASETS = {
    "train": {
        "file_id": "1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn",
        "dirname": "VisDrone2019-DET-train",
        "expected_images": 6471,
    },
    "val": {
        "file_id": "1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59",
        "dirname": "VisDrone2019-DET-val",
        "expected_images": 548,
    },
    "test-dev": {
        "file_id": "1PFdW_VFSCfZ_sTSZAGjQdifF_Xd5mf0V",
        "dirname": "VisDrone2019-DET-test-dev",
        "expected_images": 1610,
    },
}


def safe_extract(zf: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in zf.infolist():
        target = (root / member.filename).resolve()
        if root != target and root not in target.parents:
            raise ValueError(f"Entrada ZIP fora do destino: {member.filename}")
    zf.extractall(root)


def download_one(kind: str, out: Path, force: bool) -> None:
    spec = DATASETS[kind]
    dirname = str(spec["dirname"])
    expected = int(spec["expected_images"])
    archive = out / f"{dirname}.zip"
    extracted = out / dirname

    if force:
        if archive.exists():
            archive.unlink()
        if extracted.exists():
            if extracted.resolve().parent != out.resolve():
                raise RuntimeError(f"Recusa remover diretório fora de {out}: {extracted}")
            shutil.rmtree(extracted)

    if not extracted.exists():
        if not archive.exists():
            try:
                import gdown
            except ImportError as exc:
                raise RuntimeError("instale gdown ou use a imagem Docker do projeto") from exc
            print(f"Baixando VisDrone-DET {kind} da fonte oficial...")
            downloaded = gdown.download(
                id=str(spec["file_id"]), output=str(archive), quiet=False
            )
            if not downloaded or not archive.is_file():
                raise RuntimeError(f"Download não produziu o arquivo esperado: {archive}")
        print(f"Extraindo {archive}...")
        with zipfile.ZipFile(archive) as zf:
            bad = zf.testzip()
            if bad:
                raise zipfile.BadZipFile(f"CRC inválido em {bad}")
            safe_extract(zf, out)

    # Some archives contain one extra directory level; normalize the expected name.
    candidates = [p for p in out.glob(f"**/{dirname}") if p != extracted]
    if candidates and not extracted.exists():
        shutil.move(str(candidates[0]), str(extracted))
    if not (extracted / "images").is_dir() or not (extracted / "annotations").is_dir():
        raise FileNotFoundError(f"Arquivo extraído sem images/ e annotations/: {extracted}")
    n_images = len([p for p in (extracted / "images").iterdir() if p.is_file()])
    n_annotations = len(list((extracted / "annotations").glob("*.txt")))
    if (n_images, n_annotations) != (expected, expected):
        raise ValueError(
            f"{kind} incompleto: {n_images} imagens e {n_annotations} anotações; "
            f"esperado {expected} de cada"
        )
    print(f"Dataset {kind} disponível em {extracted}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=[*DATASETS, "train-val", "all"], default="test-dev")
    parser.add_argument("--out", type=Path, default=Path("/data/raw"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    kinds = {
        "train-val": ["train", "val"],
        "all": ["train", "val", "test-dev"],
    }.get(args.kind, [args.kind])
    for kind in kinds:
        download_one(kind, args.out, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
