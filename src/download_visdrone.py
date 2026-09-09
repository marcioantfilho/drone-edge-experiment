from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

# Official VisDrone-DET test-dev archive; its ground truth is public.
TEST_DEV_FILE_ID = "1PFdW_VFSCfZ_sTSZAGjQdifF_Xd5mf0V"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["test-dev"], default="test-dev")
    parser.add_argument("--out", type=Path, default=Path("/data/raw"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    archive = args.out / "VisDrone2019-DET-test-dev.zip"
    extracted = args.out / "VisDrone2019-DET-test-dev"
    if not extracted.exists() or args.force:
        if args.force and archive.exists():
            archive.unlink()
        if not archive.exists():
            try:
                import gdown
            except ImportError as exc:
                raise RuntimeError("instale gdown ou use a imagem Docker do projeto") from exc
            print("Baixando VisDrone-DET test-dev da fonte oficial...")
            gdown.download(id=TEST_DEV_FILE_ID, output=str(archive), quiet=False)
        print(f"Extraindo {archive}...")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(args.out)

    # Some archives contain one extra directory level; normalize the expected name.
    candidates = list(args.out.glob("**/VisDrone2019-DET-test-dev"))
    if candidates and candidates[0] != extracted and not extracted.exists():
        shutil.move(str(candidates[0]), str(extracted))
    if not (extracted / "images").is_dir() or not (extracted / "annotations").is_dir():
        raise FileNotFoundError(f"Arquivo extraído sem images/ e annotations/: {extracted}")
    print(f"Dataset disponível em {extracted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())