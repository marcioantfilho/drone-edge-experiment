from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fp16-dir", type=Path, required=True, help="diretório *_rknn_model do export FP16")
    ap.add_argument("--onnx", type=Path, required=True, help="ONNX intermediário do mesmo export")
    ap.add_argument("--calib", type=Path, required=True, help="calib_train.txt com caminhos locais")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--target", default="rk3588")
    args = ap.parse_args()

    from rknn.api import RKNN

    rknn_files = list(args.fp16_dir.glob("*.rknn"))
    if len(rknn_files) != 1:
        raise FileNotFoundError(f"esperado um .rknn em {args.fp16_dir}, achados {len(rknn_files)}")
    if args.out_dir.exists():
        raise FileExistsError(f"{args.out_dir} já existe; não sobrescrevo")
    shutil.copytree(args.fp16_dir, args.out_dir)
    target = args.out_dir / rknn_files[0].name

    rknn = RKNN(verbose=False)
    try:
        if rknn.config(target_platform=args.target, mean_values=[[0, 0, 0]],
                       std_values=[[255, 255, 255]]) != 0:
            raise RuntimeError("rknn.config falhou")
        if rknn.load_onnx(model=str(args.onnx)) != 0:
            raise RuntimeError("rknn.load_onnx falhou")
        if rknn.build(do_quantization=True, dataset=str(args.calib)) != 0:
            raise RuntimeError("rknn.build INT8 falhou")
        if rknn.export_rknn(str(target)) != 0:
            raise RuntimeError("rknn.export_rknn falhou")
    finally:
        rknn.release()
    print(f"RKNN INT8 em {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
