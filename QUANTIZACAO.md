# Pré-quantização e execução em Jetson Orin Nano / RK3588

Ponto operacional fixo: conf=0,25, IoU=0,50. Nada aqui toca o test-dev.

## 0. Antes de tudo

Registre no `METODOLOGIA.md`, antes de ver números quantizados:

- candidatas: YOLO11s (treino 640), imgsz ∈ {640, 960, 1280}, captura ∈ {native, 720p};
- critério de aceite: queda de `recall_op` macro em relação ao FP32 de no máximo 2 pontos
  (pareado por seed) e nenhuma classe caindo mais que 5 pontos;
- calibração INT8: `results/quant/calib_train.txt` (300 imagens de train, seed 0, hash em
  `calib_train.sha256`).

## 1. Servidor (RTX 5070)

```bash
git pull
make build-gpu && make check
DATA_DIR=/mnt/storage/sp-edge-data bash scripts/run_server_experiment.sh prepare     # testes + ambiente
DATA_DIR=/mnt/storage/sp-edge-data bash scripts/run_server_experiment.sh pre-quant   # ~1h30
```

`pre-quant` faz, em ordem:

1. move as saídas anteriores para `results/arquivo_<RUN_ID>/` (nada é apagado);
2. grade completa em conf=0,25 sem curvas PR → `grid.csv`, `grid_per_class.csv`
   (recall/precisão por classe em 0,25), `summary/`, `figs/`, `frozen_configs.yaml`;
3. predições conf=0,001 do YOLO11s, 4 seeds × {native, 720p} × {640, 960, 1280} →
   `results/predictions/*.npz` e `index.csv`;
4. verifica que o recall_op da passada de despejo reproduz o `grid.csv` (|Δ| < 0,002);
   se falhar, o script para;
5. lista de calibração de train e ONNX estático (opset 17) por seed e imgsz →
   `results/quant/onnx/`, com `SHA256SUMS`.

### Conversão RKNN (no servidor, fora do Docker GPU)

O RKNN Toolkit2 conflita com o torch da imagem; use um venv x86 separado:

```bash
python3 -m venv ~/venv-rknn && source ~/venv-rknn/bin/activate
pip install ultralytics==8.4.140 rknn-toolkit2
sed "s#^/data/#${DATA_DIR}/#" results/quant/calib_train.txt > results/quant/calib_train_host.txt
for s in 0 1 2 67; do for z in 640 960; do
  d=results/quant/rknn/yolo11s_seed${s}_${z}; mkdir -p $d && cp results/quant/onnx/yolo11s_seed${s}/best.pt $d/
  yolo export model=$d/best.pt format=rknn name=rk3588 imgsz=$z batch=1        # FP16 -> $d/best_rknn_model/
  python src/convert_rknn_int8.py --fp16-dir $d/best_rknn_model --onnx $d/best.onnx \
      --calib results/quant/calib_train_host.txt --out-dir $d/best_int8_rknn_model
done; done
```

Confira o nome do ONNX intermediário que o export deixa em `$d`; se for diferente de
`best.onnx`, ajuste `--onnx`. 1280 fica fora do RK3588 na primeira rodada (≈88 GFLOPs);
inclua só se 960 couber no orçamento de latência.

## 2. O que copiar para cada dispositivo

- `src/`, `configs/experiment.yaml`;
- val degradado: `derived/visdrone_val_native_q85*` e `derived/visdrone_val_720p_q85*`
  (imagens, labels e YAML; ajuste `path:` no YAML);
- Jetson: `results/quant/onnx/yolo11s_seed*/best.pt`, `results/quant/calib_train.*` e as
  300 imagens listadas (com labels);
- RK3588: `results/quant/rknn/`.

Instale `ultralytics==8.4.140` em ambos (no Jetson, sobre o torch do JetPack; no RK3588,
`pip install ultralytics==8.4.140 rknn-toolkit-lite2`). Registre versões de JetPack/TensorRT
ou do driver NPU/RKNN em cada rodada.

## 3. Jetson Orin Nano (TensorRT FP16 e INT8)

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks            # registre o modo de energia
D=/caminho/derived
sed -i "s#^/data/#/caminho/#" calib_train.txt calib_train.yaml
for s in 0 1 2 67; do for z in 640 960 1280; do
  w=yolo11s_seed${s}/best.pt
  yolo export model=$w format=engine imgsz=$z batch=1 half=True device=0 && mv yolo11s_seed${s}/best.engine yolo11s_seed${s}/fp16_${z}.engine
  yolo export model=$w format=engine imgsz=$z batch=1 int8=True data=calib_train.yaml device=0 && mv yolo11s_seed${s}/best.engine yolo11s_seed${s}/int8_${z}.engine
  for p in fp16 int8; do
    python src/save_predictions.py --weights yolo11s_seed${s}/${p}_${z}.engine --label yolo11s-trt-${p} \
      --seeds $s --tags native 720p --sizes $z --derived-root $D --device 0 --out-dir results/predictions_jetson
    /usr/src/tensorrt/bin/trtexec --loadEngine=yolo11s_seed${s}/${p}_${z}.engine --warmUp=2000 --iterations=200 \
      > results/predictions_jetson/trtexec_${p}_seed${s}_${z}.txt
  done
done; done
```

## 4. RK3588 (Orange Pi)

```bash
D=/caminho/derived
for s in 0 1 2 67; do for z in 640 960; do for p in best_rknn_model best_int8_rknn_model; do
  python src/save_predictions.py --weights rknn/yolo11s_seed${s}_${z}/$p --label yolo11s-rknn-${p%%_rknn_model} \
    --seeds $s --tags native 720p --sizes $z --derived-root $D --device cpu --out-dir results/predictions_rk3588
done; done; done
```

## 5. Volta ao servidor

Copie `results/predictions_jetson/` e `results/predictions_rk3588/` de volta. A comparação
FP32 × FP16 × INT8 usa o `recall_op_inrun` do `index.csv` de cada dispositivo contra o FP32
do servidor, pareado por seed, com o critério do item 0. Os `.npz` permitem a mesma análise
por classe e tamanho de objeto.
