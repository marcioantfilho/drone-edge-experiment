#!/usr/bin/env bash
set -Eeuo pipefail

# Executa o experimento no servidor sem tocar no test-dev.
# Uso:
#   DATA_DIR=/mnt/storage/sp-edge-data bash scripts/run_server_experiment.sh all
#   DATA_DIR=/mnt/storage/sp-edge-data bash scripts/run_server_experiment.sh prepare
#   DATA_DIR=/mnt/storage/sp-edge-data bash scripts/run_server_experiment.sh train
#   DATA_DIR=/mnt/storage/sp-edge-data bash scripts/run_server_experiment.sh evaluate
#   DATA_DIR=/mnt/storage/sp-edge-data bash scripts/run_server_experiment.sh pre-quant
#   DATA_DIR=/mnt/storage/sp-edge-data bash scripts/run_server_experiment.sh models-smoke
#   DATA_DIR=/mnt/storage/sp-edge-data bash scripts/run_server_experiment.sh models

PHASE="${1:-all}"
export DATA_DIR="${DATA_DIR:-/mnt/storage/sp-edge-data}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

mkdir -p results/logs "${DATA_DIR}/raw" "${DATA_DIR}/datasets" "${DATA_DIR}/derived"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="results/logs/server_${PHASE}_${RUN_ID}.log"

exec > >(tee -a "${LOG_FILE}") 2>&1

on_error() {
    local exit_code=$?
    echo
    echo "ERRO: fase '${PHASE}' interrompida (código ${exit_code})."
    echo "Log: ${LOG_FILE}"
    exit "${exit_code}"
}
trap on_error ERR

require_command() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "Comando obrigatório não encontrado: $1" >&2
        exit 2
    }
}

record_provenance() {
    echo "=== Proveniência ==="
    date --iso-8601=seconds
    git rev-parse HEAD
    if [[ -n "$(git status --porcelain)" ]]; then
        echo "ERRO: worktree com alterações não commitadas."
        git status --short
        if [[ "${ALLOW_DIRTY:-0}" != "1" ]]; then
            echo "Faça commit para tornar o experimento reproduzível (ou ALLOW_DIRTY=1 apenas para depuração)." >&2
            exit 4
        fi
        echo "ALLOW_DIRTY=1: execução marcada como não reproduzível."
    else
        echo "Worktree limpo."
    fi
    docker --version
    docker compose version
    nvidia-smi
    free -h
    DOCKER_ROOT_DIR="$(docker info --format '{{.DockerRootDir}}')"
    df -h "${DATA_DIR}" "${DOCKER_ROOT_DIR}" .
    echo "DATA_DIR=${DATA_DIR}"
    echo "DOCKER_ROOT_DIR=${DOCKER_ROOT_DIR}"
}

check_dataset() {
    docker compose -f docker/docker-compose.yml --profile gpu run --rm -T gpu \
        python -c "from pathlib import Path; from ultralytics.data.utils import check_det_dataset; d=check_det_dataset('/data/datasets/VisDrone.yaml'); train=Path(d['train']); val=Path(d['val']); print('train:', train); print('val:', val); print('classes:', len(d['names'])); assert len(d['names']) == 10; assert train.resolve() != val.resolve(), 'train e val apontam para o mesmo diretório'; assert len(list(train.glob('*'))) == 6471; assert len(list(val.glob('*'))) == 548"
}

check_checkpoints() {
    local missing=0 model seed checkpoint
    for model in yolo11n yolo11s yolov10n; do
        for seed in 0 1 2 67; do
            checkpoint="runs/train/${model}_visdrone_seed_${seed}/weights/best.pt"
            if [[ ! -f "${checkpoint}" ]]; then
                echo "Checkpoint ausente: ${checkpoint}" >&2
                missing=$((missing + 1))
            fi
        done
    done
    if (( missing > 0 )); then
        echo "Faltam ${missing} dos 12 checkpoints esperados." >&2
        return 1
    fi
    echo "Os 12 checkpoints esperados estão presentes."
}

prepare() {
    echo "=== Preparação da imagem GPU ==="
    make build-gpu
    make check
    echo "=== Testes do código dentro da imagem científica ==="
    docker compose -f docker/docker-compose.yml --profile gpu run --rm -T gpu \
        bash -c "python -m py_compile src/*.py scripts/*.py tests/*.py && python -m unittest discover -s tests"
    echo "=== Pré-download e abertura dos pesos-base ==="
    docker compose -f docker/docker-compose.yml --profile gpu run --rm -T gpu \
        python -c "from ultralytics import YOLO; [YOLO(w) for w in ('yolo11n.pt', 'yolo11s.pt', 'yolov10n.pt')]; print('pesos-base: OK')"
    docker compose -f docker/docker-compose.yml --profile gpu run --rm -T gpu \
        python -c "import platform, torch, ultralytics, cv2, numpy, pandas; print({'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda, 'ultralytics': ultralytics.__version__, 'opencv': cv2.__version__, 'numpy': numpy.__version__, 'pandas': pandas.__version__})"
    docker compose -f docker/docker-compose.yml --profile gpu run --rm -T gpu \
        python -m pip freeze > results/environment.lock.txt
    docker image inspect sp-edge:gpu --format '{{json .RepoDigests}} {{.Id}}' \
        > results/docker_image.txt

    echo "=== Validação do dataset train/val ==="
    check_dataset
    make audit-data
}

train_models() {
    echo "=== Treino: 3 modelos x 4 seeds x 100 épocas ==="
    check_dataset
    make train

    check_checkpoints
}

evaluate_val() {
    if ! check_checkpoints; then
        echo "Execute a fase train antes da avaliação." >&2
        exit 3
    fi

    echo "=== Geração das sete resoluções de captura ==="
    make data

    echo "=== Grade completa no conjunto val ==="
    make sweep

    echo "=== Estatística entre seeds e figuras ==="
    make summary
    make figs

    echo "=== Arquivos para formular/congelar a hipótese ==="
    ls -lh \
        results/grid.csv \
        results/grid_per_class.csv \
        results/grid_com_latencia.csv \
        results/config_otima_por_banda.csv \
        results/frozen_configs.yaml \
        results/summary/summary_by_seed.csv \
        results/summary/model_comparison_by_seed.csv
}

DC_GPU=(docker compose -f docker/docker-compose.yml --profile gpu run --rm -T gpu)
PRED_SEEDS="${PRED_SEEDS:-0 1 2 67}"
QUANT_SIZES="${QUANT_SIZES:-640 960 1280}"

archive_results() {
    # Move (não apaga) as saídas anteriores para results/arquivo_<RUN_ID>/.
    local dest="results/arquivo_${RUN_ID}" moved=0 item
    for item in grid.csv grid_per_class.csv grid_com_latencia.csv config_otima_por_banda.csv \
                frozen_configs.yaml summary figs; do
        if [[ -e "results/${item}" ]]; then
            mkdir -p "${dest}"
            mv "results/${item}" "${dest}/"
            moved=$((moved + 1))
        fi
    done
    echo "Saídas anteriores arquivadas em ${dest} (${moved} itens)."
}

check_derived() {
    local n
    n=$(find "${DATA_DIR}/derived" -maxdepth 1 -name 'visdrone_val_*_q85.yaml' | wc -l)
    if (( n < 7 )); then
        echo "Splits degradados incompletos (${n}/7); gerando."
        make data
    fi
}

verify_predictions() {
    # O recall_op calculado na passada de despejo precisa reproduzir a grade oficial.
    "${DC_GPU[@]}" python -c "
import pandas as pd, sys
g = pd.read_csv('results/grid.csv')
p = pd.read_csv('results/predictions/index.csv')
p = p[p['format'] == '.pt']
m = p.merge(g, on=['model', 'seed', 'capture_tag', 'imgsz'], how='left')
m['diff'] = (m['recall_op_inrun'] - m['recall_op']).abs()
print(m[['model', 'seed', 'capture_tag', 'imgsz', 'recall_op_inrun', 'recall_op', 'diff']].to_string(index=False))
bad = m[(m['diff'] >= 0.002) | m['recall_op'].isna()]
sys.exit(f'{len(bad)} células divergem >= 0.002 da grade' if len(bad) else 0)
"
}

export_onnx() {
    local seed size dir
    for seed in ${PRED_SEEDS}; do
        dir="results/quant/onnx/yolo11s_seed${seed}"
        mkdir -p "${dir}"
        cp -n "runs/train/yolo11s_visdrone_seed_${seed}/weights/best.pt" "${dir}/best.pt"
        for size in ${QUANT_SIZES}; do
            [[ -f "${dir}/yolo11s_seed${seed}_${size}.onnx" ]] && continue
            "${DC_GPU[@]}" bash -c "yolo export model=${dir}/best.pt format=onnx imgsz=${size} \
                opset=17 dynamic=False simplify=True batch=1 && mv ${dir}/best.onnx ${dir}/yolo11s_seed${seed}_${size}.onnx"
        done
    done
    (cd results/quant/onnx && sha256sum */*.onnx */best.pt > SHA256SUMS)
}

FAMILY_MODELS="yolov8s yolov9s yolov10s yolo12s yolo26s rtdetr-l"

start_gpu_monitor() {
    # Temperatura, clock, potência e motivo de redução de clock da RTX a 1 Hz (não fatal).
    nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,clocks.max.sm,power.draw,clocks_event_reasons.active \
        --format=csv -l 1 > "results/logs/gpu_${PHASE}_${RUN_ID}.csv" 2>/dev/null &
    GPU_MON_PID=$!
    trap 'kill ${GPU_MON_PID} 2>/dev/null || true' EXIT
}

models_smoke() {
    echo "=== Ensaio curto da fase models (1 época, 5% do train, 2 seeds; não é resultado) ==="
    check_dataset
    check_derived
    rm -rf runs/smoke_family results/family_smoke
    mkdir -p results/family_smoke
    "${DC_GPU[@]}" python src/train.py --config configs/train_family_smoke.yaml \
        --project runs/smoke_family --log results/family_smoke/train_log.csv
    "${DC_GPU[@]}" python src/benchmark.py --config configs/experiment.yaml --weights-root runs/smoke_family \
        --models yolo11s ${FAMILY_MODELS} --tags native --sizes 640 --seeds 0 1 \
        --no-plots --out results/family_smoke/grid_family_smoke.csv
    "${DC_GPU[@]}" python src/compare_models.py --csv results/family_smoke/grid_family_smoke.csv \
        --expected-seeds 2 --primary-imgsz 640
    echo "Ensaio OK: a fase models pode ser disparada."
}

compare_models() {
    echo "=== Comparação de modelos (pré-registrada): treino 640, 4 seeds ==="
    check_dataset
    check_derived
    if [[ -f results/docker_image.txt ]]; then
        echo "Imagem dos treinos originais: $(cat results/docker_image.txt)"
        echo "Imagem atual:                 $(docker image inspect sp-edge:gpu --format '{{json .RepoDigests}} {{.Id}}')"
    fi
    local model seed todo
    for model in ${FAMILY_MODELS}; do
        todo=""
        for seed in 0 1 2 67; do
            [[ -f "runs/train/${model}_visdrone_seed_${seed}/weights/best.pt" ]] || todo="${todo} ${seed}"
        done
        if [[ -n "${todo}" ]]; then
            "${DC_GPU[@]}" python src/train.py --config configs/train_family.yaml \
                --models "${model}" --seeds ${todo} --log results/train_log_family.csv
        else
            echo "[SKIP] ${model}: 4 seeds já treinadas"
        fi
    done

    echo "=== Avaliação: native x {640, 960, 1280}, conf=0.25 (YOLO11s reavaliado como controle) ==="
    mkdir -p results/family
    [[ -f results/family/grid_family.csv ]] && mv results/family/grid_family.csv "results/family/grid_family_${RUN_ID}.bak.csv"
    "${DC_GPU[@]}" python src/benchmark.py --config configs/experiment.yaml --weights-root runs/train \
        --models yolo11s ${FAMILY_MODELS} --tags native --sizes 640 960 1280 --seeds 0 1 2 67 \
        --no-plots --out results/family/grid_family.csv
    "${DC_GPU[@]}" python src/compare_models.py --csv results/family/grid_family.csv \
        | tee results/family/compare_models.txt
}

pre_quant() {
    check_checkpoints
    check_dataset
    check_derived

    echo "=== Arquivando saídas anteriores ==="
    archive_results

    echo "=== Grade completa em conf=0.25 (sem curvas PR: já existem e não dependem do conf) ==="
    "${DC_GPU[@]}" python src/benchmark.py --config configs/experiment.yaml \
        --weights-root runs/train --seeds 0 1 2 67 --no-plots
    make summary
    make figs

    echo "=== Predições salvas (conf=0.001) para diagnóstico offline ==="
    "${DC_GPU[@]}" python src/save_predictions.py --models yolo11s --seeds ${PRED_SEEDS} \
        --tags native 720p --sizes ${QUANT_SIZES}
    verify_predictions

    echo "=== Calibração INT8 (somente train) e ONNX das candidatas ==="
    "${DC_GPU[@]}" python src/make_calibration_set.py --n 300 --seed 0
    export_onnx

    ls -lh results/grid.csv results/grid_per_class.csv results/predictions/index.csv \
        results/quant/calib_train.txt results/quant/onnx/SHA256SUMS
}

require_command docker
require_command git
require_command make
require_command nvidia-smi

record_provenance
# O git não roda dentro do container; train.py/benchmark.py leem estes valores.
export GIT_COMMIT="$(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain)" ]]; then export GIT_DIRTY=true; else export GIT_DIRTY=false; fi

case "${PHASE}" in
    prepare)
        prepare
        ;;
    train)
        train_models
        ;;
    evaluate)
        evaluate_val
        ;;
    pre-quant)
        start_gpu_monitor
        pre_quant
        ;;
    models-smoke)
        models_smoke
        ;;
    models)
        start_gpu_monitor
        compare_models
        ;;
    all)
        prepare
        train_models
        evaluate_val
        ;;
    *)
        echo "Uso: $0 {prepare|train|evaluate|pre-quant|models-smoke|models|all}" >&2
        exit 2
        ;;
esac

echo
echo "Concluído. Log: ${LOG_FILE}"
echo "O test-dev não foi executado. Congele primeiro hipótese, modelo, resolução e threshold."
