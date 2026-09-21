#!/usr/bin/env bash
set -Eeuo pipefail

# Executa o experimento no servidor sem tocar no test-dev.
# Uso:
#   DATA_DIR=/mnt/storage/sp-edge-data bash scripts/run_server_experiment.sh all
#   DATA_DIR=/mnt/storage/sp-edge-data bash scripts/run_server_experiment.sh prepare
#   DATA_DIR=/mnt/storage/sp-edge-data bash scripts/run_server_experiment.sh train
#   DATA_DIR=/mnt/storage/sp-edge-data bash scripts/run_server_experiment.sh evaluate

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

require_command docker
require_command git
require_command make
require_command nvidia-smi

record_provenance

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
    all)
        prepare
        train_models
        evaluate_val
        ;;
    *)
        echo "Uso: $0 {prepare|train|evaluate|all}" >&2
        exit 2
        ;;
esac

echo
echo "Concluído. Log: ${LOG_FILE}"
echo "O test-dev não foi executado. Congele primeiro hipótese, modelo, resolução e threshold."
