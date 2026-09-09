.PHONY: help build-dev build-gpu dev gpu smoke mock check-code data download-test convert-test train train-smoke sweep test-sweep summary figs tx-server tx-client check clean

CFG ?= configs/experiment.yaml
DC  := docker compose -f docker/docker-compose.yml

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | column -t -s "$$(printf '\t')"

build-dev:  ## constrói a imagem CPU (máquina local)
	$(DC) --profile dev build

build-gpu:  ## constrói a imagem CUDA 12.8 (servidor RTX 5070)
	$(DC) --profile gpu build

dev:        ## shell na imagem local
	$(DC) --profile dev run --rm dev bash

gpu:        ## shell na imagem do servidor
	$(DC) --profile gpu run --rm gpu bash

check:      ## sanidade da GPU: precisa imprimir compute capability 12.0
	$(DC) --profile gpu run --rm gpu python -c "\
import torch; \
print('torch', torch.__version__, '| cuda', torch.version.cuda); \
print('gpu  ', torch.cuda.get_device_name(0)); \
print('cc   ', torch.cuda.get_device_capability(0)); \
x=torch.randn(4096,4096,device='cuda'); print('matmul', (x@x).sum().item())"

smoke:      ## pipeline inteiro em 20 imagens, na CPU local (~2 min)
	$(DC) --profile dev run --rm dev bash -c "\
python src/downsample.py --config $(CFG) --smoke 20 && \
python src/benchmark.py --config $(CFG) --smoke --device cpu --out results/smoke.csv && \
python src/plot_results.py --csv results/smoke.csv --config $(CFG) --outdir results/figs_smoke"

mock: smoke  ## alias legível para a validação local rápida

check-code:  ## valida sintaxe e interfaces sem dataset ou GPU
	$(DC) --profile dev run --rm dev bash -c "python -m py_compile src/*.py scripts/*.py && python src/train.py --help >/dev/null && python src/benchmark.py --help >/dev/null && python src/download_visdrone.py --help >/dev/null && python src/transmission.py --help >/dev/null"

download-test:  ## baixa o VisDrone-DET test-dev oficial
	$(DC) --profile gpu run --rm gpu python src/download_visdrone.py --out /data/raw

convert-test:  ## converte o test-dev baixado para labels YOLO originais
	$(DC) --profile gpu run --rm gpu python scripts/visdrone2yolo.py --src /data/raw/VisDrone2019-DET-test-dev --dst /data/datasets/VisDrone --split test

data:       ## gera todos os splits degradados (servidor)
	$(DC) --profile gpu run --rm gpu python src/downsample.py --config $(CFG)

train:      ## treina os três modelos nas quatro seeds (execução longa)
	$(DC) --profile gpu run --rm gpu python src/train.py --config configs/train.yaml

train-smoke: ## testa o loop de treino em CPU com uma seed e poucas imagens
	$(DC) --profile dev run --rm dev python src/train.py --config configs/train.yaml --smoke

sweep:      ## varredura completa da grade (servidor)
	$(DC) --profile gpu run --rm gpu python src/benchmark.py --config $(CFG) --weights-root runs/train --seeds 0 1 2 67

test-sweep: ## avaliação final no test-dev convertido, sem alterar escolhas
	$(DC) --profile gpu run --rm gpu python src/benchmark.py --config $(CFG) --split test --weights-root runs/train --seeds 0 1 2 67 --out results/test_grid.csv

summary:    ## média, desvio-padrão e IC95% entre seeds
	$(DC) --profile gpu run --rm gpu python src/summarize_results.py --csv results/grid.csv

figs:       ## gráficos + tabela de configuração ótima
	$(DC) --profile gpu run --rm gpu python src/plot_results.py --csv results/grid.csv --config $(CFG)

tx-server:  ## servidor TCP; use PORT=5000 e HOST=0.0.0.0
	$(DC) --profile gpu run --rm --service-ports gpu python src/transmission.py server --host $(or $(HOST),0.0.0.0) --port $(or $(PORT),5000)

tx-client:   ## cliente TCP; use HOST=IP_DO_SERVIDOR DIR=/data/derived/.../images
	$(DC) --profile dev run --rm dev python src/transmission.py client --host $(HOST) --port $(or $(PORT),5000) --input $(DIR)

clean:
	rm -rf results/figs* results/*.csv **/__pycache__
