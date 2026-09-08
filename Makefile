.PHONY: help build-dev build-gpu dev gpu smoke data sweep figs check clean

CFG ?= configs/experiment.yaml
DC  := docker compose -f docker/docker-compose.yml

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | column -t -s "$$(printf '\t')"

build-dev:  ## constrói a imagem CPU (máquina local)
	$(DC) --profile dev build

build-gpu:  ## constrói a imagem CUDA 12.8 (servidor RTX 5080)
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

data:       ## gera todos os splits degradados (servidor)
	$(DC) --profile gpu run --rm gpu python src/downsample.py --config $(CFG)

sweep:      ## varredura completa da grade (servidor)
	$(DC) --profile gpu run --rm gpu python src/benchmark.py --config $(CFG)

figs:       ## gráficos + tabela de configuração ótima
	$(DC) --profile gpu run --rm gpu python src/plot_results.py --csv results/grid.csv --config $(CFG)

clean:
	rm -rf results/figs* results/*.csv **/__pycache__
