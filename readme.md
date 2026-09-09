# sp-edge

Experimento reprodutivel para estudar a relacao entre resolucao capturada, latencia, transmissao e recall em deteccao de objetos aereos.

O experimento final usa o VisDrone original com 10 classes, tres modelos (YOLO11n, YOLO11s e YOLOv10n) e quatro seeds (0, 1, 2 e 67). O COCO foi removido do caminho principal: ele serviu apenas para o teste local antigo.

## Arquitetura

- `configs/train.yaml`: dataset, modelos, hiperparametros e seeds do treino.
- `configs/experiment.yaml`: resolucoes, compressao, avaliacao e enlace.
- `src/train.py`: treina cada modelo em cada seed e salva `best.pt`.
- `src/downsample.py`: cria resolucoes degradadas e o manifest de bytes.
- `src/benchmark.py`: mede val/test, latencia, PR e metricas por classe.
- `src/summarize_results.py`: calcula media, desvio-padrao e IC95% entre seeds.
- `src/plot_results.py`: produz graficos de resolucao, latencia e Pareto.
- `src/download_visdrone.py`: baixa o test-dev oficial com ground truth.
- `src/transmission.py`: mede transmissao TCP real entre cliente e servidor.

O checkpoint escolhido em cada treino e o `best.pt` do Ultralytics, selecionado pela metrica de fitness, dominada por mAP50-95. A escolha e feita apenas no `val`; o `test` e reservado para a avaliacao final.

## Preparacao

Na maquina local, para testar o pipeline:

```bash
make build-dev
make check-code
```

Esse primeiro teste nao precisa de dataset, GPU ou download. Com o dataset ja
montado, `make mock` executa tambem uma rodada curta de processamento e
graficos.

No servidor com RTX 5070:

```bash
make build-gpu
make check
```

`make check` deve mostrar uma GPU NVIDIA, CUDA funcional e compute capability 12.0. A imagem GPU usa PyTorch/CUDA compativel com Blackwell. O perfil `dev` e CPU-only e serve para mock e depuracao.

## Dataset

O YAML padrao e `/data/datasets/VisDrone.yaml`. Ele deve apontar para o VisDrone original em formato YOLO, com `train`, `val` e, depois, `test`.

Para baixar o test-dev oficial, que possui ground truth publico:

```bash
make download-test
make convert-test
```

O primeiro comando baixa da fonte oficial indicada pelo repositorio VisDrone; o segundo converte as anotacoes para as 10 classes originais. O nome local `test` representa o pacote oficial `test-dev`.

## Execucao final

Treino completo: tres modelos x quatro seeds.

```bash
make train
```

Os checkpoints ficam em `runs/train/<modelo>_visdrone_seed_<seed>/weights/best.pt`.

Gerar imagens degradadas:

```bash
make data
```

Avaliar a grade completa no `val`:

```bash
make sweep
make summary
make figs
```

O benchmark registra precision, recall, mAP50, mAP50-95, preprocessamento, forward, pos-processamento, p95, GFLOPs, payload e latencia fim-a-fim. Tambem grava `results/grid_per_class.csv`, incluindo `pedestrian`, `people` e as demais classes. O Ultralytics salva os artefatos de PR em `results/pr/`.

Somente depois de congelar modelo, resolucao e threshold, avaliar o conjunto independente:

```bash
make test-sweep
```

O conjunto de teste nao deve ser usado para escolher hiperparametros ou a configuracao final.

## Saidas

```text
results/grid.csv
results/grid_per_class.csv
results/test_grid.csv
results/summary/summary_by_seed.csv
results/summary/model_comparison_by_seed.csv
results/pr/
results/figs/
results/transmission_real.csv
```

O resumo por seed evita apresentar uma unica execucao como se fosse uma estimativa estavel. Para a dissertacao, reporte media, desvio-padrao e IC95% das quatro seeds, alem dos resultados do teste independente.

## Transmissao real

A formula de latencia do benchmark e uma estimativa. Para medir a rede real, rode o servidor na maquina edge e o cliente em outro host.

Servidor:

```bash
make tx-server PORT=5000
```

Cliente, em outro host:

```bash
make tx-client HOST=IP_DO_SERVIDOR DIR=/data/derived/visdrone_val_720p_q85/images
```

O cliente envia cada arquivo por TCP e grava bytes, tempo e throughput. Um teste em localhost valida o protocolo, mas nao representa um enlace drone-edge.

## Configuracao e mock

Para trocar de dataset, altere somente `dataset.base_yaml` em `configs/train.yaml` e `configs/experiment.yaml`, mantendo a convencao YOLO. Para trocar seeds, use o YAML ou sobrescreva:

```bash
python src/train.py --config configs/train.yaml --seeds 0
```

O mock executa poucas imagens, uma seed, poucos tamanhos e CPU. Ele valida imports, dataset, pesos, CSVs e graficos, mas nao produz resultado cientifico.

## Limites

- O test-dev tem ground truth; o test-challenge oficial nao deve ser usado como avaliacao quantitativa sem anotacoes publicas.
- A medicao TCP e real para o caminho de rede usado, mas nao modela outra rede.
- Para comparacao academica, reporte media, desvio-padrao e IC95% das quatro seeds, alem dos resultados do test independente.
