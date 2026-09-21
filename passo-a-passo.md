# Execução completa na máquina Ubuntu com RTX 5070

Pré-requisitos: driver NVIDIA, Docker Engine, Docker Compose, NVIDIA Container
Toolkit, Git e Make já instalados. A execução é local na máquina, sem SSH.

## 1. Publicar a versão final no computador de desenvolvimento

```bash
cd /caminho/para/drone-edge-experiment
git add -A
git commit -m "Prepara experimento científico VisDrone"
git push origin main
git status --porcelain
git rev-parse HEAD
git rev-parse origin/main
```

O `status` deve ficar vazio e os dois hashes devem ser iguais.

## 2. Obter o projeto na máquina Ubuntu

Primeira instalação:

```bash
cd ~
git clone https://github.com/marcioantfilho/drone-edge-experiment.git
cd drone-edge-experiment
git switch main
git pull --ff-only origin main
git status --porcelain
```

Se o projeto já estiver clonado:

```bash
cd ~/drone-edge-experiment
git switch main
git pull --ff-only origin main
git status --porcelain
```

## 3. Preparar o diretório de dados

```bash
export DATA_DIR=/mnt/storage/sp-edge-data
sudo mkdir -p "$DATA_DIR"/{raw,datasets,derived}
sudo chown -R "$USER":"$USER" "$DATA_DIR"
mkdir -p results/logs
df -h "$DATA_DIR"
```

Reserve aproximadamente 100 GB livres considerando os discos de dados e do
Docker.

## 4. Verificar Docker e RTX 5070

```bash
docker --version
docker compose version
nvidia-smi
docker run --rm --runtime=nvidia --gpus all \
  nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

## 5. Construir e testar a imagem científica

```bash
cd ~/drone-edge-experiment
export DATA_DIR=/mnt/storage/sp-edge-data
make build-gpu
make check
```

## 6. Baixar train e val oficiais

```bash
make download-train-val
```

## 7. Converter train e val para YOLO

```bash
docker compose -f docker/docker-compose.yml --profile gpu run --rm gpu \
  python scripts/visdrone2yolo.py \
  --src /data/raw/VisDrone2019-DET-train \
  --dst /data/datasets/VisDrone --split train

docker compose -f docker/docker-compose.yml --profile gpu run --rm gpu \
  python scripts/visdrone2yolo.py \
  --src /data/raw/VisDrone2019-DET-val \
  --dst /data/datasets/VisDrone --split val
```

## 8. Executar a preparação completa

```bash
git status --porcelain
export DATA_DIR=/mnt/storage/sp-edge-data
bash scripts/run_server_experiment.sh prepare
```

O primeiro comando não deve imprimir nada. A preparação executa testes,
auditoria do dataset e pré-download dos pesos-base.

## 9. Executar o smoke test na GPU

```bash
make smoke-gpu
```

Os resultados do smoke usam nomes separados e não pertencem ao artigo.

## 10. Executar treino e validação completos

No terminal principal:

```bash
cd ~/drone-edge-experiment
export DATA_DIR=/mnt/storage/sp-edge-data
systemd-inhibit --what=sleep:idle --mode=block \
  --why="Experimento VisDrone em execução" \
  bash -c 'bash scripts/run_server_experiment.sh train && \
           bash scripts/run_server_experiment.sh evaluate'
```

Mantenha esse terminal aberto e não suspenda ou reinicie a máquina.

## 11. Acompanhar a execução

Em um segundo terminal:

```bash
cd ~/drone-edge-experiment
tail -f "$(ls -1t results/logs/server_*.log | head -1)"
```

Em outro terminal:

```bash
watch -n 2 nvidia-smi
```

## 12. Conferir os 12 checkpoints

```bash
cd ~/drone-edge-experiment
find runs/train -path '*/weights/best.pt' ! -path '*_smoke*' | sort
find runs/train -path '*/weights/best.pt' ! -path '*_smoke*' | wc -l
```

O último comando deve imprimir `12`.

## 13. Conferir as 420 células de validação

```bash
export DATA_DIR=/mnt/storage/sp-edge-data
docker compose -f docker/docker-compose.yml --profile gpu run --rm gpu \
  python -c "import pandas as pd; d=pd.read_csv('results/grid.csv'); \
print('linhas:',len(d)); print(d.groupby(['model','seed']).size()); \
assert len(d)==420; \
assert not d.duplicated(['model','seed','capture_tag','imgsz','precision_mode']).any()"
```

## 14. Revisar a seleção feita em val

```bash
cat results/config_otima_por_banda.csv
cat results/frozen_configs.yaml
```

Não altere a configuração depois dessa seleção. A latência de enlace é uma
estimativa analítica, salvo se seus coeficientes tiverem sido calibrados antes.

## 15. Executar o test-dev

Somente depois de revisar e congelar a seleção do passo 14:

```bash
cd ~/drone-edge-experiment
export DATA_DIR=/mnt/storage/sp-edge-data
systemd-inhibit --what=sleep:idle --mode=block \
  --why="Avaliação VisDrone test-dev em execução" \
  bash -c 'make download-test && make convert-test && make test-sweep'
```

Não use o resultado do test-dev para escolher novamente modelos ou parâmetros.

## 16. Conferir os resultados finais

```bash
ls -lh \
  results/grid.csv \
  results/grid_per_class.csv \
  results/grid_com_latencia.csv \
  results/config_otima_por_banda.csv \
  results/frozen_configs.yaml \
  results/test_grid.csv \
  results/test_grid_per_class.csv
```

Os gráficos estão em `results/figs/`, os resumos estatísticos em
`results/summary/` e os logs em `results/logs/`.
