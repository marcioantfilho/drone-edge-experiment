
## Passo a passo completo no Ubuntu Desktop com RTX 5070

### 0. O que este roteiro pressupõe

- Ubuntu Desktop x86_64 com acesso administrativo (`sudo`).
- Uma RTX 5070 visível no host, um i9/Core Ultra e pelo menos 32 GB de RAM.
- Conexão com a internet para Docker Hub, GitHub, Google Drive e pesos
  Ultralytics.
- Aproximadamente 100 GB livres, considerando dados e armazenamento Docker.
- Nenhuma outra carga relevante de CPU/GPU durante as medições finais.

Dataset, checkpoints e resultados não são enviados ao GitHub pelo fluxo
normal: estão ignorados no `.gitignore`. O GitHub guarda o código e o protocolo
que tornam a execução reproduzível; os artefatos científicos devem ser
arquivados separadamente ao final.

Este roteiro executa integralmente a campanha computacional: treino, `val`,
agregação das seeds, Pareto, congelamento da seleção e confirmação no test-dev.
A latência de transmissão usada no Pareto continua sendo um cenário analítico
definido em `configs/experiment.yaml`. Sem medir e calibrar um enlace físico,
ela não deve ser apresentada como tempo drone→servidor observado em campo.

### 1. Preparar o sistema uma unica vez

Este projeto foi configurado para uma RTX 5070 (Blackwell, compute capability
12.0), com PyTorch 2.11 e CUDA 12.8 dentro do container. O CUDA Toolkit nao
precisa ser instalado diretamente no Ubuntu; no host sao necessarios o driver
NVIDIA, Docker Engine, o plugin Docker Compose e o NVIDIA Container Toolkit.

1. Atualize o Ubuntu e instale o driver recomendado:

   ```bash
   sudo apt update
   sudo apt upgrade -y
   sudo apt install -y git make curl ca-certificates ubuntu-drivers-common tmux
   sudo ubuntu-drivers install
   sudo reboot
   ```

2. Depois do reboot, confirme que a RTX 5070 aparece:

   ```bash
   nvidia-smi
   ```

3. Instale o Docker Engine e o plugin Compose seguindo a documentacao oficial:

   - https://docs.docker.com/engine/install/ubuntu/
   - https://docs.docker.com/compose/install/linux/

   Adicione seu usuario ao grupo `docker`, saia da sessao e entre novamente:

   ```bash
   sudo usermod -aG docker "$USER"
   ```

   Confirme a instalacao:

   ```bash
   docker --version
   docker compose version
   docker run --rm hello-world
   ```

4. Instale e configure o NVIDIA Container Toolkit conforme a documentacao
   oficial: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

   Depois de instalar o pacote `nvidia-container-toolkit`, configure o runtime:

   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

   Confirme que a GPU também está acessível **dentro** de um container. Ver a
   GPU apenas com `nvidia-smi` no host não valida o NVIDIA Container Toolkit:

   ```bash
   docker run --rm --runtime=nvidia --gpus all \
     nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
   ```

### 2. Publicar a versão científica e obter o repositório no servidor

No computador em que o código foi revisado, `git commit` cria o commit local e
`git push` publica esse commit no GitHub. `git status --porcelain` apenas
verifica se sobraram mudanças fora do commit; ele não envia nada:

```bash
git status --short
git diff --check
git add -A
git commit -m "Congela protocolo do experimento VisDrone"
git push origin main
git status --porcelain
git rev-parse HEAD
git rev-parse origin/main
```

O `status --porcelain` não deve imprimir nada, e os dois hashes devem ser
iguais. Não use `git add` para datasets, checkpoints ou resultados; eles já são
ignorados pelo projeto.

Onde há a 5070 ou uma nvidia equivalente, faça o clone na primeira instalação:

```bash
git clone https://github.com/marcioantfilho/drone-edge-experiment.git
cd drone-edge-experiment
git switch main
git pull --ff-only origin main
git status --porcelain
git rev-parse HEAD
```

Se o repositório já existir, entre nele e execute somente `git switch main` e
`git pull --ff-only origin main`. O `status --porcelain` também precisa ficar
vazio no servidor. O executor recusa uma árvore suja porque o hash do commit
não reproduziria alterações locais.

Agora escolha um disco com bastante espaço. Neste exemplo, todos os dados
persistentes ficam em `/mnt/storage/sp-edge-data`:

```bash
cd /caminho/para/drone-edge-experiment
export DATA_DIR=/mnt/storage/sp-edge-data
mkdir -p "$DATA_DIR"/raw "$DATA_DIR"/datasets "$DATA_DIR"/derived results/logs
df -h "$DATA_DIR"
DOCKER_ROOT_DIR="$(docker info --format '{{.DockerRootDir}}')"
df -h "$DOCKER_ROOT_DIR"
```

O valor de `DATA_DIR` precisa ser mantido em todos os comandos. Dentro do
container, esse diretorio sempre aparece como `/data`. Como margem conservadora,
deixe pelo menos 100 GB livres somando o disco dos dados e o disco usado pelo
Docker; a imagem CUDA, os dados derivados e os 12 treinamentos ocupam espaços
diferentes. Se a criação em `/mnt/storage` der `Permission denied`, prepare o
diretório uma única vez:

```bash
sudo mkdir -p /mnt/storage/sp-edge-data
sudo chown -R "$USER":"$USER" /mnt/storage/sp-edge-data
```

Revise `configs/train.yaml` e `configs/experiment.yaml` antes de publicar o
commit. Não altere seeds, modelos, thresholds, resolução, qualidade JPEG ou
regra de seleção depois de observar os resultados de `val`.

Em especial, decida antes do commit se o artigo usará apenas os cenários
analíticos de 2–50 Mbit/s ou coeficientes calibrados em uma rede física. Os
valores `encode_ms_per_mpixel` e `overhead_ms` atuais são hipóteses declaradas,
não medições do seu drone. Como eles influenciam a configuração escolhida no
Pareto, não devem ser ajustados depois de ver a seleção de `val`.

### 3. Preparar o VisDrone train e val

Construa a imagem e baixe automaticamente os pacotes oficiais
`VisDrone2019-DET-train` e `VisDrone2019-DET-val`:

```bash
export DATA_DIR=/mnt/storage/sp-edge-data
make build-gpu
make download-train-val
```

O downloader valida CRC, estrutura e as quantidades esperadas. Se o Google
Drive limitar o download automatizado, baixe manualmente train e val na seção
**Task 1: Object Detection in Images** do repositório oficial:

https://github.com/VisDrone/VisDrone-Dataset#task-1-object-detection-in-images

Não baixe os conjuntos de vídeo, MOT ou SOT com nomes parecidos. Tanto o
download automático quanto o manual devem resultar nesta estrutura:

```text
/mnt/storage/sp-edge-data/raw/VisDrone2019-DET-train/images/
/mnt/storage/sp-edge-data/raw/VisDrone2019-DET-train/annotations/
/mnt/storage/sp-edge-data/raw/VisDrone2019-DET-val/images/
/mnt/storage/sp-edge-data/raw/VisDrone2019-DET-val/annotations/
```

Converta os dois splits para YOLO:

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

Ao final deve existir `/mnt/storage/sp-edge-data/datasets/VisDrone.yaml`,
alem de `images/train`, `labels/train`, `images/val` e `labels/val`. Valide a
GPU e o dataset antes da execucao longa:

```bash
DATA_DIR=/mnt/storage/sp-edge-data \
  bash scripts/run_server_experiment.sh prepare
```

O teste deve mostrar a RTX 5070, CUDA funcional, compute capability `(12, 0)`,
10 classes e os caminhos de `train` e `val`. A fase também executa os testes
unitários, audita contagens/labels/vazamento entre splits, registra as versões
do ambiente e baixa/abre os três pesos-base. Assim, uma falha de rede ou nome
de checkpoint acontece antes das horas de treinamento.

Execute então o teste de fogo curto na própria GPU:

```bash
export DATA_DIR=/mnt/storage/sp-edge-data
make smoke-gpu
```

Ele treina somente `yolo11n`, uma seed, duas épocas e avalia 20 imagens. Os
resultados usam nomes `*_smoke_gpu` e não podem sobrescrever `grid.csv`, as
figuras finais ou `frozen_configs.yaml`; o treino curto usa o sufixo `_smoke`.
O log de treino curto também é separado, e as curvas PR científicas ficam
desabilitadas nessa verificação.
O cache degradado temporário usa o mesmo diretório externo das variantes, mas
`make data` detecta o fingerprint diferente e o regenera com o `val` completo.
O smoke valida o encadeamento; seus números não entram no artigo.

### 4. Executar o experimento longo

Se você **ainda não executou** `prepare`, o comando único para todas as fases é:

```bash
DATA_DIR=/mnt/storage/sp-edge-data \
  bash scripts/run_server_experiment.sh all
```

Se já seguiu a seção anterior, não repita `all`: ele reconstruiria e validaria
a imagem outra vez. Rode `train` e `evaluate`, nessa ordem. Para resistir à
desconexão SSH, use `tmux`; para impedir a suspensão do Ubuntu, mantenha o
comando sob `systemd-inhibit`:

```bash
cd /caminho/para/drone-edge-experiment
tmux new -s spedge
export DATA_DIR=/mnt/storage/sp-edge-data
systemd-inhibit --what=sleep:idle --mode=block \
  --why="Experimento VisDrone em execucao" \
  bash -c 'bash scripts/run_server_experiment.sh train && \
           bash scripts/run_server_experiment.sh evaluate'
```

Para soltar a sessão sem interromper o experimento, pressione `Ctrl-b` e depois
`d`. Para voltar, use `tmux attach -t spedge`. Não reinicie a máquina durante
o treinamento.

O modo `all` executa, na ordem:

1. build e teste da imagem GPU;
2. validacao do dataset;
3. treino de 3 modelos x 4 seeds x 100 epocas;
4. verificacao dos 12 arquivos `best.pt`;
5. geracao das sete resolucoes de captura;
6. sweep completo no conjunto `val`;
7. media, desvio-padrao, IC95% e graficos.

O Intel Core Ultra/i9 será usado na carga, decodificação e pré/pós-processamento;
o treino e o forward serão executados na RTX 5070. O Compose não limita a
máquina a 12 GB de VRAM ou 32 GB de RAM: esses são os recursos esperados do
servidor. O cache do dataset usa disco para reduzir a pressão sobre a RAM.

### 5. Acompanhar a execucao

Em outro terminal, na raiz do repositório, encontre o log mais recente:

```bash
ls -1t results/logs/server_*.log | head
tail -f "$(ls -1t results/logs/server_*.log | head -1)"
```

Para acompanhar GPU, VRAM, temperatura e utilizacao:

```bash
watch -n 2 nvidia-smi
```

Para confirmar que a sessão continua ativa:

```bash
tmux ls
```

Opcionalmente, registre telemetria da GPU em outro terminal durante a campanha:

```bash
nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,memory.used,power.draw,clocks.sm \
  --format=csv -l 60 | tee results/logs/gpu_telemetry.csv
```

Interrompa a telemetria com `Ctrl-C` quando o experimento terminar. Para reduzir
variância de latência, não use a GPU para interface gráfica pesada, jogos ou
outros processos de ML durante a avaliação. Se o Ubuntu oferecer perfis de
energia, registre `powerprofilesctl get` e mantenha o mesmo perfil em toda a
campanha.

O treinamento completo deve produzir exatamente 12 checkpoints:

```bash
find runs/train -path '*/weights/best.pt' ! -path '*_smoke*' | sort
find runs/train -path '*/weights/best.pt' ! -path '*_smoke*' | wc -l
```

O segundo comando deve imprimir `12`.

### 5.1. Se a execução for interrompida

`make train` não sobrescreve diretórios existentes. Isso é deliberado: impede
misturar um treino antigo com o experimento atual. Não execute `all` de novo às
cegas. Identifique quais combinações já têm `best.pt` e execute somente as que
faltam, por exemplo:

```bash
docker compose -f docker/docker-compose.yml --profile gpu run --rm gpu \
  python src/train.py --config configs/train.yaml \
  --models yolo11s --seeds 67
```

Se o diretório dessa combinação existir mas não tiver um `best.pt` válido,
arquive-o antes de recomeçar a combinação; não o apague sem inspeção:

```bash
mv runs/train/yolo11s_visdrone_seed_67 \
   runs/train/yolo11s_visdrone_seed_67_incompleto
```

Depois que os 12 checkpoints existirem, execute somente `evaluate`. Uma
avaliação interrompida deve ser reiniciada desde o começo: os caches de imagens
serão reaproveitados, mas `grid.csv` será reconstruído para não misturar duas
campanhas de timing.

### 6. Resultados gerados no val

Quando o script terminar, a mensagem `Concluido` aparecera no log. Os
principais resultados estarao em:

```text
runs/train/<modelo>_visdrone_seed_<seed>/weights/best.pt
results/grid.csv
results/grid_per_class.csv
results/grid_com_latencia.csv
results/config_otima_por_banda.csv
results/frozen_configs.yaml
results/summary/summary_by_seed.csv
results/summary/model_comparison_by_seed.csv
results/summary/model_comparison.png
results/pr/
results/figs/
results/logs/
```

O script unico para no `val` de proposito. `make figs` agrega as quatro seeds e
grava `frozen_configs.yaml`. Revise e congele esse manifesto antes do teste.

Antes do test-dev, confira também se todas as 420 células de validação existem:

```bash
docker compose -f docker/docker-compose.yml --profile gpu run --rm gpu \
  python -c "import pandas as pd; d=pd.read_csv('results/grid.csv'); \
print('linhas:', len(d), 'esperado: 420'); \
print(d.groupby(['model','seed']).size()); \
assert len(d)==420; \
assert not d.duplicated(['model','seed','capture_tag','imgsz','precision_mode']).any()"
sha256sum configs/train.yaml configs/experiment.yaml results/frozen_configs.yaml
```

Não edite `configs/experiment.yaml` após gerar o manifesto; o test-sweep compara
o hash da configuração e recusará uma combinação diferente.

### 7. Executar o test-dev somente depois de congelar a configuracao

Em uma nova sessao de terminal, redefina `DATA_DIR`, baixe e converta o
test-dev e execute a avaliacao final:

```bash
cd /caminho/para/drone-edge-experiment
export DATA_DIR=/mnt/storage/sp-edge-data

make download-test
make convert-test
make test-sweep
```

O resultado independente sera salvo em `results/test_grid.csv` e as metricas
por classe em `results/test_grid_per_class.csv`. O comando recusa rodar sem o
manifesto de `val` e avalia somente as configuracoes nele congeladas. Nao use
esse conjunto para voltar e escolher hiperparametros.

### 8. Checklist imediatamente antes de rodar hoje

- `git status --porcelain` não imprime nada.
- `nvidia-smi` mostra a RTX 5070 sem outros processos pesados.
- O teste NVIDIA dentro do container funciona.
- `df -h "$DATA_DIR"` e `df -h "$DOCKER_ROOT_DIR"` têm espaço suficiente.
- Train possui 6471 imagens e val possui 548.
- `prepare` terminou sem erro e gravou `results/dataset_audit.json`.
- `make smoke-gpu` terminou; não existe `results/frozen_configs.yaml` vindo do
  smoke (apenas `results/frozen_configs_smoke_gpu.yaml`).
- `configs/train.yaml` e `configs/experiment.yaml` foram revisados e commitados.
- A máquina está ligada à energia, sem suspensão automática e sem outra carga
  relevante de CPU/GPU.

### 9. O que termina com este roteiro — e o que exige outro ensaio

Ao terminar a seção 7, estarão concluídos os resultados computacionais
reprodutíveis no VisDrone para a RTX 5070/i9, incluindo a confirmação
independente no test-dev.

Dois ensaios não fazem parte automática deste roteiro:

1. Medição física drone/rede/edge. `src/transmission.py` pode medir um caminho
   TCP real, mas uma campanha de campo ainda precisa definir transmissor,
   codec, sinal, distância, repetição, jitter, perda e energia.
2. Pontuação pelo toolkit oficial VisDrone. As métricas internas são adequadas
   para o experimento controlado, mas uma comparação direta com leaderboard
   deve exportar detecções e executar separadamente o avaliador oficial, que
   trata regiões ignoradas segundo o protocolo VisDrone.

Portanto, o roteiro sozinho basta para o experimento computacional descrito em
`METODOLOGIA.md`. Ele não substitui um protocolo de rede física nem o avaliador
oficial caso essas duas alegações sejam incluídas no artigo.
