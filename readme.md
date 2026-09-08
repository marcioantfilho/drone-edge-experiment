sp-edge — Resolução × Latência × Recall para detecção aérea no edge

Infraestrutura experimental para a pergunta:

Qual a menor resolução que o drone pode capturar e transmitir ao edge mantendo recall aceitável, e qual o custo em latência de cada escolha?

Desenvolvimento local em CPU (Docker), execução pesada no servidor com RTX 5070. O mesmo proj roda nos dois lugares.

1. Protocolo experimental

O experimento é fatorial em dois eixos, deliberadamente separados:

Fator	O que é	O que ele controla
A — resolução de captura	o que o sensor capta e o link transmite (1080p…240p)	perda de informação → mexe no recall
B — imgsz da inferência	o tensor de entrada da rede (1280…320)	custo computacional → mexe na latência

Colapsar os dois num eixo só é o erro mais comum nesse tipo de estudo: fica impossível dizer se o recall caiu porque o pixel do alvo sumiu ou porque a rede rodou menor. A grade 2D responde as duas perguntas de uma vez, e a diagonal (captura == imgsz) continua disponível para comparação com a literatura.

Detalhe que economiza semanas: rótulos YOLO são normalizados em [0,1]. Reduzir a imagem não muda o rótulo. O downsample.py reaproveita os .txt por hardlink — nada de reconversão de bounding box, nada de erro de escala.

Latência fim-a-fim
T_total = T_overhead + T_encode + T_transmissão + T_inferência
                       k·Mpx      payload/banda    pre+fwd+pós

Reduzir a resolução corta T_transmissão quadraticamente (é área) e o recall de forma aproximadamente sigmoide. Existe portanto um joelho, e ele muda conforme a banda do enlace — num link de 2 Mbit/s vale a pena sacrificar recall por payload; num de 50 Mbit/s, não. Isso sai pronto em results/config_otima_por_banda.csv.

Dois pontos de operação

Toda célula é avaliada duas vezes:

conf=0.001 → curva PR completa. É o número comparável com papers.
conf=0.25 → recall no ponto de operação real do sistema de alerta. É este que vale para a dissertação: despachar viatura com conf=0.001 inunda o COPOM de falso positivo.
2. Setup
Local
bash
make build-dev
make smoke          # pipeline inteiro em 20 imagens, ~2 min

Servidor
Pré-requisito, uma vez:

bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
bash
make build-gpu
make check          # DEVE imprimir compute capability (12, 0)

RTX 50xx = Blackwell = sm_120. Só funciona com PyTorch ≥ 2.7 compilado contra CUDA 12.8+. Qualquer imagem com tag -cuda12.6- ou torch 2.6 falha com no kernel image is available for execution on the device. E nunca deixe o pip reinstalar torch por dependência transitiva — ele traz um build sem sm_120 e a GPU para de funcionar silenciosamente. Por isso torch não está no requirements.txt.

Dataset

O VisDrone.yaml do Ultralytics baixa e converte para formato YOLO sozinho:

bash
docker compose -f docker/docker-compose.yml --profile gpu run --rm gpu \
  python -c "from ultralytics.utils.downloads import *; from ultralytics import YOLO; \
             YOLO('yolo11n.pt').val(data='VisDrone.yaml', imgsz=640)"

Depois aponte dataset.base_yaml no configs/experiment.yaml para o VisDrone.yaml que ficou em /data/datasets/.

3. Fluxo
bash
make data     # gera os 6 splits degradados + manifest de payload
make sweep    # varredura da grade (modelos × captura × imgsz)
make figs     # 5 figuras + tabela de configuração ótima por banda

Saídas em results/:

Arquivo	Conteúdo
grid.csv	uma linha por célula da grade, com recall/mAP/latência/payload
grid_com_latencia.csv	idem + latência fim-a-fim para cada banda
config_otima_por_banda.csv	a tabela que responde a pergunta da dissertação
figs/01_..._vs_captura.png	recall × resolução capturada, uma curva por imgsz
figs/02_heatmap_....png	grade fatorial completa
figs/03_acuracia_vs_latencia.png	eixo duplo: recall e ms vs imgsz
figs/04_payload_transmissao.png	kB/quadro e ms de rádio por resolução
figs/05_pareto_....png	fronteira de Pareto por banda de enlace
