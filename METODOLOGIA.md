# Protocolo científico e interpretação dos resultados

## Pergunta central

O experimento estima como três decisões controláveis — modelo, resolução máxima
de captura/transmissão e `imgsz` da inferência — alteram simultaneamente a
detecção e o tempo drone→alerta sob diferentes larguras de banda.

A grade completa possui 3 modelos × 4 seeds × 7 limites de captura × 5
tamanhos de inferência = 420 células. Cada célula avalia as mesmas imagens, o
que permite comparações pareadas entre configurações.

Os checkpoints COCO são usados somente como inicialização de transfer learning.
Treino, seleção e teste usam a taxonomia original de dez classes do VisDrone;
não existe remapeamento de rótulos para COCO no protocolo final.

## O que é medido

- `mAP50` e `mAP50_95`: qualidade ao longo da curva confiança–precisão–recall.
- `recall_op` e `precision_op`: macro-médias calculadas diretamente de TP, FP e
  FN em `conf=0.25` e IoU de matching 0,50. Não são o ponto de máximo F1.
- `recall_micro_op` e `precision_micro_op`: versões ponderadas pela quantidade
  de instâncias.
- Métricas por classe, parâmetros, GFLOPs, payload decimal em kB, preprocess,
  forward mediano/p95 e pós-processamento no threshold operacional.
- Média, desvio-padrão e IC95% entre os quatro treinamentos.

Macro-recall dá o mesmo peso às dez classes. Micro-recall dá mais peso às
classes frequentes. Para uma aplicação de segurança, reporte também a pior
classe e não apenas a média.

## Modelo de latência

O modelo analítico é:

```text
T_e2e = overhead + encode + transmissão + preprocess + forward + pós-processamento
encode = ms_por_megapixel × média real de pixels por quadro
transmissão = payload_kB × 8 / banda_Mbit_s
```

Há duas colunas de fim a fim: uma usa forward mediano e outra usa forward p95.
Elas são estimativas condicionadas aos coeficientes de `configs/experiment.yaml`.
Para afirmar latência real, calibre encode/overhead e substitua a transmissão
analítica pelas medições de `src/transmission.py` no enlace verdadeiro.

## Seleção e teste independente

O Pareto é calculado somente depois de agregar as seeds. Por padrão uma
configuração é elegível apenas quando o limite inferior do IC95% de recall
atinge `recall_floor`. Entre as elegíveis, escolhe-se a menor latência média por
largura de banda.

`make figs` produz `results/frozen_configs.yaml`, contendo threshold, regra,
hash da grade de validação e a união das configurações escolhidas. O test-dev
não pode ser avaliado sem esse manifesto. `make test-sweep` executa apenas as
configurações congeladas; não refaz a busca completa no teste.

O test-dev confirma generalização. Ele não deve ser usado para alterar piso,
threshold, modelo, resolução, `imgsz` ou hipótese.

## O que será descoberto ao final

1. Qual modelo entrega maior mAP/recall em cada orçamento computacional.
2. Quanto recall é perdido ao limitar a captura a 720, 600, 480, 360, 240 ou
   180 pixels de altura máxima.
3. Quando aumentar `imgsz` recupera informação útil e quando apenas amplia
   pixels já degradados sem ganho proporcional.
4. A interação captura×`imgsz`: a resolução do sensor e a entrada da rede não
   são o mesmo fator.
5. A interação modelo×resolução: um modelo maior pode ser mais robusto à
   degradação ou apenas mais lento.
6. O ponto em que o gargalo muda de transmissão para computação conforme o
   enlace passa de 2 para 50 Mbit/s.
7. Quais configurações são dominadas e quais pertencem à fronteira de Pareto.
8. A política de menor latência que satisfaz o piso de recall em cada banda.
9. Quais classes são mais vulneráveis à redução, compressão e baixa resolução
   de inferência.
10. Quanto a conclusão varia entre seeds e se a vantagem observada é estável.
11. A diferença entre latência típica e cauda p95 no computador RTX 5070/i9.
12. Se as escolhas feitas em `val` se mantêm no test-dev independente.

## Papel do computador RTX 5070/i9

A RTX 5070 executa treino e forward; o i9 participa de leitura, decodificação,
letterbox, DataLoader e pós-processamento. Portanto o resultado de latência é
específico ao conjunto GPU, CPU, RAM, armazenamento, driver, Docker, PyTorch,
Ultralytics e estado térmico/energético registrados. A RTX 5070 é uma bancada
dedicada, não uma NPU edge: os resultados nela comparam configurações nesse
hardware, mas não provam latência ou energia em Jetson, Coral ou no drone.

Durante a medição, mantenha clocks/política de energia, temperatura, processos
concorrentes e versão do driver estáveis. Repita sessões independentes se a
latência for uma variável primária do artigo.

## Limites que devem aparecer no artigo

- O experimento usa imagens estáticas e JPEG, não um fluxo H.264/H.265 com fila.
- Banda constante não modela jitter, perda, retransmissão ou handover.
- A soma de médias não é uma distribuição fim a fim real.
- `native` significa dimensão original; numa rodada Q85 ainda há recompressão.
- 720p e 600p são limites máximos: imagens originalmente menores não sofrem
  upsample nessa etapa.
- As métricas YOLO são adequadas para comparação interna. Para comparação com
  o leaderboard VisDrone, exporte detecções e execute também o toolkit oficial,
  que desconsidera detecções em regiões ignoradas. As anotações originais são
  preservadas em `annotations/<split>` para isso.
- Quatro seeds quantificam variabilidade de treino, mas não toda a incerteza de
  cidades, clima, altitude e domínio de implantação.
- O piso de recall precisa de justificativa operacional; 0,60 é uma hipótese,
  não uma constante universal.

## Arquivos finais

```text
results/train_log.csv                         proveniência de cada treino
results/environment.lock.txt                  ambiente Python efetivamente usado
results/docker_image.txt                      identidade da imagem Docker
results/dataset_audit.json                    contagens, hashes e teste de vazamento
results/grid.csv                              420 células brutas de validação
results/grid_per_class.csv                    métricas por classe
results/summary/summary_by_seed.csv           média, DP e IC95%
results/grid_com_latencia.csv                 uma linha agregada por configuração
results/config_otima_por_banda.csv            política selecionada em val
results/frozen_configs.yaml                   contrato imutável para o teste
results/figs/05_pareto_recall_op.png          fronteira por banda
results/test_grid.csv                         confirmação independente
```

Resultados de smoke servem somente para verificar interfaces e nunca entram em
tabelas, testes estatísticos ou conclusões.
