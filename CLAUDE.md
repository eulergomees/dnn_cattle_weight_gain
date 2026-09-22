# dnn_cattle_weight_gain

## Visão geral

Rede neural densa (DNN/MLP em **PyTorch**) que prevê o **Ganho Médio Diário (GMD, `gmd_kg_dia`)** de bovinos de corte. Projeto acadêmico de TCC — IFMG, Depto. de Engenharia e Computação (Prof. Ciniro Nametala; aluno Euler Gomes). Modelo `GMDNN`, ainda em desenvolvimento.

A DNN é o **modelo central do TCC** — arquitetura/hiperparâmetros **congelados** (decidido em reunião c/ orientador). O trabalho é um **estudo comparativo**: a rede contra baselines (regressão linear, XGBoost/RF e **TabPFN**), sobre os mesmos dados/split/CV. A comparação — com curva de aprendizado justificando quando cada família vence — é a contribuição. Ver "Onde paramos".

> **Escopo desta fase: apenas código e dados.** A redação do TCC (LaTeX, classe abntex2) fica para o fim do projeto, com o modelo pronto. **Não** produzir, editar ou sugerir texto do trabalho.

## Estrutura

> Repositório enxuto: o schema antigo (notebooks, datasets `cattle_*`, modelos `.pth`, `processar_formulario.py`) foi **removido** — recuperável pelo histórico do git se necessário.

- `data/dataset_por_animal_modelo_v3.csv` — **o dataset** (schema por-animal, alvo `gmd_kg_dia`). Detalhes em "Nova versão do dataset (por animal)".
- `support_scripts/ingest_sheet.py` — **ingestão de folha de campo manuscrita → v3**: recebe as linhas transcritas + constantes da fazenda, valida consistência (ganho=saída−entrada, gmd=ganho/dias), deriva clima (NASA POWER) e anexa. `import ingest_sheet as ing; ing.ingest(animais, fazenda, dry_run=True)`.
- `support_scripts/data_prep.py` — módulo de pré-processamento compartilhado (**schema v3**): `get_data` (8 preditores, alvo `gmd_kg_dia`; dropa ids + `peso_saida_kg` + 4 constantes + `precipitacao_acumulada_mm`), `get_cv` (**5 folds disjuntos de 49 animais de teste**, estratificados por fazenda — `FarmStratifiedSplit`, p/ tuning), `leave_one_property_out` (treina 2 fazendas/testa a 3ª, p/ robustez), `augment_train` (jitter = ruído de medição, só treino), métricas e `save_result`. Scaling fica nos notebooks (fit só no treino). Uso: `sys.path.append("support_scripts"); import data_prep as dp; d = dp.get_data()`.
- `eda_gmd.ipynb` — **EDA** do dataset v3 (visão geral, alvo, variância/constantes, correlações, multicolinearidade, GMD por suplemento). Kernel conda `tcc`.
- `dnn_gmd.ipynb` — **rede neural `GMDNN`** (MLP PyTorch): grid search (72 configs, 5 folds de 49 de teste; early stop 60 / scheduler 40), melhor config com/sem jitter, leave-one-property-out, predito×real OOF, grava em `results/model_comparison.csv`.
- **Baselines (Linear/RF/GB) — removidos do repositório** (`baselines/` e `sk_eval.py`; recuperáveis no commit `18984ce`). Ao refazê-los, usar `dp.get_cv()` novo p/ ficarem comparáveis.
- **A construir:** SHAP (interpretação do melhor modelo).

## Dados

- Coletados originalmente em **3 fazendas** da região de **Bambuí-MG**: Elvis (113), Sonico (61) e Humberto (71; via Excel de pesagens). **Sonico foi fundido em Elvis** (decisão de reunião, 2026-09-22) — ver "Fusão Sonico→Elvis". Dataset final: **2 fazendas, Elvis 174 / Humberto 71**.
- Schema e regras em "Nova versão do dataset (por animal)".

## Nova versão do dataset (por animal — EM COLETA)

> `data/dataset_por_animal_modelo_v3.csv` (**19 colunas**) tem **245 animais em 2 fazendas** (Elvis 174, Humberto 71 — Sonico fundido em Elvis). **Substitui o schema antigo** e torna obsoletos todos os números medidos antes (schema antigo + dados com sintéticas, e os números de 3-fazendas pré-fusão). Detalhes de coleta/constantes em "Onde paramos".

**Mudança central:** cada linha = **um animal** (pesagem de entrada → saída), não mais uma pesagem individual. Corrige pseudo-replicação e o vazamento de ter o mesmo animal em treino e teste.

- **Alvo:** `gmd_kg_dia` (GMD do ciclo). **Nunca prever `peso_saida_kg`.** Se precisar do peso de saída, derivar: `peso_saida = peso_entrada + gmd_predito × dias_permanencia`.
- **19 colunas → 13 preditores efetivos.** Removidas do schema: `media_digestibilidade_forragem_pct` (r≈0.97 com PB da forragem — integridade do SHAP), `media_taxa_lotacao_ua_ha` (coleta não-confiável; com 2 fazendas seria ~constante por propriedade) e `idade_dias_entrada` (idade exata difícil de obter). Fora dos preditores: `id_animal`, `id_propriedade`, `data_entrada`, `data_saida` (identificação) + `peso_saida_kg` (leakage direto) + o alvo.
- **`dias_permanencia` é preditor legítimo** aqui (diferente do antigo `days_on_pasture`, que era tratado como leakage).
- **Derivadas por script (nunca digitadas):** `dias_permanencia`, `gmd_kg_dia`, `temperatura_media_c` e `precipitacao_acumulada_mm` (NASA POWER, sobre a janela entrada→saída), `proporcao_ciclo_seca`.
- **vs. schema antigo:** *removidas* **todo o eixo sanitário** (`days_since_health_event`, `health_event_duration_days`, `recent_vaccination`, `recent_deworming`) — coleta inviável/não-confiável (registros manuscritos e assistemáticos; registrar como limitação) — e `energia_metabolizavel_suplemento` (r=1.0 com PB); *reformuladas* `days_on_pasture`→`dias_permanencia`, `transport_stress`(bin)→`numero_eventos_transporte`(contagem), clima/suplemento/forragem viram médias **ponderadas por dias**; *adicionadas* ids, datas, `proporcao_ciclo_seca` (substitui `estacao_ano`). `media_taxa_lotacao_ua_ha` foi cogitada mas **removida** (coleta não-confiável).
- **Sem eixo sanitário no v3:** a v2 chegou a incluir `numero_eventos_sanitarios`/`total_dias_afetados_sanidade`, mas foram **descartados** — o eixo sanitário saiu por inteiro.

**Regras metodológicas obrigatórias:**
1. `peso_saida_kg` nunca entra nos preditores (vazamento do alvo).
2. **Split agrupado por `id_propriedade`** (`GroupShuffleSplit`/`GroupKFold`) — validade externa.
3. Augmentation (se houver) **só APÓS o split**, no treino. Técnica definida: **jitter = ruído de medição** nos contínuos (pesos ±2–3 kg, clima) rederivando `gmd_kg_dia`; implementar junto com a reescrita do `data_prep.py` v3 (avaliar sempre em dado real, com/sem augmentation).
4. `StandardScaler` ajustado só no treino.

**⚠️ Split com poucas propriedades:** o dataset tem **2 fazendas** (Elvis/Humberto, pós-fusão). Desenho adotado no `data_prep`: **5 folds disjuntos de 49 animais de teste (estratificados pela % de cada fazenda; todo animal testado 1x) para seleção/tuning** + **leave-one-property-out** (agora só **2 splits**: treina Elvis/testa Humberto e vice-versa) como **teste de robustez / validade externa**. Enquadramento confirmado com o Prof. Ciniro (reunião 2026-09-22): manter comparação DNN × baselines.

### Fusão Sonico→Elvis (2026-09-22)

Decisão de reunião com o orientador: o Sonico (61 animais) tinha LOPO catastrófico
(R² ≈ −266) e os animais **passaram grande parte da vida na propriedade do Elvis**
antes de ir para o Sonico — fundir é mais fiel à origem real do que manter como
3ª fazenda. Fusão = reclassificar `id_propriedade` sonico→elvis e trocar as colunas
**tied à propriedade** pelos valores do Elvis, mantendo as colunas **do próprio
animal** (peso, dias, GMD, suplemento por peso) como estavam:
- `pb_suplemento_pct`: 25/0 → **25** (decisão do orientador; Elvis tem 3 valores
  30/25/20 que se sobrepõem no tempo — sem resposta única, então fixou-se 25).
- `media_proteina_bruta_forragem_pct`: 9,5 → **10,42** (constante única do Elvis).
- `temperatura_media_c`/`precipitacao_acumulada_mm`/`proporcao_ciclo_seca`:
  **recalculados via NASA POWER** com as coordenadas do Elvis (−20,0072/−46,0748),
  sobre a mesma janela real (`data_entrada`→`data_saida`) de cada animal.
- Mantidos: `peso_entrada_kg`, `peso_saida_kg`, `dias_permanencia`, `gmd_kg_dia`,
  `media_suplemento_kg_dia`, `sexo_macho`, `proporcao_bos_indicus_pct`,
  `numero_eventos_transporte` — são do próprio animal, não da propriedade.
- Dataset final: **245 animais, 2 fazendas (Elvis 174, Humberto 71)**.
  Backup pré-fusão fora do repo (`/tmp/.../dataset_backup_pre_merge.csv`, sessão local).
- `data_prep.py` **não mudou** — `FarmStratifiedSplit`/`leave_one_property_out` já
  eram genéricos por nº de grupos; `leave_one_property_out` passou de 3 para 2 splits.

**Pendências do novo schema:** verificar variância de `sexo_macho`, `rotacao_piquete`, `frequencia_suplementacao_dias_semana` (constantes antes → se constantes, remover e reportar como condições controladas); baselines faltando (RF, XGBoost, linear); implementar importância por SHAP. *(Resolvida: `media_digestibilidade_forragem_pct` removida do schema pela correlação ~0.97 com PB.)*

## Ambiente

- Python 3.11; PyTorch 2.10 + CUDA 13; NumPy, Pandas, Scikit-learn, Matplotlib/Seaborn, pygame, torchinfo.
- Setup: `conda create -n cattle_env python=3.11 && pip install -r requirements.txt`.

---

## Onde paramos

**Última atualização:** 2026-09-22

**Tópico atual:** **Coleta fechada, reunião com orientador (2026-09-22).** `dataset_por_animal_modelo_v3.csv` tem **245 animais, 2 fazendas**: **Elvis 174** (113 originais + 61 fundidos do Sonico — ver "Fusão Sonico→Elvis"; forragem Decumbens/MG4/Tanzânia/Marandu, PB 10,42; suplemento proteinado 30/25/20% ou sal mineral) e **Humberto 71** (mix Brachiaria/Cynodon, PB 10,0; via Excel `Pesagem Gado`, aba "Analise 1 pes ate ultima" = 1ª→última pesagem; `proporcao_bos_indicus_pct` **varia** por raça da coluna descrição — Nelore/Guzerá 100, Angus 50, resto 75; brinco 150 colidiu → id `150H`). Coords próprias por fazenda. Todas fêmeas; suplemento por lote (proteinado 30/25% a 0,3% do peso; **sal mineral** PB 0/fixo 0,10 kg/dia); transporte 1/2 conforme origem; clima do NASA POWER por janela (coords da respectiva fazenda). Ingestão via `ingest_sheet.py`, transcrição **por partes** com checagem de consistência (ganho=saída−entrada, gmd=ganho/dias). Correções aplicadas: livro mestre reconciliado (conflitos 245/356/360/365/366/343 sobrescritos), brinco 426 corrigido, 61 animais reatribuídos Elvis→Sonico (histórico, pré-fusão). Fora: 335/350/172(morreu)/187/177 sem saída. ids especiais: `001`=S/BRINCO, `002`=2º animal com brinco 336, `044` com zero à esquerda. EDA e DNN **reexecutadas no dataset final** (245/2 fazendas, pós-fusão).

### Abordagem definida
- Modelos (decidido em reunião 2026-09-22): (1) Regressão Linear, (2) XGBoost ou Random Forest, (3) **TabPFN**, (4) MLP/DNN — **a DNN está congelada** (config final: `(128,64,32,16)`, dropout 0, lr 1e-2, BN=True, wd 1e-4; não mexer em arquitetura/hiperparâmetro salvo se o dataset mudar de novo).
- Pré-processamento **compartilhado** em `data_prep.py` → comparação justa; resultados em `results/model_comparison.csv`.
- **Atenção:** as "Descobertas dos experimentos" abaixo são do **schema antigo + dados sintéticos** (`cattle_dataset_2.csv`, já removido) — **obsoletas**, referência histórica só. `data_prep.py` já é v3; os números reais virão dos novos notebooks sobre o dataset por-animal.

### Descobertas dos experimentos (CV 10-fold, no `cattle_dataset_2.csv`, 494 — inclui sintéticas — OBSOLETO)
- Sinal existe e é **não-linear**: praticamente sem ruído irredutível (alvo ~determinístico dado X).
- Regressão Linear R²≈0.03 · **DNN otimizada R²≈0.16 (MAE 0.088)** · **Random Forest R²≈0.38 (MAE 0.082)**.
- Melhorias da DNN que ajudaram: **BatchNorm + (128,64,32)** (maior ganho), **Huber loss**, **Yeo-Johnson** no alvo, menos regularização, LR scheduler. **Não** ajudaram: redes maiores (256…), interações/polinômios; razões "domain" só marginalmente (dentro do ruído).
- **Curva de aprendizado (DNN×RF):** RF domina em todas as faixas (n=80→395) e ainda sobe; DNN sobe devagar e com alta variância. Ambos limitados por dados (mais amostras ajudam), mas a DNN não alcança a RF no alcance observável.

### Decisão
Manter a DNN como objeto de estudo do TCC e enquadrar as árvores como baseline comparativo — a comparação, com a curva de aprendizado justificando o porquê, é a contribuição. Manter **uma única base de features (as cruas)** para a comparação ser justa. Confirmar o enquadramento do tema com o Prof. Ciniro.

### Próximo passo
- ✅ **Feito:** EDA (`eda_gmd.ipynb`) e `data_prep.py` v3 (verificado: 9 preditores, `get_cv` = 5 folds×49 por fazenda, `leave_one_property_out`, `augment_train` jitter) — reexecutados no dataset final pós-fusão (245/2 fazendas).
- ✅ **Fusão Sonico→Elvis aplicada** (ver seção própria) — dataset final **245 animais, Elvis 174 / Humberto 71**.
- ✅ **DNN CONGELADA — resultado final** (`dnn_gmd.ipynb`, dataset pós-fusão, 245/2 fazendas, 9 preditores, `get_cv` 5×49 por fazenda, grid 72 configs): melhor **(128,64,32,16), dropout 0, lr 1e-2, BN=True, wd 1e-4**. **R² 0,843 (single) → 0,820 (ensemble 5 seeds — piorou dessa vez, não ajuda aqui), MAE 0,028**, R² std entre folds 0,12; baseline MAE 0,079. Jitter (1 seed): R² 0,861 — não tratar como ganho sem repetir. **LOPO (agora só 2 splits):** testa Humberto R² **0,353** (melhor que qualquer LOPO anterior), testa Elvis R² **−49,3** (quebra — Elvis pós-fusão é maior/mais heterogêneo, difícil de prever a partir só do Humberto). Salvo em `results/model_comparison.csv` (`cv_R2=0,820`, `lopo_R2=−24,49` = média dos 2 splits). ⚠️ Números **não comparáveis** aos de antes da fusão (3 fazendas) nem ao CV 10-fold antigo.
- ⚠️ **Jitter — lição importante (mantém-se):** a augmentation que **rederiva o alvo** a partir do `peso_entrada` acopla feature↔alvo e **infla o R² artificialmente** — por isso `data_prep.augment_train` é **feature-only** (mantém GMD real). Ganho do jitter feature-only é pequeno/dentro do ruído entre folds; não repetir o diagnóstico salvo se o dataset mudar.
- **Próximo (reunião 2026-09-22):** construir os baselines que faltam — **Regressão Linear, XGBoost ou RF, e TabPFN** — no `get_cv()` atual (5×49 por fazenda), pra comparar de verdade com a DNN congelada. Depois: **CV agrupado por lote** (cohort = propriedade+data_entrada+data_saida, testa se o R² da DNN é inflado por vazamento) e **SHAP** no melhor modelo (cuidado com a multicolinearidade `media_suplemento`×`pb_suplemento`, r=0,93).

---

> **Nota de trabalho:** manter esta seção "Onde paramos" atualizada ao final de cada sessão (data, tópico atual, próximo passo) e retomá-la no início da próxima.
