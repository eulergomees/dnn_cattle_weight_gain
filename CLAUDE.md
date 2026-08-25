# dnn_cattle_weight_gain

## Visão geral

Rede neural densa (DNN/MLP em **PyTorch**) que prevê o **Ganho Médio Diário (GMD, `gmd_kg_dia`)** de bovinos de corte. Projeto acadêmico de TCC — IFMG, Depto. de Engenharia e Computação (Prof. Ciniro Nametala; aluno Euler Gomes). Modelo `GMDNN`, ainda em desenvolvimento.

A DNN é o **modelo central do TCC**, mas o trabalho evoluiu para um **estudo comparativo**: a rede é avaliada contra baselines (regressão linear) e ensembles de árvores (Random Forest, Gradient Boosting), sobre os mesmos dados/split/CV. A comparação — com curva de aprendizado justificando quando cada família vence — é parte da contribuição. Ver "Onde paramos".

> **Escopo desta fase: apenas código e dados.** A redação do TCC (LaTeX, classe abntex2) fica para o fim do projeto, com o modelo pronto. **Não** produzir, editar ou sugerir texto do trabalho.

## Estrutura

> Repositório enxuto: o schema antigo (notebooks, datasets `cattle_*`, modelos `.pth`, `processar_formulario.py`) foi **removido** — recuperável pelo histórico do git se necessário.

- `data/dataset_por_animal_modelo_v3.csv` — **o dataset** (schema por-animal, alvo `gmd_kg_dia`). Detalhes em "Nova versão do dataset (por animal)".
- `support_scripts/ingest_sheet.py` — **ingestão de folha de campo manuscrita → v3**: recebe as linhas transcritas + constantes da fazenda, valida consistência (ganho=saída−entrada, gmd=ganho/dias), deriva clima (NASA POWER) e anexa. `import ingest_sheet as ing; ing.ingest(animais, fazenda, dry_run=True)`.
- `support_scripts/data_prep.py` — módulo de pré-processamento compartilhado (**schema v3**): `get_data` (8 preditores, alvo `gmd_kg_dia`; dropa ids + `peso_saida_kg` + 4 constantes + `precipitacao_acumulada_mm`), `get_cv` (KFold 10 p/ tuning), `leave_one_property_out` (Elvis↔Sonico p/ robustez), `augment_train` (jitter = ruído de medição, só treino), métricas e `save_result`. Scaling fica nos notebooks (fit só no treino). Uso: `sys.path.append("support_scripts"); import data_prep as dp; d = dp.get_data()`.
- `eda_gmd.ipynb` — **EDA** do dataset v3 (visão geral, alvo, variância/constantes, correlações, multicolinearidade, GMD por suplemento). Kernel conda `tcc`.
- `dnn_gmd.ipynb` — **rede neural `GMDNN`** (MLP PyTorch): grid search (CV 10-fold), melhor config com/sem jitter, leave-one-property-out, predito×real OOF, grava em `results/model_comparison.csv`.
- **A construir:** notebooks de comparação Regressão Linear → RF → Gradient Boosting/XGBoost (mesmo esquema, via `data_prep`), e SHAP.

## Dados

- Coletados em **3 fazendas** da região de **Bambuí-MG**: **Elvis** (113; −20,0072/−46,0748), **Sonico** (61; −19,9949/−45,9234) e **Humberto** (71; −20,0097/−45,9581, via Excel de pesagens).
- Schema e regras em "Nova versão do dataset (por animal)".

## Nova versão do dataset (por animal — EM COLETA)

> `data/dataset_por_animal_modelo_v3.csv` (**19 colunas**) tem **245 animais em 3 fazendas** (Elvis 113, Humberto 71, Sonico 61). **Substitui o schema antigo** e torna obsoletos todos os números medidos antes (schema antigo + dados com sintéticas). Detalhes de coleta/constantes em "Onde paramos".

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

**⚠️ Split com poucas propriedades:** o dataset tem **3 fazendas** (Elvis/Humberto/Sonico). Desenho adotado no `data_prep`: **CV 10-fold não-agrupado para seleção/tuning** + **leave-one-property-out** (treina 2 fazendas / testa a 3ª, rodando as 3) como **teste de robustez / validade externa**. Confirmar enquadramento com o Prof. Ciniro.

**Pendências do novo schema:** verificar variância de `sexo_macho`, `rotacao_piquete`, `frequencia_suplementacao_dias_semana` (constantes antes → se constantes, remover e reportar como condições controladas); baselines faltando (RF, XGBoost, linear); implementar importância por SHAP. *(Resolvida: `media_digestibilidade_forragem_pct` removida do schema pela correlação ~0.97 com PB.)*

## Ambiente

- Python 3.11; PyTorch 2.10 + CUDA 13; NumPy, Pandas, Scikit-learn, Matplotlib/Seaborn, pygame, torchinfo.
- Setup: `conda create -n cattle_env python=3.11 && pip install -r requirements.txt`.

---

## Onde paramos

**Última atualização:** 2026-08-20

**Tópico atual:** **Coleta em 3 propriedades.** `dataset_por_animal_modelo_v3.csv` tem **245 animais**: **Elvis 113** (forragem Decumbens/MG4/Tanzânia/Marandu, PB 10,42; suplemento proteinado 30/25/20% ou sal mineral), **Sonico 61** (Decumbens+Ruziziensis, PB 9,5) e **Humberto 71** (mix Brachiaria/Cynodon, PB 10,0; via Excel `Pesagem Gado`, aba "Analise 1 pes ate ultima" = 1ª→última pesagem; `proporcao_bos_indicus_pct` **varia** por raça da coluna descrição — Nelore/Guzerá 100, Angus 50, resto 75; brinco 150 colidiu → id `150H`). Coords próprias por fazenda. Todas fêmeas; suplemento por lote (proteinado 30/25% a 0,3% do peso; **sal mineral** PB 0/fixo 0,10 kg/dia); transporte 1/2 conforme origem; clima do NASA POWER por janela (coords da respectiva fazenda). Ingestão via `ingest_sheet.py`, transcrição **por partes** com checagem de consistência (ganho=saída−entrada, gmd=ganho/dias). Correções aplicadas: livro mestre reconciliado (conflitos 245/356/360/365/366/343 sobrescritos), brinco 426 corrigido, **61 animais reatribuídos Elvis→Sonico** (PB e clima recalculados). Fora: 335/350/172(morreu)/187/177 sem saída. ids especiais: `001`=S/BRINCO, `002`=2º animal com brinco 336, `044` com zero à esquerda. **⚠️ EDA (`eda_gmd.ipynb`) e DNN (`dnn_gmd.ipynb`, R²~0,665) foram feitos no dataset de 155 (2 propriedades) — precisam REEXECUTAR no de 226 / 3 propriedades** (mais dados = provável ganho na rede; `data_prep` já lida com 3 grupos, sem mudança).

### Abordagem definida
- Modelos (do mais simples ao mais complexo): (1) Regressão Linear, (2) Random Forest, (3) Gradient Boosting/XGBoost, (4) MLP/DNN. TabPFN/transformer **fora por ora** (só compensaria com N ordens de grandeza maior).
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
- ✅ **Feito:** EDA (`eda_gmd.ipynb`) e reescrita do `data_prep.py` p/ v3 (verificado: 8 preditores, `get_cv` KFold(10) + `leave_one_property_out`, `augment_train` jitter).
- **Smoke test (RegLinear):** CV 10-fold já dá **R²~0.37** (sinal linear real, vs ~0.03 no antigo sintético); **LOPO catastrófico p/ o linear** (R² muito negativo) — as 2 fazendas são bem diferentes (Sonico: nascidos leves 80–107 kg, ciclos longos até 1253 d, sal mineral); esperar RF melhor no LOPO (não extrapola).
- ✅ **DNN** (`dnn_gmd.ipynb`) reexecutada nos **245 animais / 3 fazendas** (9 preditores — `proporcao_bos_indicus_pct` virou variável com as raças da Humberto): grid search → melhor **(128,64,32), dropout 0, lr 1e-2, BN=True, wd 0**. **CV 10-fold: R² 0,805 (single) → 0,838 com ensemble de 5 seeds (modelo final), MAE 0,027**; baseline MAE 0,079. Subiu de 0,665 (155) — mais dados levantou o teto; parte do ganho de R² é o alvo ter mais variância (MAE é o comparável). Jitter feature-only não ajuda; ensemble de seeds ajuda de forma honesta. **LOPO melhorou:** testa Humberto R²+0,12, Elvis −0,12, mas **Sonico −154** (outlier: nascidos leves/ciclos longos/sal mineral).
- ⚠️ **Jitter — lição importante:** a augmentation que **rederiva o alvo** a partir do `peso_entrada` (feature que entra em `gmd=(peso_saida−peso_entrada)/dias`) **acopla feature↔alvo e infla o R² artificialmente** (diagnóstico: subia 0,67→0,81 só aumentando o ruído). Trocado por **jitter feature-only** (perturba features, mantém GMD real) em `data_prep.augment_train` — e assim o jitter **não ajuda** (0,665→0,61). Ou seja: o número honesto da DNN é **~0,665 sem jitter**.
- **Agora:** construir os baselines de comparação — Regressão Linear → RF → Gradient Boosting/XGBoost (mesmo esquema `data_prep` + `save_result`, com/sem jitter) e **SHAP**. Ver como a RF se sai no LOPO (não extrapola). Tratar multicolinearidade `media_suplemento`×`pb_suplemento` (0,93) na leitura do SHAP.

---

> **Nota de trabalho:** manter esta seção "Onde paramos" atualizada ao final de cada sessão (data, tópico atual, próximo passo) e retomá-la no início da próxima.
