# dnn_cattle_weight_gain

## Visão geral

Rede neural densa (DNN/MLP em **PyTorch**) que prevê o **Ganho Médio Diário (GMD, `adg_kg_day`)** de bovinos de corte. Projeto acadêmico de TCC — IFMG, Depto. de Engenharia e Computação (Prof. Ciniro Nametala; aluno Euler Gomes). Modelo `GMDNN`, ainda em desenvolvimento.

A DNN é o **modelo central do TCC**, mas o trabalho evoluiu para um **estudo comparativo**: a rede é avaliada contra baselines (regressão linear) e ensembles de árvores (Random Forest, Gradient Boosting), sobre os mesmos dados/split/CV. A comparação — com curva de aprendizado justificando quando cada família vence — é parte da contribuição. Ver "Onde paramos".

> **Escopo desta fase: apenas código e dados.** A redação do TCC (LaTeX, classe abntex2) fica para o fim do projeto, com o modelo pronto. **Não** produzir, editar ou sugerir texto do trabalho.

## Estrutura

> Repositório enxuto: o schema antigo (notebooks, datasets `cattle_*`, modelos `.pth`, `processar_formulario.py`) foi **removido** — recuperável pelo histórico do git se necessário.

- `data/dataset_por_animal_modelo_v3.csv` — **o dataset** (schema por-animal, alvo `gmd_kg_dia`). Detalhes em "Nova versão do dataset (por animal)".
- `support_scripts/ingest_sheet.py` — **ingestão de folha de campo manuscrita → v3**: recebe as linhas transcritas + constantes da fazenda, valida consistência (ganho=saída−entrada, gmd=ganho/dias), deriva clima (NASA POWER) e anexa. `import ingest_sheet as ing; ing.ingest(animais, fazenda, dry_run=True)`.
- `support_scripts/data_prep.py` — módulo de pré-processamento compartilhado dos notebooks de comparação (split, KFold(10), seleção de features, métricas MAE/RMSE/R², `save_result`). **⚠️ Ainda no schema antigo — reescrever para o v3** (alvo `gmd_kg_dia`, split das 2 propriedades). Uso: `sys.path.append("support_scripts"); import data_prep as dp`.
- `eda_gmd.ipynb` — **EDA** do dataset v3 (visão geral, alvo, variância/constantes, correlações, multicolinearidade, GMD por suplemento). Kernel conda `tcc`.
- **A construir:** notebooks de comparação (Regressão Linear → RF → Gradient Boosting/XGBoost → MLP/DNN), gravando em `results/model_comparison.csv`.

## Dados

- Coletados em **2 fazendas** da região de **Bambuí-MG**: **Elvis** (94 animais; −20,0072/−46,0748) e **Sonico** (61; −19,9949/−45,9234).
- Schema e regras em "Nova versão do dataset (por animal)".

## Nova versão do dataset (por animal — EM COLETA)

> `data/dataset_por_animal_modelo_v3.csv` (**19 colunas**) já tem a **1ª propriedade preenchida** (13 animais da fazenda Elvis); falta a 2ª fazenda. **Substitui o schema antigo** e torna obsoletos todos os números medidos antes (schema antigo + dados com sintéticas).

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

**⚠️ Restrição prática (2 propriedades):** o dataset terá só **2 fazendas**. Isso inviabiliza o split agrupado por propriedade como CV de vários folds (GroupKFold ≤ nº de grupos = 2). Desenho provável: **CV normal (não-agrupado) para seleção/tuning** + **leave-one-property-out** (treina fazenda A / testa B e vice-versa) só como **teste de robustez / validade externa**. Decidir com o Prof. Ciniro.

**Pendências do novo schema:** verificar variância de `sexo_macho`, `rotacao_piquete`, `frequencia_suplementacao_dias_semana` (constantes antes → se constantes, remover e reportar como condições controladas); baselines faltando (RF, XGBoost, linear); implementar importância por SHAP. *(Resolvida: `media_digestibilidade_forragem_pct` removida do schema pela correlação ~0.97 com PB.)*

## Ambiente

- Python 3.11; PyTorch 2.10 + CUDA 13; NumPy, Pandas, Scikit-learn, Matplotlib/Seaborn, pygame, torchinfo.
- Setup: `conda create -n cattle_env python=3.11 && pip install -r requirements.txt`.

---

## Onde paramos

**Última atualização:** 2026-08-20

**Tópico atual:** **Coleta concluída — 2 propriedades.** `dataset_por_animal_modelo_v3.csv` tem **155 animais**: **Elvis 94** (forragem Decumbens/MG4/Tanzânia/Marandu, PB 10,42; coords −20,0072/−46,0748) e **Sonico 61** (forragem Decumbens+Ruziziensis, PB 9,5 pesquisado; coords −19,9949/−45,9234). Todas fêmeas aneloradas; suplemento por lote (proteinado 30/25% a 0,3% do peso; **sal mineral** PB 0/fixo 0,10 kg/dia); transporte 1/2 conforme origem; clima do NASA POWER por janela (coords da respectiva fazenda). Ingestão via `ingest_sheet.py`, transcrição **por partes** com checagem de consistência (ganho=saída−entrada, gmd=ganho/dias). Correções aplicadas: livro mestre reconciliado (conflitos 245/356/360/365/366/343 sobrescritos), brinco 426 corrigido, **61 animais reatribuídos Elvis→Sonico** (PB e clima recalculados). Fora: 335/350/172(morreu)/187/177 sem saída. ids especiais: `001`=S/BRINCO, `002`=2º animal com brinco 336, `044` com zero à esquerda. **EDA preliminar** em `eda_gmd.ipynb` (precisa reexecutar/atualizar p/ 2 propriedades). **Modelagem destravada.**

### Abordagem definida
- Modelos (do mais simples ao mais complexo): (1) Regressão Linear, (2) Random Forest, (3) Gradient Boosting/XGBoost, (4) MLP/DNN. TabPFN/transformer **fora por ora** (só compensaria com N ordens de grandeza maior).
- Pré-processamento **compartilhado** em `data_prep.py` → comparação justa; resultados em `results/model_comparison.csv`.
- **Atenção:** `data_prep.py` aponta para `data/cattle_dataset.csv` (247, **amostras reais**) — a base oficial. Os experimentos exploratórios abaixo foram no `cattle_dataset_2.csv` (494, **com sintéticas**), então **os números NÃO refletem o dado real** e devem ser refeitos no dataset de 247.

### Descobertas dos experimentos (CV 10-fold, no `cattle_dataset_2.csv`, 494 — inclui sintéticas)
- Sinal existe e é **não-linear**: praticamente sem ruído irredutível (alvo ~determinístico dado X).
- Regressão Linear R²≈0.03 · **DNN otimizada R²≈0.16 (MAE 0.088)** · **Random Forest R²≈0.38 (MAE 0.082)**.
- Melhorias da DNN que ajudaram: **BatchNorm + (128,64,32)** (maior ganho), **Huber loss**, **Yeo-Johnson** no alvo, menos regularização, LR scheduler. **Não** ajudaram: redes maiores (256…), interações/polinômios; razões "domain" só marginalmente (dentro do ruído).
- **Curva de aprendizado (DNN×RF):** RF domina em todas as faixas (n=80→395) e ainda sobe; DNN sobe devagar e com alta variância. Ambos limitados por dados (mais amostras ajudam), mas a DNN não alcança a RF no alcance observável.

### Decisão
Manter a DNN como objeto de estudo do TCC e enquadrar as árvores como baseline comparativo — a comparação, com a curva de aprendizado justificando o porquê, é a contribuição. Manter **uma única base de features (as cruas)** para a comparação ser justa. Confirmar o enquadramento do tema com o Prof. Ciniro.

### Próximo passo (quando a coleta chegar)
1. Reescrever `support_scripts/data_prep.py` para o schema por-animal: alvo `gmd_kg_dia`, remover ids + `peso_saida_kg`, e resolver o desenho do split dado só **2 propriedades** (CV normal p/ tuning + leave-one-property-out p/ robustez).
2. Refazer a EDA no novo dataset (variância das binárias, correlação PB×digestibilidade).
3. Construir os notebooks de comparação **por partes**: Regressão Linear → RF → Gradient Boosting/XGBoost → MLP/DNN (via `data_prep` + `save_result`), e SHAP ao final.

---

> **Nota de trabalho:** manter esta seção "Onde paramos" atualizada ao final de cada sessão (data, tópico atual, próximo passo) e retomá-la no início da próxima.
