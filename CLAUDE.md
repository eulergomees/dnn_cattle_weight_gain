# dnn_cattle_weight_gain

## Visão geral

Rede neural densa (DNN/MLP em **PyTorch**) que prevê o **Ganho Médio Diário (GMD, `adg_kg_day`)** de bovinos de corte. Projeto acadêmico de TCC — IFMG, Depto. de Engenharia e Computação (Prof. Ciniro Nametala; aluno Euler Gomes). Modelo `GMDNN`, ainda em desenvolvimento.

A DNN é o **modelo central do TCC**, mas o trabalho evoluiu para um **estudo comparativo**: a rede é avaliada contra baselines (regressão linear) e ensembles de árvores (Random Forest, Gradient Boosting), sobre os mesmos dados/split/CV. A comparação — com curva de aprendizado justificando quando cada família vence — é parte da contribuição. Ver "Onde paramos".

> **Escopo desta fase: apenas código e dados.** A redação do TCC (LaTeX, classe abntex2) fica para o fim do projeto, com o modelo pronto. **Não** produzir, editar ou sugerir texto do trabalho.

## Estrutura

- `cattle_gmd_dnn.ipynb` — notebook **principal da DNN** (EDA + prep + CV 10-fold + modelo final + avaliação). Rodar todas as células (kernel conda `tcc`).
- `dnn_cattle_weight_gain.ipynb` — notebook antigo (schema em português, `cattle_gain.csv`); legado.
- **Notebooks de comparação** (um por modelo, em construção): Regressão Linear, Random Forest, Gradient Boosting/XGBoost, MLP/DNN. Todos importam o pré-processamento compartilhado e gravam em `results/model_comparison.csv`.
- `data/` — datasets:
  - `cattle_dataset.csv` (247 amostras) — **dataset principal, amostras REAIS** (coleta de campo), já processado, 20 colunas (19 features + alvo `adg_kg_day`). É a base oficial para treino/avaliação e para o estudo comparativo.
  - `cattle_dataset_2.csv` (494 amostras) — **contém amostras SINTÉTICAS** somadas às reais. Usar só para experimentos exploratórios; resultados não refletem desempenho no dado real.
  - `cattle_gain.csv` (203 linhas) — dados brutos em português (schema antigo).
- `models/` — pesos treinados: `model_adg_dnn.pth` (DNN atual, salva com stats do scaler e λ do Yeo-Johnson), `model_cattle_gain.pth`.
- `support_scripts/data_prep.py` — **módulo de pré-processamento compartilhado** dos notebooks de comparação: split hold-out fixo, KFold(10), seleção de features (dropa constantes + leakage + alvo), métricas MAE/RMSE/R², baseline e `save_result`. Constantes no topo (`DATA_PATH`, `SEED`, `N_SPLITS`, `TEST_SIZE`). Uso: `sys.path.append("support_scripts"); import data_prep as dp; d = dp.get_data()`.
- `support_scripts/ingest_sheet.py` — **ingestão de folha de campo manuscrita → v3**: recebe as linhas transcritas + constantes da fazenda, valida consistência (ganho=saída−entrada, gmd=ganho/dias), deriva clima (NASA POWER) e anexa. `import ingest_sheet as ing; ing.ingest(animais, fazenda, dry_run=True)`.
- `support_scripts/processar_formulario.py` — converte export do Google Forms → dataset pronto (idade, raça→% Bos indicus, forrageira→PB/digestibilidade, clima via NASA POWER API p/ Bambuí-MG, GMD). Uso: `python support_scripts/processar_formulario.py dados.csv [--sem-clima]`.
- `results/model_comparison.csv` — tabela agregada dos modelos (gerada pelos notebooks via `data_prep.save_result`).
- `slides_tcc/` — apresentação LaTeX/Beamer do TCC. `monitoring.ods` — planilha de acompanhamento.

## Dados

- Coletados em duas fazendas na região de **Bambuí-MG**.
- Features de `cattle_dataset.csv` (ordem das colunas): `initial_weight_kg, age_days, sex_male, bos_indicus_proportion_pct, supplement_amount_kg_day, supplement_crude_protein_pct, supplement_metabolizable_energy, supplementation_frequency_days_week, forage_crude_protein_pct, forage_digestibility_pct, days_on_pasture, paddock_rotation, transport_stress, mean_temperature_c, accumulated_rainfall_mm, days_since_health_event, health_event_duration_days, recent_vaccination, recent_deworming` → alvo `adg_kg_day`.

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

**Última atualização:** 2026-07-18

**Tópico atual:** **Coleta em andamento** no schema *por animal* (ver "Nova versão do dataset (por animal)"). **1ª propriedade (Elvis) — folhas processadas:** `dataset_por_animal_modelo_v3.csv` tem **126 animais da Elvis** (ingestão via `support_scripts/ingest_sheet.py`; livro mestre reconciliado em 3 lotes — conflitos 245/356/360/365/366 + 343 sobrescritos pelo mestre; constantes variam por origem: comprados `transporte`=2/`pb`=25, POSSES nascidos na propriedade `transporte`=1/`pb`=25, 70 primeiros `transporte`=1/`pb`=30; brinco 426 corrigido (saída 312→321); 335/350 sem saída (fora); 329/372 mantidos) (fêmeas aneloradas; suplemento 0,3% do peso médio; PB forragem 10,42; clima do NASA POWER por janela; ids baixos `001`=S/BRINCO, `002`=2º animal com brinco 336 repetido; `044` preservado com zero à esquerda). Transcrição **por partes** com checagem de consistência (ganho=saída−entrada, gmd=ganho/dias) e dias validados pelas datas. Convenção: `data_entrada = venda − dias` quando a compra manuscrita conflita com o `dias` (venda e dias são mais confiáveis). Falta a **2ª fazenda** — só então o split agrupado e a modelagem fazem sentido. Modelagem pausada até isso. Plano de fundo: **estudo comparativo** (DNN central + baselines) sobre o novo schema.

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
