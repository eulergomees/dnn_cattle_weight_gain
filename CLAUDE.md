# dnn_cattle_weight_gain

## Visão geral

Rede neural densa (DNN/MLP em **PyTorch**) que prevê o **Ganho Médio Diário (GMD, `adg_kg_day`)** de bovinos de corte. Projeto acadêmico de TCC — IFMG, Depto. de Engenharia e Computação (Prof. Ciniro Nametala; aluno Euler Gomes). Modelo `GMDNN`, ainda em desenvolvimento.

A DNN é o **modelo central do TCC**, mas o trabalho evoluiu para um **estudo comparativo**: a rede é avaliada contra baselines (regressão linear) e ensembles de árvores (Random Forest, Gradient Boosting), sobre os mesmos dados/split/CV. A comparação — com curva de aprendizado justificando quando cada família vence — é parte da contribuição. Ver "Onde paramos".

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
- `support_scripts/processar_formulario.py` — converte export do Google Forms → dataset pronto (idade, raça→% Bos indicus, forrageira→PB/digestibilidade, clima via NASA POWER API p/ Bambuí-MG, GMD). Uso: `python support_scripts/processar_formulario.py dados.csv [--sem-clima]`.
- `results/model_comparison.csv` — tabela agregada dos modelos (gerada pelos notebooks via `data_prep.save_result`).
- `slides_tcc/` — apresentação LaTeX/Beamer do TCC. `monitoring.ods` — planilha de acompanhamento.

## Dados

- Coletados em duas fazendas na região de **Bambuí-MG**.
- Features de `cattle_dataset.csv` (ordem das colunas): `initial_weight_kg, age_days, sex_male, bos_indicus_proportion_pct, supplement_amount_kg_day, supplement_crude_protein_pct, supplement_metabolizable_energy, supplementation_frequency_days_week, forage_crude_protein_pct, forage_digestibility_pct, days_on_pasture, paddock_rotation, transport_stress, mean_temperature_c, accumulated_rainfall_mm, days_since_health_event, health_event_duration_days, recent_vaccination, recent_deworming` → alvo `adg_kg_day`.

## Ambiente

- Python 3.11; PyTorch 2.10 + CUDA 13; NumPy, Pandas, Scikit-learn, Matplotlib/Seaborn, pygame, torchinfo.
- Setup: `conda create -n cattle_env python=3.11 && pip install -r requirements.txt`.

---

## Onde paramos

**Última atualização:** 2026-07-18

**Tópico atual:** Montar um **estudo comparativo de modelos** para o GMD — um notebook por abordagem, sobre os mesmos dados/split/CV via `support_scripts/data_prep.py`. A DNN segue como modelo central do TCC; árvores entram como baseline forte.

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

### Próximo passo
Construir os notebooks de comparação **por partes**, começando pela **Regressão Linear** (usando `data_prep`, com CV 10-fold + avaliação no teste + `save_result`). Depois RF, Gradient Boosting e, por fim, o MLP/DNN.

---

> **Nota de trabalho:** manter esta seção "Onde paramos" atualizada ao final de cada sessão (data, tópico atual, próximo passo) e retomá-la no início da próxima.
