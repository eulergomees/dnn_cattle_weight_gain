# Cattle Average Daily Gain (GMD) Prediction — Comparative Study

Predicts **Average Daily Gain (GMD, `gmd_kg_dia`, kg/day)** of beef cattle from field data. What started as a single PyTorch MLP (`GMDNN`) grew into a **comparative study**: the neural network against a linear baseline, XGBoost, and [TabPFN](https://github.com/PriorLabs/TabPFN) (a pretrained tabular foundation model), all evaluated on the same data, split and cross-validation scheme.

> Academic project (TCC) — Instituto Federal de Minas Gerais, Campus Bambuí, Depto. de Engenharia e Computação
> **Advisor:** Ciniro Nametala | **Student:** Euler Gomes

---

## Dataset

- `data/dataset_por_animal_modelo_v3.csv` — **245 animals, 2 farms** (Bambuí-MG region), one row per animal (entry weighing → exit weighing), not per individual weigh-in.
- **Target:** `gmd_kg_dia` — average daily gain over the whole cycle. `peso_saida_kg` (exit weight) is never a predictor (it leaks the target); derive it back with `peso_saida = peso_entrada + gmd_predito × dias_permanencia` if needed.
- **9 effective predictors** (constants and leakage/multicollinear columns dropped automatically by `data_prep.feature_columns`): `peso_entrada_kg`, `dias_permanencia`, `media_suplemento_kg_dia`, `pb_suplemento_pct`, `media_proteina_bruta_forragem_pct`, `numero_eventos_transporte`, `temperatura_media_c`, `proporcao_ciclo_seca`, `proporcao_bos_indicus_pct`.
- Climate features are derived from the [NASA POWER](https://power.larc.nasa.gov/) API over each animal's entry→exit window.
- Two of the original three source farms were merged (see `CLAUDE.md` for the full rationale and data-provenance notes) — this is a decision log, not a data dictionary; read `data_prep.py`'s docstring for the authoritative schema.

---

## Repository layout

| Path | What it does |
|---|---|
| `support_scripts/data_prep.py` | Shared preprocessing: feature selection, `get_cv()` (stratified-by-farm CV), `leave_one_property_out()`, jitter augmentation, metrics, `save_result()`. Every model notebook imports this — same data/split/metrics for a fair comparison. |
| `support_scripts/sk_eval.py` | CV/LOPO/jitter evaluation helpers for scikit-learn-style models (`.fit`/`.predict`), shared by the baseline notebooks. |
| `support_scripts/ingest_sheet.py` | Ingests a transcribed handwritten field sheet into the dataset: validates consistency (gain = exit − entry, GMD = gain / days) and derives climate. |
| `eda_gmd.ipynb` | Exploratory analysis: target distribution, variance/constants, correlations, multicollinearity. |
| `dnn_gmd.ipynb` | The `GMDNN` model (PyTorch MLP): grid search, best config with/without jitter, seed ensemble, leave-one-property-out, predicted×actual plot. |
| `baselines/` | `01_linear`, `02_xgboost` (own grid search), `03_tabpfn` — same `data_prep`/`sk_eval` pipeline as the DNN. |
| `cv_agrupado_por_lote.ipynb` | Validity check: re-evaluates all 4 models under cohort-grouped CV (no batch split across train/test) to test for leakage in the main CV scheme. |
| `shap_tabpfn.ipynb` | SHAP interpretation of the best-performing model. |
| `article/` | LaTeX source for a conference paper derived from this work (separate from the TCC monograph). |
| `results/model_comparison.csv` | Comparison table, written to by every model notebook. |

---

## Results (current)

CV = 5 disjoint folds of 49 held-out animals, stratified by farm. LOPO = leave-one-property-out (train on one farm, test on the other, both directions averaged).

| Model | CV R² | CV MAE | LOPO R² |
|---|---:|---:|---:|
| **TabPFN** | **0.863** | **0.0145** | −14.64 |
| MLP (`GMDNN`) | 0.820 | 0.0285 | −24.49 |
| XGBoost | 0.508 | 0.0512 | **−0.22** |
| Linear Regression | 0.426 | 0.0629 | −36.28 |

TabPFN and the MLP lead within the training distribution; XGBoost generalizes best to a farm the model never saw. A cohort-grouped CV check (`cv_agrupado_por_lote.ipynb`) confirms the main CV scheme inflates R² for all four models via batch leakage, but the **ranking between models holds** under the stricter, grouped evaluation.

---

## Setup

```bash
git clone https://github.com/eulergomees/dnn_cattle_weight_gain.git
cd dnn_cattle_weight_gain

conda create -n tcc python=3.11
conda activate tcc

pip install -r requirements.txt
```

TabPFN downloads pretrained weights on first use and requires a one-time license acceptance (non-interactive environments need a `TABPFN_TOKEN` — see `.env.example`).

Open any notebook (`eda_gmd.ipynb`, `dnn_gmd.ipynb`, `baselines/*.ipynb`, ...) in Jupyter or your IDE and run all cells; each one is self-contained given `data/` and `support_scripts/`.

---

## Requirements

- Python 3.11
- PyTorch (CUDA recommended), torchinfo
- NumPy, Pandas, Scikit-learn, XGBoost, TabPFN
- Matplotlib, Seaborn, SHAP
- python-dotenv

---

## Authors

- [@eulergomees](https://github.com/eulergomees) — Euler Gomes
