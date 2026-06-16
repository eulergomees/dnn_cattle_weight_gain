# DNN for Cattle Average Daily Gain (GMD) Prediction

A Dense Neural Network that predicts and classifies the **Average Daily Gain (GMD, kg/day)** of beef cattle, built with PyTorch.

> Academic project — INSTITUTO FEDERAL DE MINAS GERAIS, Departamento de Engenharia e Computação  
> **Professor:** Ciniro Nametala | **Student:** Euler Gomes

---

## Dataset

- **Samples** collected from two farms in the Bambuí-MG region
- **19 input features** after One-Hot Encoding, including animal attributes, pasture type, supplementation, health events, and climate data
- **Target:** `adg_kg_day` — daily weight gain in kg/day

Raw data lives in `data/cattle_dataset.csv`.

---

## Model

`GMDNN` — in progress...

---

## Data Processing Script

`support_scripts/processar_formulario.py` converts a Google Forms CSV export into a model-ready dataset.

**Automatic transformations:**
- Birthdate + weighing date → `age_days`
- Breed → `bos_indicus_proportion_pct`
- Forage species + season → `forage_crude_protein_pct`, `forage_digestibility_pct`
- Entry date + weighing date → `days_on_pasture`
- NASA POWER API (evaluation period) → `mean_temperature_c`, `accumulated_rainfall_mm`
- Health event date → `days_since_health_event`
- Exit weight − initial weight / days → `adg_kg_day`

**Usage:**
```bash
python support_scripts/processar_formulario.py data.csv
python support_scripts/processar_formulario.py data.csv --sem-clima  # skip climate fetch
```

---

## Requirements

- Python 3.11
- PyTorch (CUDA 13 recommended)
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn
- pygame, torchinfo

---

## Setup

```bash
git clone https://github.com/eulergomees/dnn_cattle_weight_gain.git
cd dnn_cattle_weight_gain

conda create -n cattle_env python=3.11
conda activate cattle_env

pip install -r requirements.txt
```

Open `dnn_cattle_weight_gain.ipynb` in Jupyter or your IDE and run all cells.

---

## Authors

- [@eulergomees](https://github.com/eulergomees) — Euler Gomes
