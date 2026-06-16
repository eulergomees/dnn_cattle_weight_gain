# DNN for Cattle Average Daily Gain (GMD) Prediction

A Dense Neural Network that predicts and classifies the **Average Daily Gain (GMD, kg/day)** of beef cattle, built with PyTorch.

> Academic project — INSTITUTO FEDERAL DE MINAS GERAIS, Departamento de Engenharia e Computação  
> **Professor:** Ciniro Nametala | **Student:** Euler Gomes

---

## Dataset

- **Samples** collected from two farms in the Bambuí-MG region
- **19 input features** after One-Hot Encoding, including animal attributes, pasture type, supplementation, health events, and climate data
- **Target:** `saida_gmd_kg_dia` — daily weight gain in kg/day

Raw data lives in `data/cattle_dataset.csv`.

---

## Model

`GMDNN` — in progress...

---

## Data Processing Script

`support_scripts/processar_formulario.py` converts a Google Forms CSV export into a model-ready dataset.

**Automatic transformations:**
- Birth date + weighing date → `idade_dias`
- Breed → `proporcao_bos_indicus_pct`
- Forage species + season → `proteina_bruta_forragem_pct`, `digestibilidade_forragem_pct`
- Entry date + weighing date → `dias_permanecia`
- NASA POWER API (evaluation period) → `temperatura_media_c`, `precipitacao_acumulada_mm`
- Health event date → `dias_desde_evento_sanitario`
- Exit weight − initial weight / days → `saida_gmd_kg_dia`

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
