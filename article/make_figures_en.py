"""Figures for the article (anonymized: farms A and B). Run from the repo root:
    python article/make_figures_en.py
Output: article/figs/fig_eda_en.pdf, article/figs/fig_corr_en.pdf
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "article/figs"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})
BLUE, ORANGE = "#0072B2", "#E69F00"          # Okabe-Ito palette
COL = {"A": BLUE, "B": ORANGE}

df = pd.read_csv("data/dataset_por_animal_modelo_v3.csv", dtype={"id_animal": str})
df["farm"] = df["id_propriedade"].map({"elvis": "A", "humberto": "B"})
df["entry_year"] = pd.to_datetime(df["data_entrada"]).dt.year

fig, axes = plt.subplots(2, 3, figsize=(4.8, 3.2))
ax = axes[0, 0]
years = sorted(df["entry_year"].unique()); x = np.arange(len(years)); w = 0.4
for k, f in enumerate(["A", "B"]):
    n = [((df.farm == f) & (df.entry_year == y)).sum() for y in years]
    ax.bar(x + (k - 0.5) * w, n, w, color=COL[f], label=f"Farm {f}")
ax.set_xticks(x); ax.set_xticklabels(years, rotation=45); ax.set_ylabel("Animals")
ax.set_title("(a) Entries per year"); ax.legend(frameon=False)

ax = axes[0, 1]
bins = np.linspace(df.gmd_kg_dia.min(), df.gmd_kg_dia.max(), 25)
for f in ["A", "B"]:
    ax.hist(df.loc[df.farm == f, "gmd_kg_dia"], bins=bins, alpha=0.75, color=COL[f])
ax.set_xlabel("ADG (kg/day)"); ax.set_ylabel("Animals"); ax.set_title("(b) ADG")

ax = axes[0, 2]
bins = np.linspace(0, 1260, 22)
for f in ["A", "B"]:
    ax.hist(df.loc[df.farm == f, "dias_permanencia"], bins=bins, alpha=0.75, color=COL[f])
ax.set_xlabel("Days on farm"); ax.set_ylabel("Animals"); ax.set_title("(c) Stay")

ax = axes[1, 0]
for f in ["A", "B"]:
    d_ = df[df.farm == f]
    ax.scatter(d_.peso_entrada_kg, d_.gmd_kg_dia, s=4, alpha=0.7, color=COL[f], edgecolor="none")
ax.set_xlabel("Entry weight (kg)"); ax.set_ylabel("ADG (kg/day)"); ax.set_title("(d) ADG vs. entry weight")

ax = axes[1, 1]
for f in ["A", "B"]:
    d_ = df[df.farm == f]
    ax.scatter(d_.proporcao_ciclo_seca, d_.gmd_kg_dia, s=4, alpha=0.7, color=COL[f], edgecolor="none")
ax.set_xlabel("Dry-season share"); ax.set_title("(e) ADG vs. dry season")

ax = axes[1, 2]
groups = [("A", 20), ("A", 25), ("A", 30), ("B", 30)]
data = [df[(df.farm == f) & (df.pb_suplemento_pct == t)].gmd_kg_dia.values for f, t in groups]
bp = ax.boxplot(data, widths=0.6, patch_artist=True, medianprops=dict(color="k"),
                flierprops=dict(markersize=1.5))
for patch, (f, t) in zip(bp["boxes"], groups):
    patch.set_facecolor(COL[f]); patch.set_alpha(0.8)
ax.set_xticks(range(1, 5)); ax.set_xticklabels([f"{f}\n{t}%" for f, t in groups])
ax.set_title("(f) ADG by supplement CP")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_eda_en.pdf"); plt.close(fig)

cols = ["peso_entrada_kg", "dias_permanencia", "proporcao_bos_indicus_pct", "media_suplemento_kg_dia",
        "pb_suplemento_pct", "media_proteina_bruta_forragem_pct", "numero_eventos_transporte",
        "temperatura_media_c", "proporcao_ciclo_seca"]
lab = {"peso_entrada_kg": "Entry weight", "dias_permanencia": "Days on farm",
       "proporcao_bos_indicus_pct": "Bos indicus share", "media_suplemento_kg_dia": "Mean supplement intake",
       "pb_suplemento_pct": "Supplement CP", "media_proteina_bruta_forragem_pct": "Forage CP",
       "numero_eventos_transporte": "Transport events", "temperatura_media_c": "Mean temperature",
       "proporcao_ciclo_seca": "Dry-season share"}
r = df[cols + ["gmd_kg_dia"]].corr()["gmd_kg_dia"].drop("gmd_kg_dia").sort_values()
fig, ax = plt.subplots(figsize=(3.6, 2.2))
ax.barh([lab[c] for c in r.index], r.values, color=[BLUE if v >= 0 else ORANGE for v in r.values])
for yv, v in enumerate(r.values):
    ax.text(v + (0.02 if v >= 0 else -0.02), yv, f"{v:+.2f}", va="center",
            ha="left" if v >= 0 else "right", fontsize=6)
ax.axvline(0, color="k", lw=0.5); ax.set_xlim(-0.8, 0.45)
ax.set_xlabel("Pearson correlation with ADG")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_corr_en.pdf"); plt.close(fig)
print("ok")

# Correlation matrix: nine predictors, accumulated precipitation (excluded) and ADG
mat_cols = {"peso_entrada_kg": "Entry weight", "dias_permanencia": "Days on farm",
            "proporcao_bos_indicus_pct": "Bos indicus share", "media_suplemento_kg_dia": "Supplement intake",
            "pb_suplemento_pct": "Supplement CP", "media_proteina_bruta_forragem_pct": "Forage CP",
            "numero_eventos_transporte": "Transport events", "temperatura_media_c": "Mean temperature",
            "proporcao_ciclo_seca": "Dry-season share", "precipitacao_acumulada_mm": "Accum. precipitation*",
            "gmd_kg_dia": "ADG"}
C = df[list(mat_cols)].corr()
lab = list(mat_cols.values())
fig, ax = plt.subplots(figsize=(4.8, 4.0))
im = ax.imshow(C.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab, rotation=45, ha="right")
ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab)
for i in range(len(lab)):
    for j in range(len(lab)):
        v = C.values[i, j]
        if abs(v) < 0.3:
            continue
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5,
                color="white" if abs(v) > 0.6 else "black")
for s_ in ax.spines.values():
    s_.set_visible(False)
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Pearson correlation")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_corrmat_en.pdf"); plt.close(fig)
