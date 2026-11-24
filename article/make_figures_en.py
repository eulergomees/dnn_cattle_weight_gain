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
                     "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})
BLUE, ORANGE = "#0072B2", "#E69F00"          # Okabe-Ito palette
COL = {"A": BLUE, "B": ORANGE}

df = pd.read_csv("data/dataset_por_animal_modelo_v3.csv", dtype={"id_animal": str})
df["farm"] = df["id_propriedade"].map({"elvis": "A", "humberto": "B"})
df["entry_year"] = pd.to_datetime(df["data_entrada"]).dt.year

fig, axes = plt.subplots(2, 3, figsize=(6.0, 4.0))
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
ax.set_xlabel("Days on farm"); ax.set_ylabel("Animals"); ax.set_title("(c) Days on farm")

ax = axes[1, 0]
for f in ["A", "B"]:
    d_ = df[df.farm == f]
    ax.scatter(d_.peso_entrada_kg, d_.gmd_kg_dia, s=6, alpha=0.7, color=COL[f], edgecolor="none")
ax.set_xlabel("Entry weight (kg)"); ax.set_ylabel("ADG (kg/day)"); ax.set_title("(d) ADG vs. entry weight")

ax = axes[1, 1]
for f in ["A", "B"]:
    d_ = df[df.farm == f]
    ax.scatter(d_.proporcao_ciclo_seca, d_.gmd_kg_dia, s=6, alpha=0.7, color=COL[f], edgecolor="none")
ax.set_xlabel("Dry-season share"); ax.set_ylabel("ADG (kg/day)"); ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["0", "", "0.5", "", "1"]); ax.set_title("(e) ADG vs. dry season")

ax = axes[1, 2]
groups = [("A", 20), ("A", 25), ("A", 30), ("B", 30)]
data = [df[(df.farm == f) & (df.pb_suplemento_pct == t)].gmd_kg_dia.values for f, t in groups]
bp = ax.boxplot(data, widths=0.6, patch_artist=True, medianprops=dict(color="k"),
                flierprops=dict(markersize=1.5))
for patch, (f, t) in zip(bp["boxes"], groups):
    patch.set_facecolor(COL[f]); patch.set_alpha(0.8)
ax.set_xticks(range(1, 5)); ax.set_xticklabels([f"{f}\n{t}%" for f, t in groups])
ax.set_ylabel("ADG (kg/day)"); ax.set_title("(f) ADG by supplement CP")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_eda_en.pdf"); plt.close(fig)


# Correlation matrix (lower triangle): six predictors, four excluded variables (*) and ADG
mat_cols = {"peso_entrada_kg": "Entry weight", "dias_permanencia": "Days on farm",
            "proporcao_bos_indicus_pct": "Bos indicus share", "pb_suplemento_pct": "Supplement CP",
            "temperatura_media_c": "Mean temperature", "proporcao_ciclo_seca": "Dry-season share",
            "media_suplemento_kg_dia": "Supplement intake*", "media_proteina_bruta_forragem_pct": "Forage CP*",
            "numero_eventos_transporte": "Transport events*", "precipitacao_acumulada_mm": "Accum. precipitation*",
            "gmd_kg_dia": "ADG"}
C = df[list(mat_cols)].corr().values
lab = list(mat_cols.values()); n = len(lab)
M = np.ma.masked_where(np.triu(np.ones((n, n)), 1).astype(bool), C)
fig, ax = plt.subplots(figsize=(5.2, 4.6))
im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(n)); ax.set_xticklabels(lab, rotation=45, ha="right")
ax.set_yticks(range(n)); ax.set_yticklabels(lab)
for i in range(n):
    for j in range(i + 1):
        v = C[i, j]
        if i == j or abs(v) < 0.3:
            continue
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                color="white" if abs(v) > 0.6 else "black")
for s_ in ax.spines.values():
    s_.set_visible(False)
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Pearson correlation")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_corrmat_en.pdf", bbox_inches="tight", pad_inches=0.05); plt.close(fig)
print("ok")
