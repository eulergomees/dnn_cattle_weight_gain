"""Gera as figuras do artigo a partir de results/. Rodar da raiz do repo:
    python article/make_figures.py
Saída: article/figs/fig_eda.pdf, fig_corr.pdf, fig_esquemas_cv.pdf, fig_shap.pdf
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("article/figs", exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE = "#0072B2", "#E69F00"      # paleta Okabe-Ito (daltônicos)
ORDER = ["TabPFN", "DNN", "XGBoost", "Linear"]

# ---- EDA: panorama do conjunto (6 painéis) e correlação com o alvo ----
df = pd.read_csv("data/dataset_por_animal_modelo_v3.csv", dtype={"id_animal": str})
df["prop"] = df["id_propriedade"].map({"elvis": "A", "humberto": "B"})
df["ano_ent"] = pd.to_datetime(df["data_entrada"]).dt.year
COL = {"A": BLUE, "B": ORANGE}
fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.3))
ax = axes[0, 0]                                   # (a) entradas por ano
anos = sorted(df["ano_ent"].unique()); xa = np.arange(len(anos)); wa = 0.4
for k, pr in enumerate(["A", "B"]):
    n = [(df[(df.prop == pr) & (df.ano_ent == a)]).shape[0] for a in anos]
    ax.bar(xa + (k - 0.5) * wa, n, wa, color=COL[pr], label=f"Prop. {pr}")
ax.set_xticks(xa); ax.set_xticklabels(anos, rotation=45); ax.set_ylabel("Animais"); ax.set_title("(a) Entradas por ano"); ax.legend(frameon=False)
ax = axes[0, 1]                                   # (b) distribuição do GMD
bins = np.linspace(df.gmd_kg_dia.min(), df.gmd_kg_dia.max(), 25)
for pr in ["A", "B"]:
    ax.hist(df.loc[df.prop == pr, "gmd_kg_dia"], bins=bins, alpha=0.75, color=COL[pr], label=f"Prop. {pr}")
ax.set_xlabel("GMD (kg/dia)"); ax.set_ylabel("Animais"); ax.set_title("(b) Distribui\u00e7\u00e3o do GMD")
ax = axes[0, 2]                                   # (c) dias de permanência
bins = np.linspace(0, 1260, 22)
for pr in ["A", "B"]:
    ax.hist(df.loc[df.prop == pr, "dias_permanencia"], bins=bins, alpha=0.75, color=COL[pr])
ax.set_xlabel("Dias de perman\u00eancia"); ax.set_ylabel("Animais"); ax.set_title("(c) Perman\u00eancia")
ax = axes[1, 0]                                   # (d) GMD x peso de entrada
for pr in ["A", "B"]:
    d_ = df[df.prop == pr]; ax.scatter(d_.peso_entrada_kg, d_.gmd_kg_dia, s=7, alpha=0.7, color=COL[pr], edgecolor="none")
ax.set_xlabel("Peso de entrada (kg)"); ax.set_ylabel("GMD (kg/dia)"); ax.set_title("(d) GMD \u00d7 peso de entrada")
ax = axes[1, 1]                                   # (e) GMD x proporção do ciclo na seca
for pr in ["A", "B"]:
    d_ = df[df.prop == pr]; ax.scatter(d_.proporcao_ciclo_seca, d_.gmd_kg_dia, s=7, alpha=0.7, color=COL[pr], edgecolor="none")
ax.set_xlabel("Propor\u00e7\u00e3o do ciclo na seca"); ax.set_title("(e) GMD \u00d7 estiagem")
ax = axes[1, 2]                                   # (f) GMD por teor de PB do suplemento
grupos = [("A", 20), ("A", 25), ("A", 30), ("B", 30)]
dados = [df[(df.prop == p_) & (df.pb_suplemento_pct == t)].gmd_kg_dia.values for p_, t in grupos]
bp = ax.boxplot(dados, widths=0.6, patch_artist=True, medianprops=dict(color="k"), flierprops=dict(markersize=2))
for patch, (p_, t) in zip(bp["boxes"], grupos):
    patch.set_facecolor(COL[p_]); patch.set_alpha(0.8)
ax.set_xticks(range(1, 5)); ax.set_xticklabels([f"{p_}\n{t}%" for p_, t in grupos])
ax.set_title("(f) GMD por PB do suplemento")
fig.tight_layout(); fig.savefig("article/figs/fig_eda.pdf"); plt.close(fig)

alvo = "gmd_kg_dia"
pred = ["peso_entrada_kg", "dias_permanencia", "proporcao_bos_indicus_pct", "media_suplemento_kg_dia",
        "pb_suplemento_pct", "media_proteina_bruta_forragem_pct", "numero_eventos_transporte",
        "temperatura_media_c", "proporcao_ciclo_seca"]
lab = {"peso_entrada_kg": "Peso de entrada", "dias_permanencia": "Dias de perman\u00eancia",
       "proporcao_bos_indicus_pct": "% Bos indicus", "media_suplemento_kg_dia": "Suplemento m\u00e9dio",
       "pb_suplemento_pct": "PB do suplemento", "media_proteina_bruta_forragem_pct": "PB da forragem",
       "numero_eventos_transporte": "N\u00ba de transportes", "temperatura_media_c": "Temperatura m\u00e9dia",
       "proporcao_ciclo_seca": "Prop. do ciclo na seca"}
rr = df[pred + [alvo]].corr()[alvo].drop(alvo).sort_values()
fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.barh([lab[c] for c in rr.index], rr.values, color=[BLUE if v >= 0 else ORANGE for v in rr.values])
for yv, v in enumerate(rr.values):
    ax.text(v + (0.02 if v >= 0 else -0.02), yv, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=6.5)
ax.axvline(0, color="k", lw=0.6); ax.set_xlim(-0.8, 0.45)
ax.set_xlabel("Correla\u00e7\u00e3o de Pearson com o GMD")
fig.tight_layout(); fig.savefig("article/figs/fig_corr.pdf"); plt.close(fig)

# ---- Fig. 1: R2 sob CV por propriedade x CV agrupada por lote ----
r = pd.read_csv("results/cv_lote_vs_farm.csv").set_index("modelo").loc[ORDER]
fig, ax = plt.subplots(figsize=(3.4, 2.5))
x = np.arange(len(ORDER)); w = 0.38
b1 = ax.bar(x - w/2, r["R2_farm"], w, color=BLUE, label="por propriedade")
b2 = ax.bar(x + w/2, r["R2_lote"], w, color=ORANGE, label="por lote")
for b in list(b1) + list(b2):
    h = b.get_height()
    ax.text(b.get_x() + b.get_width()/2, h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=6.5)
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(x); ax.set_xticklabels(ORDER)
ax.set_ylabel("R² médio nos folds"); ax.set_ylim(0, 1.0)
ax.legend(loc="upper right", frameon=False, title="Validação cruzada", title_fontsize=7.5)
fig.tight_layout(); fig.savefig("article/figs/fig_esquemas_cv.pdf"); plt.close(fig)

# ---- Fig. 2: importancia SHAP (TabPFN) ----
sv = np.load("results/shap_values_tabpfn.npy")
labels = {"peso_entrada_kg": "Peso de entrada", "media_suplemento_kg_dia": "Suplemento médio (kg/dia)",
          "dias_permanencia": "Dias de permanência", "proporcao_ciclo_seca": "Prop. do ciclo na seca",
          "temperatura_media_c": "Temperatura média", "media_proteina_bruta_forragem_pct": "PB da forragem",
          "numero_eventos_transporte": "Nº de transportes", "pb_suplemento_pct": "PB do suplemento",
          "proporcao_bos_indicus_pct": "% Bos indicus"}
cols = ["peso_entrada_kg", "dias_permanencia", "proporcao_bos_indicus_pct", "media_suplemento_kg_dia",
        "pb_suplemento_pct", "media_proteina_bruta_forragem_pct", "numero_eventos_transporte",
        "temperatura_media_c", "proporcao_ciclo_seca"]       # ordem das colunas de shap_values_tabpfn.npy
assert sv.shape[1] == len(cols)
imp = pd.Series(np.abs(sv).mean(0), index=cols).sort_values()
fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.barh([labels[c] for c in imp.index], imp.values, color=BLUE)
for yv, v in enumerate(imp.values):
    ax.text(v + 0.003, yv, f"{v:.3f}", va="center", fontsize=6.5)
ax.set_xlabel("|SHAP| médio (kg/dia)"); ax.set_xlim(0, imp.max() * 1.18)
fig.tight_layout(); fig.savefig("article/figs/fig_shap.pdf"); plt.close(fig)
print("figuras geradas em article/figs/")
