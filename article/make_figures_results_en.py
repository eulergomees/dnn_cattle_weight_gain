"""Results figures (English). Run from repo root: python article/make_figures_results_en.py
Output: article/figs/fig_cv_en.pdf, article/figs/fig_shap_en.pdf"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
COL = {"Linear": "#999999", "XGBoost": "#E69F00", "MLP": "#009E73", "TabPFN": "#0072B2"}
order = ["Linear", "XGBoost", "MLP", "TabPFN"]

f = pd.read_csv("results/repeated_cv_folds.csv")
fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.2), sharey=True)
for ax, (sch, t) in zip(axes, [("farm", "(a) Farm-stratified"), ("cohort", "(b) Lot-grouped")]):
    data = [f[(f.scheme == sch) & (f.model == m)].MAE.values for m in order]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False,
                    medianprops=dict(color="black", lw=0.8))
    for p, m in zip(bp["boxes"], order):
        p.set(facecolor=COL[m], alpha=0.8, lw=0.5)
    ax.set_xticklabels(order); ax.set_title(t)
axes[0].set_ylabel("MAE per fold (kg/day)")
fig.tight_layout(); fig.savefig("article/figs/fig_cv_en.pdf")

v = np.abs(np.load("results/shap_values_tabpfn.npy")).mean(0)
names = ["Entry weight", "Days on farm", "Bos indicus share", "Mean supplement intake", "Supplement CP",
         "Forage CP", "Transport events", "Mean temperature", "Dry-season share"]
i = np.argsort(v)
fig, ax = plt.subplots(figsize=(3.4, 2.2))
ax.barh([names[k] for k in i], v[i], color="#0072B2")
ax.set_xlabel("Mean |SHAP| (kg/day)")
fig.tight_layout(); fig.savefig("article/figs/fig_shap_en.pdf")
