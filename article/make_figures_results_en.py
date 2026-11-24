"""Results figures (English). Run from repo root: python article/make_figures_results_en.py
Output: article/figs/fig_cv_en.pdf, article/figs/fig_shap_en.pdf"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
COL = {"Mean": "#DDDDDD", "FarmMean": "#BBBBBB", "Linear": "#999999", "XGBoost": "#E69F00",
       "MLP": "#009E73", "TabPFN": "#0072B2"}
order = ["Mean", "FarmMean", "Linear", "XGBoost", "MLP", "TabPFN"]
lab = {"Mean": "Mean", "FarmMean": "Farm\nmean", "Linear": "Linear", "XGBoost": "XGB", "MLP": "DNN", "TabPFN": "TabPFN"}

f = pd.read_csv("results/repeated_cv_folds.csv")
fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.0), sharey=True)
for ax, (sch, t) in zip(axes, [("cohort", "(a) LG-CV"), ("farm", "(b) FS-CV")]):
    data = [f[(f.scheme == sch) & (f.model == m)].MAE.values for m in order]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False,
                    medianprops=dict(color="black", lw=1))
    for p, m in zip(bp["boxes"], order):
        p.set(facecolor=COL[m], lw=0.6)
    ax.set_xticklabels([lab[m] for m in order]); ax.set_title(t)
axes[0].set_ylabel("MAE per fold (kg/day)")
fig.tight_layout(); fig.savefig("article/figs/fig_cv_en.pdf"); plt.close(fig)

v = np.abs(np.load("results/shap_values_tabpfn.npy")).mean(0)
names = ["Entry weight", "Days on farm", "Bos indicus share", "Supplement CP", "Mean temperature", "Dry-season share"]
i = np.argsort(v)
fig, ax = plt.subplots(figsize=(4.2, 2.6))
ax.barh([names[k] for k in i], v[i], color="#0072B2")
ax.set_xlabel("Mean |SHAP| (kg/day)")
fig.tight_layout(); fig.savefig("article/figs/fig_shap_en.pdf"); plt.close(fig)
