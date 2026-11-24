"""SHAP (KernelExplainer, background 20) do TabPFN treinado nos 245 animais. Mesmo procedimento do
`shap_tabpfn.ipynb`; reexecutado em 2026-11-24 sem `media_suplemento_kg_dia`. Saída:
results/shap_values_tabpfn.npy (245, n_preditores) e results/shap_importance_tabpfn.csv."""
import os, sys, time
import numpy as np, pandas as pd, shap
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import data_prep as dp
OUT = os.environ.get("GMD_OUT", "results")

if __name__ == "__main__":
    from dotenv import load_dotenv; load_dotenv()
    from tabpfn import TabPFNRegressor
    d = dp.get_data(); df, cols = d["df"], d["scale_cols"]
    X, y = df[cols], df[dp.TARGET]
    np.random.seed(dp.SEED)
    model = TabPFNRegressor(random_state=dp.SEED).fit(X.values, y.values)
    bg = shap.sample(X, 20, random_state=dp.SEED)
    t0 = time.time()
    sv = shap.KernelExplainer(model.predict, bg).shap_values(X, nsamples="auto")
    print(f"shap: {time.time()-t0:.0f}s {sv.shape}")
    np.save(f"{OUT}/shap_values_tabpfn.npy", sv)
    imp = pd.Series(np.abs(sv).mean(0), index=cols).sort_values(ascending=False)
    imp.round(4).to_csv(f"{OUT}/shap_importance_tabpfn.csv", header=["mean_abs_shap"])
    print(imp.round(4))
