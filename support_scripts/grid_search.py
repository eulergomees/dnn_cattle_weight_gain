"""Grid search de XGBoost (48 configs) e DNN (72 configs) nos 5 folds de `dp.get_cv()`
(folds disjuntos de 49 animais, estratificados por fazenda). Critério: maior R² médio entre os 5 folds
(mesmo critério dos notebooks). Reexecutado em 2026-11-24 depois de remover `media_suplemento_kg_dia`
dos preditores (vazamento do alvo). Uma rede por fold (seed = dp.SEED), como no `dnn_gmd.ipynb`.

    python support_scripts/grid_search.py [--models xgb dnn]
Saídas: results/grid_search_{xgb,dnn}.csv (ordenado por R²)."""
import argparse, os, sys
from itertools import product
import numpy as np, pandas as pd
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import data_prep as dp
import repeated_cv as rc

XGB_GRID = dict(n_estimators=[200, 400], max_depth=[2, 3, 4], learning_rate=[0.03, 0.1],
                subsample=[0.8, 1.0], min_child_weight=[1, 5])
DNN_GRID = dict(hidden=[(64, 32), (128, 64, 32), (128, 64, 32, 16)], p_drop=[0.0, 0.1],
                lr=[3e-3, 5e-3, 1e-2], batchnorm=[False, True], wd=[0.0, 1e-4])


def cv_metrics(df, cols, predict):
    maes, r2s = [], []
    for tr, te in dp.get_cv().split(df):
        y = df.iloc[te][dp.TARGET].values
        p = predict(df.iloc[tr], df.iloc[te])
        maes.append(np.abs(p - y).mean()); r2s.append(r2_score(y, p))
    return np.mean(maes), np.mean(r2s), np.std(r2s)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", default=["xgb", "dnn"])
    a = ap.parse_args()
    d = dp.get_data(); df, cols = d["df"], d["scale_cols"]
    print("preditores:", cols, flush=True)
    if "xgb" in a.models:
        rows = []
        for vals in product(*XGB_GRID.values()):
            cfg = dict(zip(XGB_GRID, vals))
            f = lambda tr, te, cfg=cfg: XGBRegressor(**cfg, random_state=dp.SEED, n_jobs=4).fit(
                tr[cols], tr[dp.TARGET]).predict(te[cols])
            mae, r2, sd = cv_metrics(df, cols, f)
            rows.append({**cfg, "MAE": mae, "R2": r2, "R2_std": sd})
        res = pd.DataFrame(rows).sort_values("R2", ascending=False)
        res.to_csv(f"{rc.OUT}/grid_search_xgb.csv", index=False)
        print("XGB melhor:\n", res.head(3).round(4).to_string(), flush=True)
    if "dnn" in a.models:
        rows = []
        for i, vals in enumerate(product(*DNN_GRID.values())):
            cfg = dict(zip(DNN_GRID, vals))
            f = lambda tr, te, cfg=cfg: rc._mlp_one(tr, te, cols, dp.SEED, cfg)
            mae, r2, sd = cv_metrics(df, cols, f)
            rows.append({**cfg, "hidden": str(cfg["hidden"]), "MAE": mae, "R2": r2, "R2_std": sd})
            print(i + 1, "/72", cfg, round(r2, 4), flush=True)
        res = pd.DataFrame(rows).sort_values("R2", ascending=False)
        res.to_csv(f"{rc.OUT}/grid_search_dnn.csv", index=False)
        print("DNN melhor:\n", res.head(3).round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
