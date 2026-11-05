"""Validação aninhada (XGBoost e MLP) — análise de sensibilidade ao viés de seleção.

No estudo principal, os hiperparâmetros do XGBoost e da MLP foram escolhidos nas MESMAS
dobras usadas para reportar o desempenho (o TabPFN e a regressão linear não são ajustados).
Aqui a busca em grade roda DENTRO de cada fold externo, só com o treino dele, e o fold de
teste só vê o modelo final. A diferença para o desempenho não aninhado mede o viés.

A configuração congelada da MLP continua sendo o resultado principal; este script não a
altera. Usa as mesmas grades do estudo (XGBoost 48, MLP 72 configurações) e as partições
`farm` de `repeated_cv.py` (mesmos folds externos), com 3 folds internos estratificados
por propriedade e seleção por R² médio interno (como no estudo).

    python support_scripts/nested_cv.py --repeats 2                 # da raiz do repo
    python support_scripts/nested_cv.py --repeats 1 --models XGBoost

Saídas (results/): nested_cv_folds.csv e nested_cv_summary.csv (nested × não aninhado).
"""
import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import data_prep as dp
import repeated_cv as rcv

FOLDS_CSV = f"{rcv.OUT}/nested_cv_folds.csv"
SUMMARY_CSV = f"{rcv.OUT}/nested_cv_summary.csv"
INNER_SPLITS = 3

XGB_GRID = dict(n_estimators=[200, 400], max_depth=[2, 3, 4], learning_rate=[0.03, 0.1],
                subsample=[0.8, 1.0], min_child_weight=[1, 5])
MLP_GRID = dict(hidden=[(64, 32), (128, 64, 32), (128, 64, 32, 16)], p_drop=[0.0, 0.1],
                lr=[3e-3, 5e-3, 1e-2], batchnorm=[False, True], wd=[0.0, 1e-4])


def configs(grid):
    keys = list(grid)
    return [dict(zip(keys, v)) for v in itertools.product(*grid.values())]


def inner_folds(tr_df, seed):
    skf = StratifiedKFold(INNER_SPLITS, shuffle=True, random_state=seed)
    return list(skf.split(tr_df, tr_df[dp.GROUP]))


def fit_predict(model, cfg, tr_df, te_df, cols, seed, n_ens):
    if model == "XGBoost":
        m = XGBRegressor(**cfg, random_state=seed, n_jobs=4)
        return m.fit(tr_df[cols], tr_df[dp.TARGET]).predict(te_df[cols])
    preds = [rcv._mlp_one(tr_df, te_df, cols, seed + i, cfg=cfg) for i in range(n_ens)]
    return np.mean(preds, axis=0)


def select(model, grid, tr_df, cols, seed):
    """Melhor configuração pelo R² médio nos folds internos (1 rede por configuração)."""
    inner = inner_folds(tr_df, seed)
    best_cfg, best_r2 = None, -np.inf
    for cfg in configs(grid):
        r2 = []
        for itr, iva in inner:
            p = fit_predict(model, cfg, tr_df.iloc[itr], tr_df.iloc[iva], cols, seed, n_ens=1)
            r2.append(dp.metrics(tr_df.iloc[iva][dp.TARGET].values, p)["R2"])
        if np.mean(r2) > best_r2:
            best_cfg, best_r2 = cfg, float(np.mean(r2))
    return best_cfg, best_r2


def run(model, repeats, n_ens, df, cols):
    grid = XGB_GRID if model == "XGBoost" else MLP_GRID
    y = df[dp.TARGET].values
    rows = []
    for r in range(repeats):
        for k, (tr, te) in enumerate(rcv.farm_splits(df, seed=dp.SEED + r)):
            t0 = time.time()
            seed = dp.SEED + 1000 * r + 10 * k
            cfg, inner_r2 = select(model, grid, df.iloc[tr], cols, seed)
            p = fit_predict(model, cfg, df.iloc[tr], df.iloc[te], cols, seed, n_ens)
            rows.append(dict(scheme="farm", model=model, repeat=r, fold=k, n_test=len(te),
                             **dp.metrics(y[te], p), inner_R2=inner_r2, cfg=str(cfg)))
            print(f"[{model}] rep {r} fold {k}: R2={rows[-1]['R2']:.3f} MAE={rows[-1]['MAE']:.4f} "
                  f"(interno R2={inner_r2:.3f}) {time.time() - t0:.0f}s  {cfg}", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--models", nargs="+", default=["XGBoost", "MLP"], choices=["XGBoost", "MLP"])
    ap.add_argument("--n-ens", type=int, default=5, help="redes por fold externo na MLP")
    a = ap.parse_args()

    d = dp.get_data()
    df, cols = d["df"], d["scale_cols"]
    os.makedirs(rcv.OUT, exist_ok=True)
    out = []
    for model in a.models:
        f = run(model, a.repeats, a.n_ens, df, cols)
        out.append(f)
        pd.concat(out).to_csv(FOLDS_CSV, index=False)

    nested = pd.concat(out)
    summary = []
    if os.path.exists(rcv.FOLDS_CSV):
        plain = pd.read_csv(rcv.FOLDS_CSV)
        plain = plain[plain.scheme == "farm"]
    else:
        plain = pd.DataFrame()
    for model, g in nested.groupby("model"):
        row = dict(model=model, repeats=a.repeats, nested_MAE=g.MAE.mean(), nested_R2=g.R2.mean(),
                   nested_R2_std=g.R2.std())
        if len(plain):
            p = plain[(plain.model == model) & (plain.repeat < a.repeats)]
            m = g.merge(p, on=["repeat", "fold"], suffixes=("", "_plain"))
            row.update(plain_MAE=m.MAE_plain.mean(), plain_R2=m.R2_plain.mean(),
                       delta_MAE=m.MAE.mean() - m.MAE_plain.mean(),
                       delta_R2=m.R2.mean() - m.R2_plain.mean())
        summary.append(row)
    summary = pd.DataFrame(summary).round(4)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
