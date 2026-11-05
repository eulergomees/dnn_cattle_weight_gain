"""CV repetido (por fazenda e por lote), teste pareado e intervalos preditivos do TabPFN.

Roda os 4 modelos do estudo (Linear, XGBoost, MLP congelada, TabPFN) em R repetições
de 5 folds, nos dois esquemas de validação, com as MESMAS partições para todos os
modelos (comparação pareada). Não altera `data_prep.py` nem reajusta hiperparâmetros:
XGBoost e MLP usam as configurações já escolhidas (as mesmas do artigo).

    python support_scripts/repeated_cv.py --repeats 10            # da raiz do repo
    python support_scripts/repeated_cv.py --repeats 1 --models Linear XGBoost   # teste rápido

Saídas (results/): repeated_cv_folds.csv (uma linha por fold×modelo), repeated_cv_oof.csv
(R²/MAE com as predições OOF de cada repetição) e repeated_cv_summary.csv / _tests.csv.
Cada combinação (esquema, modelo) é gravada ao terminar; reexecutar pula as já feitas
(use --force para refazer).

Esquemas:
  farm   — `FarmStratifiedSplit(seed=r)`: 5 folds de 49, estratificados por propriedade.
  cohort — folds agrupados por lote (propriedade+data_entrada+data_saida), sorteados por
           repetição; nenhum lote é dividido entre treino e teste. Não estratificado.

Teste pareado: t reamostrado corrigido de Nadeau & Bengio (2003), sobre o MAE por fold,
com correção de Holm entre os pares de cada esquema.
"""
import argparse
import copy
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import data_prep as dp

OUT = "results"
FOLDS_CSV = f"{OUT}/repeated_cv_folds.csv"
OOF_CSV = f"{OUT}/repeated_cv_oof.csv"
SUMMARY_CSV = f"{OUT}/repeated_cv_summary.csv"
TESTS_CSV = f"{OUT}/repeated_cv_tests.csv"
MODELS = ["Linear", "XGBoost", "MLP", "TabPFN"]
SCHEMES = ["farm", "cohort"]
QUANTILES = [0.05, 0.95]                       # intervalo preditivo de 90% do TabPFN

# configurações já escolhidas no estudo (mesmas do artigo)
XGB_PARAMS = dict(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.8,
                  min_child_weight=1)
MLP_CFG = dict(hidden=(128, 64, 32, 16), p_drop=0.0, lr=1e-2, batchnorm=True, wd=1e-4)

device = "cuda" if torch.cuda.is_available() else "cpu"


# ----- partições -----

def farm_splits(df, seed):
    return dp.FarmStratifiedSplit(seed=seed).split(df)


def cohort_splits(df, seed, n_splits=dp.N_SPLITS):
    """5 folds sem dividir lotes; a ordem dos lotes é sorteada por `seed`, depois cada lote
    (do maior ao menor) vai para o fold mais vazio — folds de tamanho parecido, mas não iguais."""
    cohort = df[dp.GROUP].astype(str) + "|" + df["data_entrada"] + "|" + df["data_saida"]
    codes, _ = pd.factorize(cohort)
    sizes = np.bincount(codes)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sizes))
    order = order[np.argsort(-sizes[order], kind="stable")]
    fold_of, load = np.empty(len(sizes), dtype=int), np.zeros(n_splits)
    for c in order:
        k = int(np.argmin(load))
        fold_of[c] = k
        load[k] += sizes[c]
    fold = fold_of[codes]
    for k in range(n_splits):
        yield np.flatnonzero(fold != k), np.flatnonzero(fold == k)


SPLITTERS = {"farm": farm_splits, "cohort": cohort_splits}


# ----- modelos: cada um devolve dict(pred=..., [lo=..., hi=...]) -----

class GMDNN(nn.Module):               # mesma rede de dnn_gmd.ipynb
    def __init__(self, n_in, hidden, p_drop, batchnorm):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers += [nn.ReLU(), nn.Dropout(p_drop)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _mlp_one(tr_df, te_df, cols, seed, cfg=MLP_CFG):
    """Mesmo treino de `train_eval` (dnn_gmd.ipynb), sem jitter."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    fit_df, es_df = train_test_split(tr_df, test_size=0.25, random_state=seed)
    xs = StandardScaler().fit(fit_df[cols])
    ys = StandardScaler().fit(fit_df[[dp.TARGET]])

    def tens(dd):
        return (torch.tensor(xs.transform(dd[cols]), dtype=torch.float32),
                torch.tensor(ys.transform(dd[[dp.TARGET]]), dtype=torch.float32))

    Xtr, ytr = tens(fit_df)
    Xes, yes = tens(es_df)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=32, shuffle=True, drop_last=True)
    model = GMDNN(len(cols), cfg["hidden"], cfg["p_drop"], cfg["batchnorm"]).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8)
    crit = nn.SmoothL1Loss()
    Xes_d, yes_d = Xes.to(device), yes.to(device)
    best, best_state, wait = 1e9, None, 0
    for _ in range(300):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = crit(model(Xes_d), yes_d).item()
        sched.step(v)
        if v < best - 1e-5:
            best, best_state, wait = v, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= 60:
                break
    model.load_state_dict(best_state)
    model.eval()
    Xe = torch.tensor(xs.transform(te_df[cols]), dtype=torch.float32).to(device)
    with torch.no_grad():
        return ys.inverse_transform(model(Xe).cpu().numpy()).ravel()


def predict(model_name, tr_df, te_df, cols, seed, n_ens):
    if model_name == "Linear":
        m = make_pipeline(StandardScaler(), LinearRegression())
        return dict(pred=m.fit(tr_df[cols], tr_df[dp.TARGET]).predict(te_df[cols]))
    if model_name == "XGBoost":
        m = XGBRegressor(**XGB_PARAMS, random_state=seed, n_jobs=4)
        return dict(pred=m.fit(tr_df[cols], tr_df[dp.TARGET]).predict(te_df[cols]))
    if model_name == "MLP":
        preds = [_mlp_one(tr_df, te_df, cols, seed + i) for i in range(n_ens)]
        return dict(pred=np.mean(preds, axis=0))
    if model_name == "TabPFN":
        from tabpfn import TabPFNRegressor
        m = TabPFNRegressor(random_state=seed).fit(tr_df[cols], tr_df[dp.TARGET])
        lo, hi = m.predict(te_df[cols], output_type="quantiles", quantiles=QUANTILES)
        return dict(pred=m.predict(te_df[cols]), lo=np.asarray(lo), hi=np.asarray(hi))
    raise ValueError(model_name)


# ----- execução -----

def run_combo(df, cols, scheme, model_name, repeats, n_ens):
    fold_rows, oof_rows = [], []
    y = df[dp.TARGET].values
    for r in range(repeats):
        oof = np.full(len(df), np.nan)
        for k, (tr, te) in enumerate(SPLITTERS[scheme](df, seed=dp.SEED + r)):
            out = predict(model_name, df.iloc[tr], df.iloc[te], cols,
                          seed=dp.SEED + 1000 * r + 10 * k, n_ens=n_ens)
            oof[te] = out["pred"]
            row = dict(scheme=scheme, model=model_name, repeat=r, fold=k, n_test=len(te),
                       **dp.metrics(y[te], out["pred"]))
            if "lo" in out:
                row["cover90"] = float(np.mean((y[te] >= out["lo"]) & (y[te] <= out["hi"])))
                row["width90"] = float(np.mean(out["hi"] - out["lo"]))
            fold_rows.append(row)
        oof_rows.append(dict(scheme=scheme, model=model_name, repeat=r,
                             **{f"oof_{k}": v for k, v in dp.metrics(y, oof).items()}))
    return pd.DataFrame(fold_rows), pd.DataFrame(oof_rows)


def _upsert(path, new, scheme, model_name):
    if os.path.exists(path):
        old = pd.read_csv(path)
        old = old[~((old.scheme == scheme) & (old.model == model_name))]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)


def _done(scheme, model_name):
    if not os.path.exists(FOLDS_CSV):
        return False
    f = pd.read_csv(FOLDS_CSV)
    return bool(((f.scheme == scheme) & (f.model == model_name)).any())


# ----- estatística -----

def nb_test(d, k=dp.N_SPLITS):
    """t reamostrado corrigido (Nadeau & Bengio, 2003) para diferenças pareadas por fold.
    n_teste/n_treino = 1/(k-1). Devolve (média, erro-padrão corrigido, t, p bilateral)."""
    d = np.asarray(d, float)
    J = len(d)
    mean, var = d.mean(), d.var(ddof=1)
    se = np.sqrt((1.0 / J + 1.0 / (k - 1)) * var)
    t = mean / se if se > 0 else np.inf
    return mean, se, t, 2 * stats.t.sf(abs(t), J - 1)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def summarize(folds, oof):
    rows = []
    for (scheme, model), g in folds.groupby(["scheme", "model"]):
        J = len(g)
        mean = g.MAE.mean()
        se = np.sqrt((1.0 / J + 1.0 / (dp.N_SPLITS - 1)) * g.MAE.var(ddof=1))
        tcrit = stats.t.ppf(0.975, J - 1)
        o = oof[(oof.scheme == scheme) & (oof.model == model)]
        row = dict(scheme=scheme, model=model, repeats=int(g.repeat.nunique()), folds=J,
                   MAE=mean, MAE_std=g.MAE.std(), MAE_ci95_lo=mean - tcrit * se,
                   MAE_ci95_hi=mean + tcrit * se, R2_fold=g.R2.mean(), R2_fold_std=g.R2.std(),
                   R2_oof=o.oof_R2.mean(), R2_oof_std=o.oof_R2.std(), MAE_oof=o.oof_MAE.mean())
        if "cover90" in g and g.cover90.notna().any():
            row.update(cover90=g.cover90.mean(), width90=g.width90.mean())
        rows.append(row)
    summary = pd.DataFrame(rows)

    trows = []
    for scheme, g in folds.groupby("scheme"):
        wide = g.pivot_table(index=["repeat", "fold"], columns="model", values="MAE")
        pairs = list(itertools.combinations(wide.columns, 2))
        res = []
        for a, b in pairs:
            mean, se, t, p = nb_test((wide[a] - wide[b]).dropna())
            res.append(dict(scheme=scheme, model_a=a, model_b=b, mean_diff_MAE=mean, se=se, t=t, p=p))
        adj = holm([r["p"] for r in res])
        for r, pa in zip(res, adj):
            r["p_holm"] = pa
            trows.append(r)
    return summary, pd.DataFrame(trows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--schemes", nargs="+", default=SCHEMES, choices=SCHEMES)
    ap.add_argument("--n-ens", type=int, default=1, help="redes por fold na MLP (artigo: 5)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    if "TabPFN" in a.models:
        from dotenv import load_dotenv
        load_dotenv()
    os.makedirs(OUT, exist_ok=True)
    d = dp.get_data()
    df, cols = d["df"], d["scale_cols"]
    print(f"{len(df)} animais | {len(cols)} preditores | device={device}")

    for scheme in a.schemes:
        for model in a.models:
            if not a.force and _done(scheme, model):
                print(f"[{scheme}/{model}] já feito — pulando")
                continue
            t0 = time.time()
            f, o = run_combo(df, cols, scheme, model, a.repeats, a.n_ens)
            _upsert(FOLDS_CSV, f, scheme, model)
            _upsert(OOF_CSV, o, scheme, model)
            print(f"[{scheme}/{model}] {a.repeats} repetições em {time.time() - t0:.0f}s | "
                  f"MAE={f.MAE.mean():.4f}  R2(fold)={f.R2.mean():.3f}  R2(OOF)={o.oof_R2.mean():.3f}")

    folds, oof = pd.read_csv(FOLDS_CSV), pd.read_csv(OOF_CSV)
    summary, tests = summarize(folds, oof)
    summary.round(4).to_csv(SUMMARY_CSV, index=False)
    tests.round(5).to_csv(TESTS_CSV, index=False)
    print(summary.round(4).to_string(index=False))
    print(tests.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
