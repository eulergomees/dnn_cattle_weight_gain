"""Avaliação padronizada de modelos scikit-learn para os notebooks de baseline.

Usa o MESMO `data_prep` da DNN (folds, métricas, LOPO, jitter) → comparação justa.
`scale=True` p/ modelos lineares (StandardScaler fit só no treino); árvores não
precisam. `augment=True` aplica o jitter feature-only só no treino.

    import sk_eval
    d = dp.get_data()
    sk_eval.registrar("RandomForest", lambda: RandomForestRegressor(...),
                      scale=False, d=d)
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import data_prep as dp


def _predict(train_df, eval_df, make_model, scale, augment, cols, tgt):
    tr = dp.augment_train(train_df, k=2) if augment else train_df
    est = make_model()
    model = make_pipeline(StandardScaler(), est) if scale else est
    model.fit(tr[cols], tr[tgt])
    return model.predict(eval_df[cols])


def cv_eval(d, make_model, scale, augment=False):
    """CV 10-fold -> DataFrame de métricas por fold."""
    df, cols, tgt = d["df"], d["scale_cols"], dp.TARGET
    fold_m = []
    for tr, va in dp.get_cv().split(df):
        p = _predict(df.iloc[tr], df.iloc[va], make_model, scale, augment, cols, tgt)
        fold_m.append(dp.metrics(df.iloc[va][tgt].values, p))
    return dp.summarize(fold_m)[0]


def oof(d, make_model, scale, augment=False):
    """Predições out-of-fold (para o gráfico predito × real)."""
    df, cols, tgt = d["df"], d["scale_cols"], dp.TARGET
    y = np.zeros(len(df))
    for tr, va in dp.get_cv().split(df):
        y[va] = _predict(df.iloc[tr], df.iloc[va], make_model, scale, augment, cols, tgt)
    return y


def lopo(d, make_model, scale):
    """Leave-one-property-out (treina 2 fazendas / testa a 3ª)."""
    df, cols, tgt, groups = d["df"], d["scale_cols"], dp.TARGET, d["groups"]
    rows = []
    for tr, te, prop in dp.leave_one_property_out(groups):
        p = _predict(df.iloc[tr], df.iloc[te], make_model, scale, False, cols, tgt)
        m = dp.metrics(df.iloc[te][tgt].values, p); m["testa"] = prop
        rows.append(m)
    return pd.DataFrame(rows)[["testa", "MAE", "RMSE", "R2"]].round(4)


def registrar(nome, make_model, scale, d):
    """Roda CV (com/sem jitter) + LOPO, grava em results/model_comparison.csv,
    e devolve dict com os resultados. Salva o melhor entre com/sem jitter."""
    cvp = cv_eval(d, make_model, scale, augment=False)
    cva = cv_eval(d, make_model, scale, augment=True)
    lp = lopo(d, make_model, scale)
    print(f"{nome}: CV R2 sem={cvp.R2.mean():.3f} / com jitter={cva.R2.mean():.3f}"
          f"  | MAE={min(cvp.MAE.mean(), cva.MAE.mean()):.4f}")
    print("  LOPO R2:", {r.testa: round(r.R2, 2) for r in lp.itertuples()})
    best = cva if cva.R2.mean() > cvp.R2.mean() else cvp
    dp.save_result(nome, best, ext_metrics={"MAE": lp.MAE.mean(), "R2": lp.R2.mean()})
    return dict(sem_jitter=cvp, com_jitter=cva, lopo=lp)
