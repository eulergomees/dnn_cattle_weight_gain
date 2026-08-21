"""Pré-processamento compartilhado (schema por-animal v3) dos notebooks de modelo.

Todo modelo (linear, árvores, MLP, ...) usa os MESMOS dados, folds e métricas —
comparação justa. Import a partir da raiz do repositório:

    import sys; sys.path.append("support_scripts")
    import data_prep as dp
    d = dp.get_data()                 # X, y, groups, feature_cols, scale_cols, df

Alvo: `gmd_kg_dia`. Split: 10-fold CV (não-agrupado) para comparar/tunar +
leave-one-property-out (Elvis↔Sonico) para robustez/validade externa.
Scaling NÃO é feito aqui (cada notebook aplica StandardScaler fit só no treino).
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "data/dataset_por_animal_modelo_v3.csv"
TARGET = "gmd_kg_dia"
GROUP = "id_propriedade"
# fora dos preditores: identificação, vazamento do alvo e multicolinear (r=0.97 c/ dias)
DROP = ["id_animal", "id_propriedade", "data_entrada", "data_saida",
        "peso_saida_kg", "precipitacao_acumulada_mm"]
SEED = 42
N_SPLITS = 10
RESULTS_PATH = "results/model_comparison.csv"

# escalas do jitter (ruído de medição) — só aplicado no treino, pós-split
NOISE = {"peso_entrada_kg": 2.5, "peso_saida_kg": 2.5, "temperatura_media_c": 0.5}


def load_raw(path=DATA_PATH):
    return pd.read_csv(path, dtype={"id_animal": str})


def feature_columns(df, verbose=True):
    """(feature_cols, scale_cols) após dropar ids/leak/alvo/precip e CONSTANTES."""
    base = [c for c in df.columns if c not in set(DROP) | {TARGET}]
    consts = [c for c in base if df[c].nunique() == 1]
    feature_cols = [c for c in base if c not in consts]
    scale_cols = list(feature_cols)          # todos numéricos; notebooks decidem escalar
    if verbose:
        print(f"constantes removidas ({len(consts)}):", consts)
        print(f"preditores efetivos ({len(feature_cols)}):", feature_cols)
    return feature_cols, scale_cols


def get_data():
    """dict com df (completo), X, y, groups (id_propriedade), feature_cols, scale_cols."""
    df = load_raw()
    feature_cols, scale_cols = feature_columns(df, verbose=False)
    return dict(df=df, X=df[feature_cols], y=df[TARGET], groups=df[GROUP],
                feature_cols=feature_cols, scale_cols=scale_cols)


def get_cv():
    """10-fold CV (não-agrupado) — métrica principal de comparação/tuning."""
    return KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)


def leave_one_property_out(groups):
    """Gera (train_idx, test_idx, propriedade_de_teste) para cada propriedade.

    Robustez/validade externa: treina numa fazenda, testa na outra.
    """
    g = np.asarray(groups)
    idx = np.arange(len(g))
    for p in pd.unique(g):
        yield idx[g != p], idx[g == p], p


def augment_train(df_train, k=2, seed=SEED):
    """Jitter (ruído de medição) — expande SÓ o treino com `k` cópias por linha.

    Perturba pesos e temperatura e REDERIVA `gmd_kg_dia` (e `media_suplemento_kg_dia`
    dos proteinados) a partir dos pesos perturbados. NUNCA chamar em val/test.
    Recebe um recorte do df COMPLETO (precisa de peso_saida_kg e dias_permanencia).
    """
    rng = np.random.default_rng(seed)
    copies = [df_train]
    for _ in range(k):
        a = df_train.copy()
        pe = (a["peso_entrada_kg"] + rng.normal(0, NOISE["peso_entrada_kg"], len(a))).clip(lower=1)
        ps = (a["peso_saida_kg"] + rng.normal(0, NOISE["peso_saida_kg"], len(a))).clip(lower=1)
        a["peso_entrada_kg"] = pe
        a["peso_saida_kg"] = ps
        a["gmd_kg_dia"] = (ps - pe) / a["dias_permanencia"]
        prot = a["pb_suplemento_pct"] > 0                     # proteinado: 0,3% do peso médio
        a.loc[prot, "media_suplemento_kg_dia"] = (0.003 * (pe[prot] + ps[prot]) / 2).round(3)
        a["temperatura_media_c"] = a["temperatura_media_c"] + rng.normal(0, NOISE["temperatura_media_c"], len(a))
        copies.append(a)
    return pd.concat(copies, ignore_index=True)


# ----- utilidades schema-agnósticas (métricas / resultados) -----

def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {"MAE":  mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2":   r2_score(y_true, y_pred)}


def baseline_mae(y_train, y_eval):
    """MAE do baseline ingênuo (prever a média do treino)."""
    return mean_absolute_error(y_eval, np.full(len(y_eval), np.mean(y_train)))


def summarize(fold_metrics):
    """lista de dicts -> (DataFrame por fold, DataFrame média/dp)."""
    cv = pd.DataFrame(fold_metrics)
    return cv, cv.agg(["mean", "std"]).round(4)


def save_result(name, cv_df, ext_metrics=None, path=RESULTS_PATH):
    """Grava/atualiza os scores do modelo na tabela comparativa compartilhada.

    cv_df: métricas por fold do 10-fold CV. ext_metrics (opcional): dict com a
    média do leave-one-property-out, p.ex. {"MAE":..., "R2":...}.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = {"model": name,
           "cv_MAE": cv_df["MAE"].mean(), "cv_MAE_std": cv_df["MAE"].std(),
           "cv_R2": cv_df["R2"].mean(), "cv_R2_std": cv_df["R2"].std()}
    if ext_metrics:
        row["lopo_MAE"] = ext_metrics.get("MAE")
        row["lopo_R2"] = ext_metrics.get("R2")
    if os.path.exists(path):
        res = pd.read_csv(path)
        res = res[res["model"] != name]
        res = pd.concat([res, pd.DataFrame([row])], ignore_index=True)
    else:
        res = pd.DataFrame([row])
    res.to_csv(path, index=False)
    return res
