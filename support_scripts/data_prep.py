"""Pré-processamento compartilhado (schema por-animal v3) dos notebooks de modelo.

Todo modelo (linear, árvores, MLP, ...) usa os MESMOS dados, folds e métricas —
comparação justa. Import a partir da raiz do repositório:

    import sys; sys.path.append("support_scripts")
    import data_prep as dp
    d = dp.get_data()                 # X, y, groups, feature_cols, scale_cols, df

Alvo: `gmd_kg_dia`. Split: 5 folds DISJUNTOS com **N_TEST=49 animais de teste** cada
(estratificado por fazenda, proporcional ao tamanho de cada uma) para comparar/tunar +
leave-one-property-out (3 fazendas) para robustez/validade externa.
Scaling NÃO é feito aqui (cada notebook aplica StandardScaler fit só no treino).
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "data/dataset_por_animal_modelo_v3.csv"
TARGET = "gmd_kg_dia"
GROUP = "id_propriedade"
# fora dos preditores: identificação, vazamento do alvo e multicolinear (r=0.97 c/ dias)
DROP = ["id_animal", "id_propriedade", "data_entrada", "data_saida",
        "peso_saida_kg", "precipitacao_acumulada_mm"]
SEED = 42
N_TEST = 49          # amostras de teste por fold (fixo), distribuídas pela % de cada fazenda
N_SPLITS = 5         # folds disjuntos de 49 = 245 -> todo animal é testado exatamente 1x
RESULTS_PATH = "results/model_comparison.csv"

# escalas do jitter (ruído de medição nas FEATURES) — só aplicado no treino, pós-split
NOISE = {"peso_entrada_kg": 2.5, "temperatura_media_c": 0.5}


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


class FarmStratifiedSplit:
    """Folds DISJUNTOS com EXATAMENTE `n_test` animais de teste, estratificados por fazenda.

    Cada fazenda é embaralhada (seed fixa) e distribuída em rodízio pelos folds, com o
    rodízio continuando de uma fazenda para a próxima. Resultado: cada fazenda entra em
    cada fold com a sua % (±1 animal) e os folds ficam com tamanhos idênticos. Ex.:
    245 animais (113/71/61), 5 folds -> 49 por fold (23/14/12, 22/15/12, 22/14/13...).
    Partição: cada animal é testado 1x (`oof[va] = preds` vale). Exige
    `n_splits * n_test == n_animais`; se o dataset crescer, ajuste `N_TEST`.
    Uso igual ao do KFold: `for tr, va in dp.get_cv().split(df)` (df com `id_propriedade`).
    """
    def __init__(self, n_splits=N_SPLITS, n_test=N_TEST, seed=SEED):
        self.n_splits, self.n_test, self.seed = n_splits, n_test, seed

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        farm = np.asarray(X[GROUP] if hasattr(X, "columns") else groups)
        if len(farm) != self.n_splits * self.n_test:
            raise ValueError(f"{len(farm)} animais != {self.n_splits} folds x {self.n_test} de teste; "
                             f"ajuste N_TEST/N_SPLITS em data_prep.")
        rng = np.random.default_rng(self.seed)
        order = np.concatenate([rng.permutation(np.flatnonzero(farm == n))
                                for n in np.unique(farm)])
        fold = np.empty(len(farm), dtype=int)
        fold[order] = np.arange(len(farm)) % self.n_splits       # rodízio contínuo
        for k in range(self.n_splits):
            te = np.flatnonzero(fold == k)
            yield np.flatnonzero(fold != k), te


def get_cv():
    """Folds de comparação/tuning: 5 folds disjuntos de N_TEST=49 animais, por fazenda."""
    return FarmStratifiedSplit()


def leave_one_property_out(groups):
    """Gera (train_idx, test_idx, propriedade_de_teste) para cada propriedade.

    Robustez/validade externa: treina numa fazenda, testa na outra.
    """
    g = np.asarray(groups)
    idx = np.arange(len(g))
    for p in pd.unique(g):
        yield idx[g != p], idx[g == p], p


def augment_train(df_train, k=2, seed=SEED):
    """Jitter (ruído de medição) **feature-only** — expande SÓ o treino com `k` cópias.

    Perturba as FEATURES contínuas (`peso_entrada_kg`, `temperatura_media_c`) com ruído
    de medição, **mantendo o alvo `gmd_kg_dia` REAL**. NUNCA chamar em val/test.

    Por que feature-only: rederivar o alvo a partir do `peso_entrada` — que também é
    feature e entra em `gmd=(peso_saida−peso_entrada)/dias` — acopla feature↔alvo; um
    diagnóstico mostrou que isso ensina a rede a derivada exata ∂gmd/∂peso_entrada e
    **infla o R² artificialmente** (0,67→0,81 com ruído crescente), sem sinal preditivo
    novo. A versão feature-only é a honesta.
    """
    rng = np.random.default_rng(seed)
    copies = [df_train]
    for _ in range(k):
        a = df_train.copy()
        a["peso_entrada_kg"] = (a["peso_entrada_kg"]
                                + rng.normal(0, NOISE["peso_entrada_kg"], len(a))).clip(lower=1)
        a["temperatura_media_c"] = (a["temperatura_media_c"]
                                    + rng.normal(0, NOISE["temperatura_media_c"], len(a)))
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

    cv_df: métricas por fold de `get_cv()`. ext_metrics (opcional): dict com a
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
