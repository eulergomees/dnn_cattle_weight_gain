"""Shared data preparation for the model-comparison notebooks.

Every model (linear, trees, MLP, ...) is evaluated on IDENTICAL splits, folds
and metrics so the comparison is fair. Import from a notebook run at the repo
root with:

    import sys; sys.path.append("support_scripts")
    import data_prep as dp
    d = dp.get_data()
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "data/cattle_dataset.csv"
TARGET = "adg_kg_day"
LEAKAGE = "days_on_pasture"      # denominator of the ADG formula -> excluded (leakage)
SEED = 42
N_SPLITS = 10
TEST_SIZE = 0.15
RESULTS_PATH = "results/model_comparison.csv"


def load_raw(path=DATA_PATH):
    df = pd.read_csv(path)
    if "origem" in df.columns:                     # drop tag column if present
        df = df.drop(columns=["origem"])
    return df


def feature_columns(df):
    """Return (feature_cols, scale_cols) after dropping constants, leakage, target."""
    const = [c for c in df.columns if df[c].nunique() == 1]
    num = [c for c in df.select_dtypes("number").columns if c not in const]
    bin_ = [c for c in num if set(df[c].dropna().unique()) <= {0, 1}]
    cat = [c for c in num if c not in bin_ and c != TARGET and df[c].nunique() <= 6]
    cont = [c for c in num if c not in bin_ and c not in cat and c != TARGET]
    feature_cols = [c for c in df.columns if c not in set(const) | {LEAKAGE, TARGET}]
    scale_cols = [c for c in (cont + cat) if c in feature_cols]
    return feature_cols, scale_cols


def get_data():
    """Fixed hold-out split shared by every notebook.

    Returns a dict: X_dev, X_test, y_dev, y_test, feature_cols, scale_cols.
    """
    df = load_raw()
    feature_cols, scale_cols = feature_columns(df)
    X, y = df[feature_cols], df[TARGET]
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED)
    X_dev = X_dev.reset_index(drop=True); y_dev = y_dev.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True); y_test = y_test.reset_index(drop=True)
    return dict(X_dev=X_dev, X_test=X_test, y_dev=y_dev, y_test=y_test,
                feature_cols=feature_cols, scale_cols=scale_cols)


def get_cv():
    """The single 10-fold splitter used by all models."""
    return KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)


def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {"MAE":  mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2":   r2_score(y_true, y_pred)}


def baseline_mae(y_train, y_eval):
    """MAE of the naive 'predict the training mean' baseline."""
    return mean_absolute_error(y_eval, np.full(len(y_eval), np.mean(y_train)))


def summarize(fold_metrics):
    """fold_metrics: list of metric dicts -> (per-fold DataFrame, mean/std DataFrame)."""
    cv = pd.DataFrame(fold_metrics)
    return cv, cv.agg(["mean", "std"]).round(4)


def save_result(name, cv_df, test_metrics, path=RESULTS_PATH):
    """Append/overwrite this model's scores in the shared comparison table."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = {
        "model":      name,
        "cv_MAE":     cv_df["MAE"].mean(),
        "cv_MAE_std": cv_df["MAE"].std(),
        "cv_R2":      cv_df["R2"].mean(),
        "cv_R2_std":  cv_df["R2"].std(),
        "test_MAE":   test_metrics["MAE"],
        "test_R2":    test_metrics["R2"],
    }
    if os.path.exists(path):
        res = pd.read_csv(path)
        res = res[res["model"] != name]            # overwrite previous entry
        res = pd.concat([res, pd.DataFrame([row])], ignore_index=True)
    else:
        res = pd.DataFrame([row])
    res.to_csv(path, index=False)
    return res
