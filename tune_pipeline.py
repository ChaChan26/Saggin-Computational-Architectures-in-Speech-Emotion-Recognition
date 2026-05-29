import json
import os
import warnings
from typing import Any

import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score
import torch

warnings.filterwarnings("ignore")

CUDA_AVAILABLE = torch.cuda.is_available()


RANDOM_STATE = 42


def resolve_csv_path(base_dir: str) -> str:
    dataset_path = os.path.normpath(os.path.join(base_dir, "dataset", "all_emotions.csv"))
    if os.path.isfile(dataset_path):
        return dataset_path

    root_path = os.path.normpath(os.path.join(base_dir, "all_emotions.csv"))
    if os.path.isfile(root_path):
        return root_path

    raise FileNotFoundError(
        f"Could not find all_emotions.csv in '{dataset_path}' or '{root_path}'."
    )


def load_data(csv_path: str) -> tuple[pd.DataFrame, str, list[str]]:
    df = pd.read_csv(csv_path)
    target_col = "label"
    if target_col not in df.columns and "Label" in df.columns:
        target_col = "Label"

    df_cleaned = df.dropna(subset=[target_col]).copy()
    df_cleaned = df_cleaned[df_cleaned[target_col].astype(str).str.strip().str.lower() != "nan"]

    # Use all columns except target as features
    feature_cols = [col for col in df_cleaned.columns if col not in [target_col]]

    for col in feature_cols:
        s = pd.to_numeric(df_cleaned[col], errors="coerce")
        s = s.replace([np.inf, -np.inf], np.nan)
        med = s.median()
        if pd.isna(med):
            med = 0.0
        df_cleaned[col] = s.fillna(med)

    return df_cleaned, target_col, feature_cols



def objective_xgb(trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray, skf: StratifiedKFold) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 11),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "eval_metric": "mlogloss",
        "device": "cuda" if CUDA_AVAILABLE else "cpu",
    }
    
    scores = []
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_va, y_va = X_train[val_idx], y_train[val_idx]

        clf = xgb.XGBClassifier(**params)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_va)
        scores.append(f1_score(y_va, preds, average="weighted"))
        
    return float(np.mean(scores))


def objective_lgb(trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray, skf: StratifiedKFold) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
        "device": "gpu" if CUDA_AVAILABLE else "cpu",
    }
    
    scores = []
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_va, y_va = X_train[val_idx], y_train[val_idx]

        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_va)
        scores.append(f1_score(y_va, preds, average="weighted"))
        
    return float(np.mean(scores))


def objective_cb(trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray, skf: StratifiedKFold) -> float:
    params = {
        "iterations": trial.suggest_int("iterations", 100, 500),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-1, 10.0, log=True),
        "loss_function": "MultiClass",
        "random_seed": RANDOM_STATE,
        "thread_count": -1,
        "verbose": False,
        "task_type": "GPU" if CUDA_AVAILABLE else "CPU",
    }
    
    scores = []
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_va, y_va = X_train[val_idx], y_train[val_idx]

        clf = CatBoostClassifier(**params)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_va).flatten()
        scores.append(f1_score(y_va, preds, average="weighted"))
        
    return float(np.mean(scores))


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = resolve_csv_path(base_dir)
    
    df_cleaned, target_col, feature_cols = load_data(csv_path)
    X = df_cleaned[feature_cols].values
    y = df_cleaned[target_col].astype(str).str.strip().values

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, _, y_train, _ = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    best_params = {}

    print("--- Starting XGBoost Hyperparameter Tuning (15 trials) ---")
    study_xgb = optuna.create_study(direction="maximize")
    study_xgb.optimize(
        lambda t: objective_xgb(t, X_train_scaled, y_train, skf),
        n_trials=15,
        n_jobs=3,
        show_progress_bar=True,
    )
    print(f"Best XGBoost Mean CV F1: {study_xgb.best_value:.4f}")
    best_params["xgboost"] = study_xgb.best_params

    print("\n--- Starting LightGBM Hyperparameter Tuning (15 trials) ---")
    study_lgb = optuna.create_study(direction="maximize")
    study_lgb.optimize(
        lambda t: objective_lgb(t, X_train_scaled, y_train, skf),
        n_trials=15,
        n_jobs=3,
        show_progress_bar=True,
    )
    print(f"Best LightGBM Mean CV F1: {study_lgb.best_value:.4f}")
    best_params["lightgbm"] = study_lgb.best_params

    print("\n--- Starting CatBoost Hyperparameter Tuning (10 trials) ---")
    study_cb = optuna.create_study(direction="maximize")
    study_cb.optimize(
        lambda t: objective_cb(t, X_train_scaled, y_train, skf),
        n_trials=10,
        n_jobs=3,
        show_progress_bar=True,
    )
    print(f"Best CatBoost Mean CV F1: {study_cb.best_value:.4f}")
    best_params["catboost"] = study_cb.best_params

    out_path = os.path.join(base_dir, "best_params.json")
    with open(out_path, "w") as f:
        json.dump(best_params, f, indent=4)

    print(f"\nOptimization complete! Parameters saved to: {out_path}")


if __name__ == "__main__":
    main()
