import json
import os
import warnings
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
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



def load_best_params(base_dir: str) -> dict[str, dict[str, Any]]:
    params_path = os.path.join(base_dir, "best_params.json")
    if os.path.isfile(params_path):
        with open(params_path, "r") as f:
            return json.load(f)
    print("best_params.json not found. Using default parameters.")
    return {}


def build_models(num_classes: int, best_params: dict[str, dict[str, Any]]):
    xgb_params = best_params.get("xgboost", {
        "n_estimators": 468,
        "max_depth": 10,
        "learning_rate": 0.17549140891728818,
        "subsample": 0.9690394070981359,
        "colsample_bytree": 0.7725519831966194,
        "gamma": 1.0492767129301485e-08,
    })
    xgb_model = xgb.XGBClassifier(
        **xgb_params,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="mlogloss",
        objective="multi:softprob",
        num_class=num_classes,
        device="cuda" if CUDA_AVAILABLE else "cpu",
    )

    lgb_params = best_params.get("lightgbm", {
        "n_estimators": 499,
        "max_depth": 11,
        "num_leaves": 67,
        "learning_rate": 0.24625126683753454,
        "subsample": 0.6909971314141472,
        "colsample_bytree": 0.7554088091076056,
    })
    lgb_model = lgb.LGBMClassifier(
        **lgb_params,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
        objective="multiclass",
        num_class=num_classes,
        device="gpu" if CUDA_AVAILABLE else "cpu",
    )

    cb_params = best_params.get("catboost", {
        "iterations": 500,
        "depth": 8,
        "learning_rate": 0.15,
        "l2_leaf_reg": 3.0,
    })
    cb_model = CatBoostClassifier(
        **cb_params,
        loss_function="MultiClass",
        random_seed=RANDOM_STATE,
        thread_count=-1,
        verbose=False,
        task_type="GPU" if CUDA_AVAILABLE else "CPU",
    )

    return xgb_model, lgb_model, cb_model


def optimize_ensemble_weights(
    xgb_oof: np.ndarray,
    lgb_oof: np.ndarray,
    cb_oof: np.ndarray,
    y_train: np.ndarray
) -> np.ndarray:
    best_f1 = 0.0
    best_weights = np.array([1.0, 1.0, 1.0]) / 3.0
    
    # Grid search over possible weight combinations
    for w_xgb in np.linspace(0, 1, 11):
        for w_lgb in np.linspace(0, 1 - w_xgb, 11):
            w_cb = 1.0 - w_xgb - w_lgb
            if w_cb < 0:
                continue
                
            weights = np.array([w_xgb, w_lgb, w_cb])
            val_proba = (
                w_xgb * xgb_oof +
                w_lgb * lgb_oof +
                w_cb * cb_oof
            )
            preds = np.argmax(val_proba, axis=1)
            score = f1_score(y_train, preds, average="weighted")
            
            if score > best_f1:
                best_f1 = score
                best_weights = weights
                
    return best_weights


def ensure_figures_dir(base_dir: str) -> str:
    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


def plot_confusion(y_true, y_pred, labels, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_model_comparison(results, save_path):
    names = list(results.keys())
    f1_scores = [results[name]["f1"] for name in names]
    kappas = [results[name]["kappa"] for name in names]

    x = np.arange(len(names))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, f1_scores, width, label="Weighted F1")
    plt.bar(x + width / 2, kappas, width, label="Cohen Kappa")
    plt.xticks(x, names)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Model comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_class_f1(report_dict, labels, save_path):
    class_scores = [report_dict[label]["f1-score"] for label in labels]
    plt.figure(figsize=(8, 5))
    sns.barplot(x=labels, y=class_scores, color="steelblue")
    plt.ylim(0, 1)
    plt.ylabel("F1-score")
    plt.title("Ensemble per-class F1")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = resolve_csv_path(base_dir)
    print(f"Loading dataset from: {csv_path}")
    figures_dir = ensure_figures_dir(base_dir)

    df_cleaned, target_col, feature_cols = load_data(csv_path)
    X = df_cleaned[feature_cols].values
    y = df_cleaned[target_col].astype(str).str.strip().values

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    best_params = load_best_params(base_dir)
    xgb_model, lgb_model, cb_model = build_models(len(encoder.classes_), best_params)

    # 1. 3-Fold Stratified CV to get out-of-fold validation predictions for weights optimization
    print("\n--- Training Out-Of-Fold for ensemble weight optimization ---")
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    xgb_oof = np.zeros((len(X_train_scaled), len(encoder.classes_)))
    lgb_oof = np.zeros((len(X_train_scaled), len(encoder.classes_)))
    cb_oof = np.zeros((len(X_train_scaled), len(encoder.classes_)))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train)):
        print(f"Fold {fold + 1}...")
        X_tr, y_tr = X_train_scaled[train_idx], y_train[train_idx]
        X_va, y_va = X_train_scaled[val_idx], y_train[val_idx]

        # Train fold models
        xgb_m, lgb_m, cb_m = build_models(len(encoder.classes_), best_params)
        xgb_m.fit(X_tr, y_tr)
        lgb_m.fit(X_tr, y_tr)
        cb_m.fit(X_tr, y_tr)

        xgb_oof[val_idx] = xgb_m.predict_proba(X_va)
        lgb_oof[val_idx] = lgb_m.predict_proba(X_va)
        cb_oof[val_idx] = cb_m.predict_proba(X_va)

    print("\nOptimizing ensemble weights...")
    best_weights = optimize_ensemble_weights(xgb_oof, lgb_oof, cb_oof, y_train)
    print(f"Optimized weights: XGBoost={best_weights[0]:.2f}, LightGBM={best_weights[1]:.2f}, CatBoost={best_weights[2]:.2f}")

    # 2. Fit on the full training set
    print("\n--- Fitting final models on full training set ---")
    xgb_model.fit(X_train_scaled, y_train)
    lgb_model.fit(X_train_scaled, y_train)
    cb_model.fit(X_train_scaled, y_train)

    xgb_pred = xgb_model.predict(X_test_scaled)
    lgb_pred = lgb_model.predict(X_test_scaled)
    cb_pred = cb_model.predict(X_test_scaled).flatten()

    xgb_proba = xgb_model.predict_proba(X_test_scaled)
    lgb_proba = lgb_model.predict_proba(X_test_scaled)
    cb_proba = cb_model.predict_proba(X_test_scaled)

    # Soft Voting Ensemble
    ensemble_proba = (
        best_weights[0] * xgb_proba +
        best_weights[1] * lgb_proba +
        best_weights[2] * cb_proba
    )
    ensemble_pred = np.argmax(ensemble_proba, axis=1)

    # Calculate metrics
    xgb_f1 = f1_score(y_test, xgb_pred, average="weighted")
    xgb_kappa = cohen_kappa_score(y_test, xgb_pred)

    lgb_f1 = f1_score(y_test, lgb_pred, average="weighted")
    lgb_kappa = cohen_kappa_score(y_test, lgb_pred)

    cb_f1 = f1_score(y_test, cb_pred, average="weighted")
    cb_kappa = cohen_kappa_score(y_test, cb_pred)

    ensemble_f1 = f1_score(y_test, ensemble_pred, average="weighted")
    ensemble_kappa = cohen_kappa_score(y_test, ensemble_pred)
    ensemble_report = classification_report(y_test, ensemble_pred, target_names=encoder.classes_, output_dict=True)

    results = {
        "XGBoost": {"f1": xgb_f1, "kappa": xgb_kappa},
        "LightGBM": {"f1": lgb_f1, "kappa": lgb_kappa},
        "CatBoost": {"f1": cb_f1, "kappa": cb_kappa},
        "Ensemble": {"f1": ensemble_f1, "kappa": ensemble_kappa},
    }

    print("\n=== XGBoost ===")
    print(classification_report(y_test, xgb_pred, target_names=encoder.classes_))
    print(f"XGBoost weighted F1: {xgb_f1:.4f}")

    print("\n=== LightGBM ===")
    print(classification_report(y_test, lgb_pred, target_names=encoder.classes_))
    print(f"LightGBM weighted F1: {lgb_f1:.4f}")

    print("\n=== CatBoost ===")
    print(classification_report(y_test, cb_pred, target_names=encoder.classes_))
    print(f"CatBoost weighted F1: {cb_f1:.4f}")

    print("\n=== Optimally Weighted Ensemble (XGB + LGB + CB) ===")
    print(classification_report(y_test, ensemble_pred, target_names=encoder.classes_))
    print(f"Ensemble weighted F1: {ensemble_f1:.4f}")
    print(f"Ensemble Cohen Kappa: {ensemble_kappa:.4f}")

    # Plot metrics
    plot_confusion(y_test, xgb_pred, encoder.classes_, "XGBoost confusion matrix", os.path.join(figures_dir, "opt_xgboost_confusion.png"))
    plot_confusion(y_test, lgb_pred, encoder.classes_, "LightGBM confusion matrix", os.path.join(figures_dir, "opt_lightgbm_confusion.png"))
    plot_confusion(y_test, cb_pred, encoder.classes_, "CatBoost confusion matrix", os.path.join(figures_dir, "opt_catboost_confusion.png"))
    plot_confusion(y_test, ensemble_pred, encoder.classes_, "Optimally weighted ensemble confusion matrix", os.path.join(figures_dir, "opt_ensemble_confusion.png"))
    
    plot_model_comparison(results, os.path.join(figures_dir, "opt_model_comparison.png"))
    plot_class_f1(ensemble_report, list(encoder.classes_), os.path.join(figures_dir, "opt_ensemble_class_f1.png"))

    print(f"\nSaved plots to: {figures_dir}")


if __name__ == "__main__":
    main()
