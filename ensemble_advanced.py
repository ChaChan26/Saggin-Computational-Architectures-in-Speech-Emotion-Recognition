import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


RANDOM_STATE = 42
FEATURE_COLS = [
    "F0_mean",
    "F0_std",
    "F0_range",
    "Energy_ mean",
    "Energy_ std",
    "ZCR_mean",
    "ZCR_std",
    "Spectral_centroid_mean",
    "Spectral_centroid_std",
    "Spectral_flux_mean",
    "MFCC_C0_mean",
    "MFCC_C1_mean",
    "MFCC_C2_mean",
    "MFCC_C3_mean",
    "MFCC_C5_mean",
    "MFCC_C7_mean",
    "MFCC_C10_mean",
    "MFCC_C0_std",
    "MFCC_C1_std",
    "MFCC_C2_std",
    "MFCC_C3_std",
    "MFCC_C5_std",
    "MFCC_C7_std",
    "Delta_MFCC_C0_std",
    "Delta_MFCC_C2_std",
    "Delta_MFCC_C3_std",
]


def resolve_csv_path(base_dir: str) -> str:
    # Prefer the dataset folder, but keep the root file as a fallback for this repo layout.
    dataset_path = os.path.normpath(os.path.join(base_dir, "dataset", "all_emotions.csv"))
    if os.path.isfile(dataset_path):
        return dataset_path

    root_path = os.path.normpath(os.path.join(base_dir, "all_emotions.csv"))
    if os.path.isfile(root_path):
        return root_path

    raise FileNotFoundError(
        f"Could not find all_emotions.csv in '{dataset_path}' or '{root_path}'."
    )


def load_data(csv_path: str) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(csv_path)
    target_col = "label"
    if target_col not in df.columns and "Label" in df.columns:
        target_col = "Label"

    df_cleaned = df.dropna(subset=[target_col]).copy()
    df_cleaned = df_cleaned[df_cleaned[target_col].astype(str).str.strip().str.lower() != "nan"]

    for col in FEATURE_COLS:
        s = pd.to_numeric(df_cleaned[col], errors="coerce")
        s = s.replace([np.inf, -np.inf], np.nan)
        med = s.median()
        if pd.isna(med):
            med = 0.0
        df_cleaned[col] = s.fillna(med)

    return df_cleaned, target_col


def build_models(num_classes: int):
    xgb_model = xgb.XGBClassifier(
        n_estimators=468,
        max_depth=10,
        learning_rate=0.17549140891728818,
        subsample=0.9690394070981359,
        colsample_bytree=0.7725519831966194,
        gamma=1.0492767129301485e-08,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="mlogloss",
        objective="multi:softprob",
        num_class=num_classes,
    )

    lgb_model = lgb.LGBMClassifier(
        n_estimators=499,
        max_depth=11,
        num_leaves=67,
        learning_rate=0.24625126683753454,
        subsample=0.6909971314141472,
        colsample_bytree=0.7554088091076056,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
        objective="multiclass",
        num_class=num_classes,
    )

    return xgb_model, lgb_model


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

    df_cleaned, target_col = load_data(csv_path)
    X = df_cleaned[FEATURE_COLS].values
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

    xgb_model, lgb_model = build_models(len(encoder.classes_))
    xgb_model.fit(X_train_scaled, y_train)
    lgb_model.fit(X_train_scaled, y_train)

    xgb_pred = xgb_model.predict(X_test_scaled)
    lgb_pred = lgb_model.predict(X_test_scaled)

    xgb_f1 = f1_score(y_test, xgb_pred, average="weighted")
    xgb_kappa = cohen_kappa_score(y_test, xgb_pred)
    lgb_f1 = f1_score(y_test, lgb_pred, average="weighted")
    lgb_kappa = cohen_kappa_score(y_test, lgb_pred)

    xgb_proba = xgb_model.predict_proba(X_test_scaled)
    lgb_proba = lgb_model.predict_proba(X_test_scaled)

    ensemble_weights = np.array([0.8334582426563992, 0.842069322599305], dtype=float)
    ensemble_proba = (
        ensemble_weights[0] * xgb_proba + ensemble_weights[1] * lgb_proba
    ) / ensemble_weights.sum()
    ensemble_pred = np.argmax(ensemble_proba, axis=1)

    ensemble_f1 = f1_score(y_test, ensemble_pred, average="weighted")
    ensemble_kappa = cohen_kappa_score(y_test, ensemble_pred)
    ensemble_report = classification_report(y_test, ensemble_pred, target_names=encoder.classes_, output_dict=True)

    results = {
        "XGBoost": {"f1": xgb_f1, "kappa": xgb_kappa},
        "LightGBM": {"f1": lgb_f1, "kappa": lgb_kappa},
        "Ensemble": {"f1": ensemble_f1, "kappa": ensemble_kappa},
    }

    print("\n=== XGBoost ===")
    print(classification_report(y_test, xgb_pred, target_names=encoder.classes_))
    print(f"XGBoost weighted F1: {xgb_f1:.4f}")
    print(f"XGBoost Cohen Kappa: {xgb_kappa:.4f}")

    print("\n=== LightGBM ===")
    print(classification_report(y_test, lgb_pred, target_names=encoder.classes_))
    print(f"LightGBM weighted F1: {lgb_f1:.4f}")
    print(f"LightGBM Cohen Kappa: {lgb_kappa:.4f}")

    print("\n=== Weighted Ensemble (XGBoost + LightGBM) ===")
    print(classification_report(y_test, ensemble_pred, target_names=encoder.classes_))
    print(f"Ensemble weighted F1: {ensemble_f1:.4f}")
    print(f"Ensemble Cohen Kappa: {ensemble_kappa:.4f}")

    plot_confusion(y_test, xgb_pred, encoder.classes_, "XGBoost confusion matrix", os.path.join(figures_dir, "xgboost_confusion.png"))
    plot_confusion(y_test, lgb_pred, encoder.classes_, "LightGBM confusion matrix", os.path.join(figures_dir, "lightgbm_confusion.png"))
    plot_confusion(
        y_test,
        ensemble_pred,
        encoder.classes_,
        "Weighted ensemble confusion matrix",
        os.path.join(figures_dir, "ensemble_confusion.png"),
    )
    plot_model_comparison(results, os.path.join(figures_dir, "model_comparison.png"))
    plot_class_f1(ensemble_report, list(encoder.classes_), os.path.join(figures_dir, "ensemble_class_f1.png"))

    print(f"\nSaved plots to: {figures_dir}")


if __name__ == "__main__":
    main()
