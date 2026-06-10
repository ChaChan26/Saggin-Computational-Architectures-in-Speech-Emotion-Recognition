import json
import os
import warnings
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import torch

warnings.filterwarnings("ignore")

CUDA_AVAILABLE = torch.cuda.is_available()
RANDOM_STATE = 42


def resolve_csv_path(base_dir: str) -> str:
    search_paths = [
        os.path.normpath(os.path.join(base_dir, "dataset", "all_emotions.csv")),
        os.path.normpath(os.path.join(base_dir, "all_emotions.csv")),
        os.path.normpath(os.path.join(os.path.dirname(base_dir), "dataset", "all_emotions.csv")),
        os.path.normpath(os.path.join(os.path.dirname(base_dir), "all_emotions.csv")),
    ]
    for path in search_paths:
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        f"Could not find all_emotions.csv. Searched in: {search_paths}"
    )


def load_best_params(base_dir: str) -> dict[str, Any]:
    project_root = os.path.dirname(base_dir) if os.path.basename(base_dir).lower() in ["src", "notebooks"] else base_dir
    params_path = os.path.join(project_root, "best_params.json")
    if os.path.isfile(params_path):
        with open(params_path, "r") as f:
            full_params = json.load(f)
            return full_params.get("lightgbm", {})
    print("best_params.json not found. Using default tuned parameters.")
    return {
        "n_estimators": 499,
        "max_depth": 11,
        "num_leaves": 67,
        "learning_rate": 0.24625126683753454,
        "subsample": 0.6909971314141472,
        "colsample_bytree": 0.7554088091076056,
    }


def ensure_figures_dir(base_dir: str) -> str:
    project_root = os.path.dirname(base_dir) if os.path.basename(base_dir).lower() in ["src", "notebooks"] else base_dir
    figures_dir = os.path.join(project_root, "figures")
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


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    csv_path = resolve_csv_path(base_dir)
    print(f"Loading dataset from: {csv_path}")
    figures_dir = ensure_figures_dir(base_dir)

    df = pd.read_csv(csv_path)
    
    # Target column extraction
    target_col = "label"
    if target_col not in df.columns and "Label" in df.columns:
        target_col = "Label"

    # Clean missing labels
    df_cleaned = df.dropna(subset=[target_col]).copy()
    df_cleaned = df_cleaned[df_cleaned[target_col].astype(str).str.strip().str.lower() != "nan"]

    # 26 features pruned via Random Forest feature importances
    feature_cols = [
        "F0_mean", "F0_std", "F0_range", "Energy_ mean", "Energy_ std", 
        "ZCR_mean", "ZCR_std", "Spectral_centroid_mean", "Spectral_centroid_std", 
        "Spectral_flux_mean", "MFCC_C0_mean", "MFCC_C1_mean", "MFCC_C2_mean", 
        "MFCC_C3_mean", "MFCC_C5_mean", "MFCC_C7_mean", "MFCC_C10_mean", 
        "MFCC_C0_std", "MFCC_C1_std", "MFCC_C2_std", "MFCC_C3_std", 
        "MFCC_C5_std", "MFCC_C7_std", "Delta_MFCC_C0_std", "Delta_MFCC_C2_std", 
        "Delta_MFCC_C3_std"
    ]

    # Convert all feature columns to numeric
    for col in feature_cols:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    X = df_cleaned[feature_cols].values
    y = df_cleaned[target_col].astype(str).str.strip().values

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Stratified Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    # Impute missing values with column median (Fitted ONLY on training data to prevent leakage)
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Scale features using StandardScaler (Fitted ONLY on training data to prevent leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    # Load tuned parameters
    lgb_params = load_best_params(base_dir)
    print(f"Training LightGBM model with parameters: {lgb_params}")

    # Build LightGBM classifier (with Try-Except fallback for GPU support)
    try:
        device_type = "gpu" if CUDA_AVAILABLE else "cpu"
        model = lgb.LGBMClassifier(
            **lgb_params,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
            objective="multiclass",
            num_class=len(encoder.classes_),
            device=device_type,
        )
        model.fit(X_train_scaled, y_train)
    except Exception as e:
        print(f"Warning: GPU training failed or GPU build not installed ({e}). Falling back to CPU...")
        model = lgb.LGBMClassifier(
            **lgb_params,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
            objective="multiclass",
            num_class=len(encoder.classes_),
            device="cpu",
        )
        model.fit(X_train_scaled, y_train)

    print("\n--- Model Training Complete ---")

    # Evaluate on test set
    y_pred = model.predict(X_test_scaled)
    test_f1 = f1_score(y_test, y_pred, average="weighted")
    test_kappa = cohen_kappa_score(y_test, y_pred)

    print("\n=== LightGBM Test Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    print(f"LightGBM Weighted F1-score: {test_f1:.4f}")
    print(f"LightGBM Cohen's Kappa:     {test_kappa:.4f}")

    # Save confusion matrix plot
    confusion_path = os.path.join(figures_dir, "lightgbm_only_confusion.png")
    plot_confusion(y_test, y_pred, encoder.classes_, "LightGBM Confusion Matrix", confusion_path)
    print(f"\nSaved confusion matrix to: {confusion_path}")


if __name__ == "__main__":
    main()
