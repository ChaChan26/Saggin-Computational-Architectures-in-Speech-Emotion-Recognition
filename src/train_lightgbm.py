import json
import os
import warnings
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import lightgbm as lgb
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
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


def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    plt.figure(figsize=(9, 7))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("LightGBM Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_f1_scores(y_true, y_pred, labels, save_path):
    f1_per_class = f1_score(y_true, y_pred, average=None)
    plt.figure(figsize=(9, 5))
    sns.barplot(x=list(labels), y=f1_per_class, palette='Blues_d')
    plt.title("Per-Class F1-Score")
    plt.ylabel("F1-Score")
    plt.ylim(0, 1.0)
    for i, val in enumerate(f1_per_class):
        plt.text(i, val + 0.02, f"{val:.3f}", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    project_root = os.path.dirname(base_dir) if os.path.basename(base_dir).lower() in ["src", "notebooks"] else base_dir
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

    FEATURE_COLS = [col for col in df_cleaned.columns if col not in [target_col]]
    print(f"Number of features selected dynamically: {len(FEATURE_COLS)}")

    # Convert feature columns to numeric, transforming infinite values to NaN
    for col in FEATURE_COLS:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    X = df_cleaned[FEATURE_COLS].values
    y = df_cleaned[target_col].astype(str).str.strip().values

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Stratified Train-Test Split (90/10)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.10,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    print(f"Raw Train shape: {X_train.shape}")
    print(f"Raw Test shape:  {X_test.shape}")

    # 1. Median Imputation (Fitted ONLY on training data to prevent leakage)
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # 2. StandardScaler (Fitted ONLY on training data to prevent leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    # Load tuned parameters
    lgb_params = load_best_params(base_dir)
    print(f"Training LightGBM model with parameters: {lgb_params}")

    # Compute global class weights on training labels to handle class imbalance
    from sklearn.utils.class_weight import compute_class_weight
    unique_classes = np.unique(y_train)
    global_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=y_train
    )
    global_weight_dict = dict(zip(unique_classes, global_weights))

    # Build LightGBM classifier (with Try-Except fallback for GPU support)
    device_type = "gpu" if CUDA_AVAILABLE else "cpu"
    try:
        model = lgb.LGBMClassifier(
            **lgb_params,
            class_weight=global_weight_dict,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
            objective="multiclass",
            num_class=len(encoder.classes_),
            device=device_type,
        )
        model.fit(X_train_scaled, y_train)
    except Exception as e:
        print(f"Warning: GPU training failed ({e}). Falling back to CPU...")
        model = lgb.LGBMClassifier(
            **lgb_params,
            class_weight=global_weight_dict,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
            objective="multiclass",
            num_class=len(encoder.classes_),
            device="cpu",
        )
        model.fit(X_train_scaled, y_train)

    print("\n--- Model Training Complete ---")

    # Evaluate on unseen test set
    lgb_pred = model.predict(X_test_scaled)
    test_f1 = f1_score(y_test, lgb_pred, average="weighted")
    test_kappa = cohen_kappa_score(y_test, lgb_pred)

    print("\n=== LightGBM Test Classification Report ===")
    print(classification_report(y_test, lgb_pred, target_names=encoder.classes_))
    print(f"LightGBM Test Weighted F1: {test_f1:.8f}")
    print(f"LightGBM Test Cohen's Kappa: {test_kappa:.8f}")

    # Save visualization plots
    confusion_path = os.path.join(figures_dir, "lightgbm_confusion.png")
    plot_confusion_matrix(y_test, lgb_pred, encoder.classes_, confusion_path)
    print(f"\nSaved confusion matrix plot to: {confusion_path}")

    f1_path = os.path.join(figures_dir, "lightgbm_class_f1.png")
    plot_f1_scores(y_test, lgb_pred, encoder.classes_, f1_path)
    print(f"Saved class F1 plot to: {f1_path}")

    # Save Trained Model, Preprocessors, and Encoder
    print("\n--- Serializing production artifacts for standalone LightGBM model ---")
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)

    saved_files = []
    def save_asset(obj, name):
        path = os.path.join(models_dir, name)
        joblib.dump(obj, path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        saved_files.append(f"  - {name} ({size_mb:.2f} MB)")

    save_asset(model,   'ser_lgb_standalone_model.joblib')
    save_asset(imputer, 'ser_lgb_standalone_imputer.joblib')
    save_asset(scaler,  'ser_lgb_standalone_scaler.joblib')
    save_asset(encoder, 'ser_lgb_standalone_encoder.joblib')

    for f_saved in saved_files:
        print(f_saved)

    # Save Standalone Performance Report
    report_path = os.path.join(project_root, 'lightgbm_report.txt')
    with open(report_path, 'w') as f:
        f.write('=== Speech Emotion Recognition Standalone LightGBM Training Report ===\n\n')
        f.write('=== LightGBM Weighted F1 ===\n')
        f.write(f'{test_f1:.8f}\n\n')
        f.write('=== LightGBM Cohen Kappa ===\n')
        f.write(f'{test_kappa:.8f}\n\n')
        f.write('=== LightGBM Classification Report ===\n')
        f.write(classification_report(y_test, lgb_pred, target_names=encoder.classes_))

    print(f'Saved performance report to {report_path}')

    # Append summary of this run to a persistent training history log
    import datetime
    history_path = os.path.join(project_root, 'lightgbm_history.log')
    with open(history_path, 'a') as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] Device: {device_type}\n")
        f.write(f"  Weighted F1: {test_f1:.8f}\n")
        f.write(f"  Cohen Kappa: {test_kappa:.8f}\n")
        f.write(f"  Params: {json.dumps(lgb_params)}\n")
        f.write("-" * 80 + "\n")
    print(f'Appended training summary to {history_path}')


if __name__ == "__main__":
    main()
