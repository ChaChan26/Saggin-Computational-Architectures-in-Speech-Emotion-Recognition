import datetime
import json
import os
import sys
import warnings
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import optuna

import lightgbm as lgb
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# Make sure feature_engineering is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import engineer_features

RANDOM_STATE = 42
N_TRIALS = 500
N_JOBS_OPTUNA = 8
N_JOBS_LGB = 2


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


def ensure_figures_dir(base_dir: str) -> str:
    project_root = os.path.dirname(base_dir) if os.path.basename(base_dir).lower() in ["src", "notebooks"] else base_dir
    figures_dir = os.path.join(project_root, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    plt.figure(figsize=(9, 7))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("LightGBM Boosted Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_f1_scores(y_true, y_pred, labels, save_path):
    f1_per_class = f1_score(y_true, y_pred, average=None)
    plt.figure(figsize=(9, 5))
    sns.barplot(x=list(labels), y=f1_per_class, palette='Blues_d')
    plt.title("Boosted Per-Class F1-Score")
    plt.ylabel("F1-Score")
    plt.ylim(0, 1.0)
    for i, val in enumerate(f1_per_class):
        plt.text(i, val + 0.02, f"{val:.3f}", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def objective(trial: optuna.Trial, X_train_raw: np.ndarray, y_train: np.ndarray, imputer: SimpleImputer, class_weights: dict) -> float:
    # 1. Suggest parameters
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
        "max_depth": trial.suggest_int("max_depth", 5, 16),
        "num_leaves": trial.suggest_int("num_leaves", 31, 256),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 120),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        "random_state": RANDOM_STATE,
        "n_jobs": N_JOBS_LGB,
        "verbose": -1,
        "objective": "multiclass",
        "class_weight": class_weights,
    }
    
    # 2. Stratified 5-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_raw, y_train)):
        X_tr_raw, y_tr = X_train_raw[train_idx], y_train[train_idx]
        X_va_raw, y_va = X_train_raw[val_idx], y_train[val_idx]
        
        # Impute
        X_tr_imp = imputer.fit_transform(X_tr_raw)
        X_va_imp = imputer.transform(X_va_raw)
        
        # Feature Engineering
        X_tr_fe = engineer_features(X_tr_imp)
        X_va_fe = engineer_features(X_va_imp)
        
        # Scale
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_fe)
        X_va_scaled = scaler.transform(X_va_fe)
        
        # Model
        model = lgb.LGBMClassifier(**params)
        
        # Train with early stopping
        model.fit(
            X_tr_scaled, y_tr,
            eval_set=[(X_va_scaled, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        # Predict & Evaluate
        preds = model.predict(X_va_scaled)
        f1 = f1_score(y_va, preds, average="weighted")
        scores.append(f1)
        
        # Early pruning: if after 2 folds, the mean score is much worse than study best, stop CV early
        if fold >= 1:
            mean_so_far = np.mean(scores[:fold+1])
            # Prune trial if mean score is way below the current best (e.g. 0.03 worse)
            try:
                best_value = trial.study.best_value
                if mean_so_far < best_value - 0.03:
                    raise optuna.TrialPruned()
            except ValueError:
                # No best value yet
                pass
                
    return float(np.mean(scores))


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

    # Stratified Train-Test Split (80/20)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    print(f"Raw Train shape: {X_train_raw.shape}")
    print(f"Raw Test shape:  {X_test_raw.shape}")

    # Initialize SimpleImputer
    imputer = SimpleImputer(strategy="median")

    # Compute global class weights on training labels to handle class imbalance
    unique_classes = np.unique(y_train)
    global_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=y_train
    )
    global_weight_dict = dict(zip(unique_classes, global_weights))

    # Optuna tuning
    print(f"\n--- Starting Optuna Tuning ({N_TRIALS} trials, parallelized with {N_JOBS_OPTUNA} workers) ---")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Using local SQLite database to allow crash recovery / inspectability
    db_path = os.path.join(project_root, "optuna_study.db")
    study = optuna.create_study(
        study_name="lgbm_boosted_tuning",
        storage=f"sqlite:///{db_path}",
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1)
    )
    
    study.optimize(
        lambda t: objective(t, X_train_raw, y_train, imputer, global_weight_dict),
        n_trials=N_TRIALS,
        n_jobs=N_JOBS_OPTUNA,
        show_progress_bar=True
    )
    
    print(f"Best CV Weighted F1 score: {study.best_value:.6f}")
    best_params = study.best_params
    print(f"Best parameters: {best_params}")

    # Save to best_params.json
    params_file_path = os.path.join(project_root, "best_params.json")
    try:
        if os.path.exists(params_file_path):
            with open(params_file_path, "r") as f:
                full_params = json.load(f)
        else:
            full_params = {}
        full_params["lightgbm_boosted"] = best_params
        with open(params_file_path, "w") as f:
            json.dump(full_params, f, indent=4)
        print(f"Saved best parameters to {params_file_path}")
    except Exception as e:
        print(f"Error saving parameters: {e}")

    # --- Train final model on full 80% train set ---
    print("\n--- Training Final Model on Full Training Set ---")
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_test_imp = imputer.transform(X_test_raw)

    X_train_fe = engineer_features(X_train_imp)
    X_test_fe = engineer_features(X_test_imp)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_fe)
    X_test_scaled = scaler.transform(X_test_fe)

    final_params = best_params.copy()
    final_params.update({
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
        "objective": "multiclass",
        "class_weight": global_weight_dict,
    })

    model = lgb.LGBMClassifier(**final_params)
    model.fit(X_train_scaled, y_train)

    # Evaluate on unseen test set
    lgb_pred = model.predict(X_test_scaled)
    test_f1 = f1_score(y_test, lgb_pred, average="weighted")
    test_kappa = cohen_kappa_score(y_test, lgb_pred)

    print("\n=== LightGBM Boosted Test Classification Report ===")
    print(classification_report(y_test, lgb_pred, target_names=encoder.classes_))
    print(f"LightGBM Boosted Test Weighted F1: {test_f1:.8f}")
    print(f"LightGBM Boosted Test Cohen's Kappa: {test_kappa:.8f}")

    # Check historical best from log or review report
    current_best_f1 = 0.88083849  # Historical best
    print(f"Historical Best F1: {current_best_f1:.8f}")

    # Save visualization plots
    confusion_path = os.path.join(figures_dir, "lightgbm_boosted_confusion.png")
    plot_confusion_matrix(y_test, lgb_pred, encoder.classes_, confusion_path)
    # Also save to the standard path for pipeline compatibility
    plot_confusion_matrix(y_test, lgb_pred, encoder.classes_, os.path.join(figures_dir, "lightgbm_confusion.png"))
    print(f"\nSaved confusion matrix plots")

    f1_path = os.path.join(figures_dir, "lightgbm_boosted_class_f1.png")
    plot_f1_scores(y_test, lgb_pred, encoder.classes_, f1_path)
    # Also save to the standard path for pipeline compatibility
    plot_f1_scores(y_test, lgb_pred, encoder.classes_, os.path.join(figures_dir, "lightgbm_class_f1.png"))
    print(f"Saved class F1 plots")

    # Serializing production artifacts if it improves or matches high performance
    print("\n--- Serializing production artifacts for standalone LightGBM model ---")
    best_model_dir = os.path.join(project_root, 'best_model')
    os.makedirs(best_model_dir, exist_ok=True)

    # Save to best_model folder (overwrites if user desires or if it is our new best)
    joblib.dump(model,   os.path.join(best_model_dir, 'ser_optuna_lightgbm.joblib'))
    joblib.dump(imputer, os.path.join(best_model_dir, 'ser_optuna_imputer.joblib'))
    joblib.dump(scaler,  os.path.join(best_model_dir, 'ser_optuna_scaler.joblib'))
    joblib.dump(encoder, os.path.join(best_model_dir, 'ser_optuna_encoder.joblib'))
    print(f"Saved model, imputer, scaler, and encoder to {best_model_dir}")

    # Save Standalone Performance Reports
    def write_report(path):
        with open(path, 'w') as f:
            f.write('=== Speech Emotion Recognition Standalone Boosted LightGBM Training Report ===\n\n')
            f.write('=== LightGBM Weighted F1 ===\n')
            f.write(f'{test_f1:.8f}\n\n')
            f.write('=== LightGBM Cohen Kappa ===\n')
            f.write(f'{test_kappa:.8f}\n\n')
            f.write('=== LightGBM Classification Report ===\n')
            f.write(classification_report(y_test, lgb_pred, target_names=encoder.classes_))
            
    write_report(os.path.join(project_root, 'lightgbm_boosted_report.txt'))
    write_report(os.path.join(project_root, 'lightgbm_standalone_report.txt'))
    print(f'Saved performance reports')

    # Append summary of this run to persistent training history log
    history_path = os.path.join(project_root, 'lightgbm_history.log')
    with open(history_path, 'a') as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] Device: cpu (BOOSTED WITH FEATURE ENGINEERING)\n")
        f.write(f"  Weighted F1: {test_f1:.8f}\n")
        f.write(f"  Cohen Kappa: {test_kappa:.8f}\n")
        f.write(f"  Params: {json.dumps(best_params)}\n")
        f.write("-" * 80 + "\n")
    print(f'Appended training summary to {history_path}')


if __name__ == "__main__":
    main()
