import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy.optimize import minimize

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


def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    plt.figure(figsize=(9, 7))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("LightGBM Threshold-Optimized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_f1_scores(y_true, y_pred, labels, save_path):
    f1_per_class = f1_score(y_true, y_pred, average=None)
    plt.figure(figsize=(9, 5))
    sns.barplot(x=list(labels), y=f1_per_class, palette='Blues_d')
    plt.title("Threshold-Optimized Per-Class F1-Score")
    plt.ylabel("F1-Score")
    plt.ylim(0, 1.0)
    for i, val in enumerate(f1_per_class):
        plt.text(i, val + 0.02, f"{val:.3f}", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    project_root = os.path.dirname(base_dir) if os.path.basename(base_dir).lower() in ["src", "notebooks"] else base_dir
    csv_path = resolve_csv_path(base_dir)
    figures_dir = os.path.join(project_root, "figures")
    best_model_dir = os.path.join(project_root, "best_model")

    print("=" * 80)
    print("      SPEECH EMOTION RECOGNITION - DECISION THRESHOLD CALIBRATION")
    print("=" * 80)

    # 1. Load trained model & preprocessors
    print("Loading serialized production artifacts...")
    model = joblib.load(os.path.join(best_model_dir, 'ser_optuna_lightgbm.joblib'))
    imputer = joblib.load(os.path.join(best_model_dir, 'ser_optuna_imputer.joblib'))
    scaler = joblib.load(os.path.join(best_model_dir, 'ser_optuna_scaler.joblib'))
    encoder = joblib.load(os.path.join(best_model_dir, 'ser_optuna_encoder.joblib'))

    # Patch SimpleImputer if needed
    if not hasattr(imputer, '_fill_dtype'):
        imputer._fill_dtype = imputer.statistics_.dtype

    # 2. Load dataset
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    target_col = "label" if "label" in df.columns else "Label"
    
    df_cleaned = df.dropna(subset=[target_col]).copy()
    df_cleaned = df_cleaned[df_cleaned[target_col].astype(str).str.strip().str.lower() != "nan"]
    
    FEATURE_COLS = [col for col in df_cleaned.columns if col not in [target_col]]
    for col in FEATURE_COLS:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    X = df_cleaned[FEATURE_COLS].values
    y = df_cleaned[target_col].astype(str).str.strip().values
    y_encoded = encoder.transform(y)

    # 3. Recreate 80/20 train/test split (MUST use same random_state=42)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=RANDOM_STATE, stratify=y_encoded
    )

    print("\nGenerating Out-of-Fold (OOF) Probabilities using 5-fold CV...")
    
    # 4. Run 5-fold CV to generate out-of-fold probabilities
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_probas = np.zeros((len(X_train_raw), len(encoder.classes_)))
    
    # Get hyperparameters from the loaded model (only valid LGBMClassifier constructor params)
    valid_lgb_params = [
        'boosting_type', 'num_leaves', 'max_depth', 'learning_rate', 'n_estimators',
        'subsample_for_bin', 'objective', 'class_weight', 'min_split_gain',
        'min_child_weight', 'min_child_samples', 'subsample', 'subsample_freq',
        'colsample_bytree', 'reg_alpha', 'reg_lambda', 'random_state',
        'n_jobs', 'num_iterations', 'num_class', 'verbose',
    ]
    model_params = model.get_params()
    lgb_params = {k: v for k, v in model_params.items() if k in valid_lgb_params and v is not None}

    # Ensure class weights are a balanced dictionary
    unique_classes = np.unique(y_train)
    global_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=y_train
    )
    global_weight_dict = dict(zip(unique_classes, global_weights))
    lgb_params['class_weight'] = global_weight_dict
    
    # Run CV loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_raw, y_train)):
        X_tr_raw, y_tr = X_train_raw[train_idx], y_train[train_idx]
        X_va_raw, y_va = X_train_raw[val_idx], y_train[val_idx]
        
        # Preprocess
        X_tr_imp = imputer.fit_transform(X_tr_raw)
        X_va_imp = imputer.transform(X_va_raw)
        
        # Check feature shape to see if we apply feature engineering
        if model.n_features_in_ > 48:
            X_tr_proc = engineer_features(X_tr_imp)
            X_va_proc = engineer_features(X_va_imp)
        else:
            X_tr_proc = X_tr_imp
            X_va_proc = X_va_imp
            
        fold_scaler = StandardScaler()
        X_tr_scaled = fold_scaler.fit_transform(X_tr_proc)
        X_va_scaled = fold_scaler.transform(X_va_proc)
        
        # Train fold model
        fold_model = lgb.LGBMClassifier(**lgb_params)
        fold_model.fit(
            X_tr_scaled, y_tr,
            eval_set=[(X_va_scaled, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        # Store validation probabilities
        oof_probas[val_idx] = fold_model.predict_proba(X_va_scaled)
        print(f"  Fold {fold+1} complete.")

    # 5. Optimize thresholds using Powell algorithm
    def objective(weights, probas, y_true):
        weighted_probas = probas * weights
        preds = np.argmax(weighted_probas, axis=1)
        # We minimize the negative weighted F1 score
        return -f1_score(y_true, preds, average="weighted")

    print("\nOptimizing prediction thresholds using Scipy Powell algorithm...")
    init_weights = np.ones(len(encoder.classes_))
    bounds = [(0.1, 10.0)] * len(encoder.classes_)
    
    res = minimize(objective, init_weights, args=(oof_probas, y_train), method="Powell", bounds=bounds)
    best_thresholds = res.x

    print("\n=== Optimized Decision Threshold Multipliers ===")
    for cls_name, w in zip(encoder.classes_, best_thresholds):
        print(f"  {cls_name}: {w:.4f}")

    # 6. Save optimized thresholds
    thresholds_path = os.path.join(best_model_dir, 'ser_optuna_thresholds.joblib')
    joblib.dump(best_thresholds, thresholds_path)
    print(f"\nSaved optimized thresholds to: {thresholds_path}")

    # 7. Evaluate on the 20% test set
    print("\nEvaluating threshold-optimized model on test split...")
    X_test_imp = imputer.transform(X_test_raw)
    if model.n_features_in_ > 48:
        X_test_proc = engineer_features(X_test_imp)
    else:
        X_test_proc = X_test_imp
        
    X_test_scaled = scaler.transform(X_test_proc)
    
    # Standard prediction
    y_pred_std = model.predict(X_test_scaled)
    if isinstance(y_pred_std[0], str):
        y_pred_std = encoder.transform(y_pred_std)
        
    std_f1 = f1_score(y_test, y_pred_std, average="weighted")
    std_kappa = cohen_kappa_score(y_test, y_pred_std)
    
    # Optimized prediction
    probas_test = model.predict_proba(X_test_scaled)
    y_pred_opt = np.argmax(probas_test * best_thresholds, axis=1)
    
    opt_f1 = f1_score(y_test, y_pred_opt, average="weighted")
    opt_kappa = cohen_kappa_score(y_test, y_pred_opt)

    print("\n=================== STANDARD MODEL TEST REPORT ===================")
    print(classification_report(y_test, y_pred_std, target_names=encoder.classes_, digits=4))
    print(f"Weighted F1: {std_f1:.8f}")
    print(f"Cohen Kappa: {std_kappa:.8f}")

    print("\n=================== THRESHOLD-OPTIMIZED TEST REPORT ===================")
    print(classification_report(y_test, y_pred_opt, target_names=encoder.classes_, digits=4))
    print(f"Weighted F1: {opt_f1:.8f}")
    print(f"Cohen Kappa: {opt_kappa:.8f}")

    improvement = opt_f1 - std_f1
    print(f"\nF1 Score Improvement: {improvement * 100:+.4f}%")

    # 8. Overwrite reports and plots with optimized model outcomes
    print("\nUpdating visualization plots and reports with optimized metrics...")
    
    # Plots
    plot_confusion_matrix(y_test, y_pred_opt, encoder.classes_, os.path.join(figures_dir, "lightgbm_confusion.png"))
    plot_confusion_matrix(y_test, y_pred_opt, encoder.classes_, os.path.join(figures_dir, "lightgbm_boosted_confusion.png"))
    plot_f1_scores(y_test, y_pred_opt, encoder.classes_, os.path.join(figures_dir, "lightgbm_class_f1.png"))
    plot_f1_scores(y_test, y_pred_opt, encoder.classes_, os.path.join(figures_dir, "lightgbm_boosted_class_f1.png"))
    
    # Text Reports
    def write_report(path):
        with open(path, 'w') as f:
            f.write('=== Speech Emotion Recognition Standalone Boosted LightGBM Training Report (Threshold-Optimized) ===\n\n')
            f.write('=== LightGBM Weighted F1 ===\n')
            f.write(f'{opt_f1:.8f}\n\n')
            f.write('=== LightGBM Cohen Kappa ===\n')
            f.write(f'{opt_kappa:.8f}\n\n')
            f.write('=== LightGBM Classification Report ===\n')
            f.write(classification_report(y_test, y_pred_opt, target_names=encoder.classes_, digits=4))
            
    write_report(os.path.join(project_root, 'lightgbm_boosted_report.txt'))
    write_report(os.path.join(project_root, 'lightgbm_standalone_report.txt'))
    
    # Append to history log
    history_path = os.path.join(project_root, 'lightgbm_history.log')
    with open(history_path, 'a') as f:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] Device: cpu (BOOSTED + THRESHOLD OPTIMIZED)\n")
        f.write(f"  Weighted F1: {opt_f1:.8f}\n")
        f.write(f"  Cohen Kappa: {opt_kappa:.8f}\n")
        f.write(f"  Thresholds: {list(best_thresholds)}\n")
        f.write("-" * 80 + "\n")
        
    print("Done! Optimization report completed successfully.")


if __name__ == "__main__":
    main()
