import os
import json
import warnings
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# PyTorch MLP definition for PyTorch model loading
class EmotionMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(EmotionMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)

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
    raise FileNotFoundError(f"Could not find all_emotions.csv. Searched in: {search_paths}")

def get_base_directories() -> tuple[str, str, str]:
    base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    project_root = os.path.dirname(base_dir) if os.path.basename(base_dir).lower() in ["src", "notebooks"] else base_dir
    models_dir = os.path.join(project_root, "models")
    figures_dir = os.path.join(project_root, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return project_root, models_dir, figures_dir

def main():
    print("=" * 80)
    print("      SPEECH EMOTION RECOGNITION - SYSTEMATIC MODEL PERFORMANCE REVIEW")
    print("=" * 80)
    
    project_root, models_dir, figures_dir = get_base_directories()
    csv_path = resolve_csv_path(project_root)
    print(f"Loading dataset from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    target_col = "label" if "label" in df.columns else "Label"
    
    # Clean null targets
    df_cleaned = df.dropna(subset=[target_col]).copy()
    df_cleaned = df_cleaned[df_cleaned[target_col].astype(str).str.strip().str.lower() != "nan"]
    
    # Define classes order
    classes = sorted(list(df_cleaned[target_col].astype(str).str.strip().unique()))
    print(f"Dataset unique classes: {classes}")
    
    # -------------------------------------------------------------
    # Prepare raw features (no imputation yet - done after splitting)
    # -------------------------------------------------------------
    # 1. All 48 features
    FEATURE_COLS_48 = [col for col in df_cleaned.columns if col not in [target_col]]
    
    # Convert to numeric and replace inf with NaN (but do NOT fill NaN yet)
    df_numeric_48 = df_cleaned.copy()
    for col in FEATURE_COLS_48:
        df_numeric_48[col] = pd.to_numeric(df_numeric_48[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        
    X_48_raw = df_numeric_48[FEATURE_COLS_48].values
    y_str = df_numeric_48[target_col].astype(str).str.strip().values
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_str)
    
    # 80/20 split (raw, unimputed)
    X_train_80_raw, X_test_80_raw, y_train_80, y_test_80 = train_test_split(
        X_48_raw, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Impute 80/20 split using train-only medians
    imputer_80 = SimpleImputer(strategy="median")
    X_train_80 = imputer_80.fit_transform(X_train_80_raw)
    X_test_80 = imputer_80.transform(X_test_80_raw)
    
    # 90/10 split (raw, unimputed)
    X_train_90_raw, X_test_90_raw, y_train_90, y_test_90 = train_test_split(
        X_48_raw, y_encoded, test_size=0.1, random_state=42, stratify=y_encoded
    )
    
    # Impute 90/10 split using train-only medians
    imputer_90 = SimpleImputer(strategy="median")
    X_train_90 = imputer_90.fit_transform(X_train_90_raw)
    X_test_90 = imputer_90.transform(X_test_90_raw)
    
    # 3. 70/15/15 split on 26 features (for PyTorch MLP baseline)
    FEATURE_COLS_26 = [
        "F0_mean", "F0_std", "F0_range", "Energy_ mean", "Energy_ std", 
        "ZCR_mean", "ZCR_std", "Spectral_centroid_mean", "Spectral_centroid_std", 
        "Spectral_flux_mean", "MFCC_C0_mean", "MFCC_C1_mean", "MFCC_C2_mean", 
        "MFCC_C3_mean", "MFCC_C5_mean", "MFCC_C7_mean", "MFCC_C10_mean", 
        "MFCC_C0_std", "MFCC_C1_std", "MFCC_C2_std", "MFCC_C3_std", 
        "MFCC_C5_std", "MFCC_C7_std", "Delta_MFCC_C0_std", "Delta_MFCC_C2_std", 
        "Delta_MFCC_C3_std"
    ]
    df_numeric_26 = df_cleaned.copy()
    for col in FEATURE_COLS_26:
        df_numeric_26[col] = pd.to_numeric(df_numeric_26[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        
    X_26_raw = df_numeric_26[FEATURE_COLS_26].values
    
    X_tr_val_raw, X_test_15_raw, y_tr_val, y_test_15 = train_test_split(
        X_26_raw, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
    )
    
    print("-" * 80)
    print(f"Data Splits Configured:")
    print(f"  - 80/20 split test size: {len(y_test_80)} samples")
    print(f"  - 90/10 split test size: {len(y_test_90)} samples")
    print(f"  - 70/15/15 split test size: {len(y_test_15)} samples")
    
    results = {}
    reports = {}
    confusion_matrices = {}
    class_f1_scores = {}
    
    # -------------------------------------------------------------
    # MODEL 1: Standalone LightGBM (90/10 Split)
    # -------------------------------------------------------------
    print("\nEvaluating Standalone LightGBM model...")
    try:
        lgb_standalone_model = joblib.load(os.path.join(models_dir, "ser_lgb_standalone_model.joblib"))
        lgb_standalone_imputer = joblib.load(os.path.join(models_dir, "ser_lgb_standalone_imputer.joblib"))
        lgb_standalone_scaler = joblib.load(os.path.join(models_dir, "ser_lgb_standalone_scaler.joblib"))
        lgb_standalone_encoder = joblib.load(os.path.join(models_dir, "ser_lgb_standalone_encoder.joblib"))
        lgb_standalone_thresholds = joblib.load(os.path.join(models_dir, "ser_lgb_standalone_thresholds.joblib"))
        
        # Use the saved imputer and scaler on the raw 90/10 test split
        if not hasattr(lgb_standalone_imputer, '_fill_dtype'):
            lgb_standalone_imputer._fill_dtype = lgb_standalone_imputer.statistics_.dtype
        X_test_90_imputed = lgb_standalone_imputer.transform(X_test_90_raw)
        X_test_90_scaled = lgb_standalone_scaler.transform(X_test_90_imputed)
        
        # Standard predict
        lgb_sa_pred = lgb_standalone_model.predict(X_test_90_scaled)
        # Handle predictions (if they are string or numerical labels)
        if isinstance(lgb_sa_pred[0], str):
            lgb_sa_pred_encoded = lgb_standalone_encoder.transform(lgb_sa_pred)
        else:
            lgb_sa_pred_encoded = lgb_sa_pred
            
        lgb_sa_f1 = f1_score(y_test_90, lgb_sa_pred_encoded, average="weighted")
        lgb_sa_kappa = cohen_kappa_score(y_test_90, lgb_sa_pred_encoded)
        lgb_sa_acc = np.mean(y_test_90 == lgb_sa_pred_encoded)
        
        results["LGBM Standalone (Std)"] = {"F1": lgb_sa_f1, "Kappa": lgb_sa_kappa, "Accuracy": lgb_sa_acc, "Split": "90/10"}
        reports["LGBM Standalone (Std)"] = classification_report(y_test_90, lgb_sa_pred_encoded, target_names=lgb_standalone_encoder.classes_)
        confusion_matrices["LGBM Standalone (Std)"] = confusion_matrix(y_test_90, lgb_sa_pred_encoded)
        class_f1_scores["LGBM Standalone (Std)"] = f1_score(y_test_90, lgb_sa_pred_encoded, average=None)
        
        # Optimized predict (with threshold multipliers)
        lgb_sa_proba = lgb_standalone_model.predict_proba(X_test_90_scaled)
        lgb_sa_pred_opt = np.argmax(lgb_sa_proba * lgb_standalone_thresholds, axis=1)
        
        lgb_sa_opt_f1 = f1_score(y_test_90, lgb_sa_pred_opt, average="weighted")
        lgb_sa_opt_kappa = cohen_kappa_score(y_test_90, lgb_sa_pred_opt)
        lgb_sa_opt_acc = np.mean(y_test_90 == lgb_sa_pred_opt)
        
        results["LGBM Standalone (Opt)"] = {"F1": lgb_sa_opt_f1, "Kappa": lgb_sa_opt_kappa, "Accuracy": lgb_sa_opt_acc, "Split": "90/10"}
        reports["LGBM Standalone (Opt)"] = classification_report(y_test_90, lgb_sa_pred_opt, target_names=lgb_standalone_encoder.classes_)
        confusion_matrices["LGBM Standalone (Opt)"] = confusion_matrix(y_test_90, lgb_sa_pred_opt)
        class_f1_scores["LGBM Standalone (Opt)"] = f1_score(y_test_90, lgb_sa_pred_opt, average=None)
        
        print("Standalone LightGBM evaluated successfully!")
    except Exception as e:
        print(f"Error evaluating Standalone LightGBM: {e}")
        import traceback
        traceback.print_exc()

    # -------------------------------------------------------------
    # MODEL 2: Optuna LightGBM (80/20 Split)
    # -------------------------------------------------------------
    print("\nEvaluating Optuna LightGBM model...")
    try:
        best_model_dir = os.path.join(project_root, "best_model")
        optuna_model_path = os.path.join(best_model_dir, "ser_optuna_lightgbm.joblib") if os.path.exists(os.path.join(best_model_dir, "ser_optuna_lightgbm.joblib")) else os.path.join(models_dir, "ser_optuna_lightgbm.joblib")
        optuna_scaler_path = os.path.join(best_model_dir, "ser_optuna_scaler.joblib") if os.path.exists(os.path.join(best_model_dir, "ser_optuna_scaler.joblib")) else os.path.join(models_dir, "ser_optuna_scaler.joblib")
        optuna_encoder_path = os.path.join(best_model_dir, "ser_optuna_encoder.joblib") if os.path.exists(os.path.join(best_model_dir, "ser_optuna_encoder.joblib")) else os.path.join(models_dir, "ser_optuna_encoder.joblib")
        
        optuna_model = joblib.load(optuna_model_path)
        optuna_scaler = joblib.load(optuna_scaler_path)
        optuna_encoder = joblib.load(optuna_encoder_path)
        
        # Optuna LGBM preprocessed test data (on 80/20 split)
        # Impute test data using train-only imputer, then scale with saved scaler
        X_test_80_imputed = imputer_80.transform(X_test_80_raw)
        X_test_80_scaled = optuna_scaler.transform(X_test_80_imputed)
        
        optuna_pred = optuna_model.predict(X_test_80_scaled)
        if isinstance(optuna_pred[0], str):
            optuna_pred_encoded = optuna_encoder.transform(optuna_pred)
        else:
            optuna_pred_encoded = optuna_pred
            
        optuna_f1 = f1_score(y_test_80, optuna_pred_encoded, average="weighted")
        optuna_kappa = cohen_kappa_score(y_test_80, optuna_pred_encoded)
        optuna_acc = np.mean(y_test_80 == optuna_pred_encoded)
        
        results["LGBM Optuna Tuned"] = {"F1": optuna_f1, "Kappa": optuna_kappa, "Accuracy": optuna_acc, "Split": "80/20"}
        reports["LGBM Optuna Tuned"] = classification_report(y_test_80, optuna_pred_encoded, target_names=optuna_encoder.classes_)
        confusion_matrices["LGBM Optuna Tuned"] = confusion_matrix(y_test_80, optuna_pred_encoded)
        class_f1_scores["LGBM Optuna Tuned"] = f1_score(y_test_80, optuna_pred_encoded, average=None)
        
        print("Optuna LightGBM evaluated successfully!")
    except Exception as e:
        print(f"Error evaluating Optuna LightGBM: {e}")

    # -------------------------------------------------------------
    # MODEL 3: Stacking Ensemble & Base Classifiers (80/20 Split)
    # -------------------------------------------------------------
    print("\nEvaluating Stacking Ensemble & Base Classifiers...")
    try:
        xgb_m = joblib.load(os.path.join(models_dir, "ser_xgb_model.joblib"))
        lgb_m = joblib.load(os.path.join(models_dir, "ser_lgb_model.joblib"))
        cb_m = joblib.load(os.path.join(models_dir, "ser_cb_model.joblib"))
        mlp_m = joblib.load(os.path.join(models_dir, "ser_mlp_model.joblib"))
        rf_m = joblib.load(os.path.join(models_dir, "ser_rf_model.joblib"))
        
        meta_model = joblib.load(os.path.join(models_dir, "ser_meta_model.joblib"))
        ensemble_scaler = joblib.load(os.path.join(models_dir, "ser_ensemble_scaler.joblib"))
        ensemble_encoder = joblib.load(os.path.join(models_dir, "ser_ensemble_encoder.joblib"))
        ensemble_thresholds = joblib.load(os.path.join(models_dir, "ser_ensemble_thresholds.joblib"))
        
        # Scale test data
        X_test_80_ens_scaled = ensemble_scaler.transform(X_test_80)
        
        # Get base model probability predictions
        xgb_proba = xgb_m.predict_proba(X_test_80_ens_scaled)
        lgb_proba = lgb_m.predict_proba(X_test_80_ens_scaled)
        cb_proba = cb_m.predict_proba(X_test_80_ens_scaled)
        mlp_proba = mlp_m.predict_proba(X_test_80_ens_scaled)
        rf_proba = rf_m.predict_proba(X_test_80_ens_scaled)
        
        # Individual Base Classifiers Metrics
        base_models = {
            "Stack Base - XGBoost": xgb_proba,
            "Stack Base - LightGBM": lgb_proba,
            "Stack Base - CatBoost": cb_proba,
            "Stack Base - MLP": mlp_proba,
            "Stack Base - Random Forest": rf_proba
        }
        
        for name, proba in base_models.items():
            pred = np.argmax(proba, axis=1)
            f1 = f1_score(y_test_80, pred, average="weighted")
            kappa = cohen_kappa_score(y_test_80, pred)
            acc = np.mean(y_test_80 == pred)
            
            results[name] = {"F1": f1, "Kappa": kappa, "Accuracy": acc, "Split": "80/20"}
            reports[name] = classification_report(y_test_80, pred, target_names=ensemble_encoder.classes_)
            confusion_matrices[name] = confusion_matrix(y_test_80, pred)
            class_f1_scores[name] = f1_score(y_test_80, pred, average=None)
            
        # Concat meta-features
        test_meta_features = np.hstack([xgb_proba, lgb_proba, cb_proba, mlp_proba, rf_proba])
        
        # Meta model predictions
        ensemble_proba = meta_model.predict_proba(test_meta_features)
        
        # Standard Stacking Ensemble
        ensemble_pred = np.argmax(ensemble_proba, axis=1)
        ens_f1 = f1_score(y_test_80, ensemble_pred, average="weighted")
        ens_kappa = cohen_kappa_score(y_test_80, ensemble_pred)
        ens_acc = np.mean(y_test_80 == ensemble_pred)
        
        results["Stacking Ensemble (Std)"] = {"F1": ens_f1, "Kappa": ens_kappa, "Accuracy": ens_acc, "Split": "80/20"}
        reports["Stacking Ensemble (Std)"] = classification_report(y_test_80, ensemble_pred, target_names=ensemble_encoder.classes_)
        confusion_matrices["Stacking Ensemble (Std)"] = confusion_matrix(y_test_80, ensemble_pred)
        class_f1_scores["Stacking Ensemble (Std)"] = f1_score(y_test_80, ensemble_pred, average=None)
        
        # Threshold-Optimized Stacking Ensemble
        ensemble_pred_opt = np.argmax(ensemble_proba * ensemble_thresholds, axis=1)
        ens_opt_f1 = f1_score(y_test_80, ensemble_pred_opt, average="weighted")
        ens_opt_kappa = cohen_kappa_score(y_test_80, ensemble_pred_opt)
        ens_opt_acc = np.mean(y_test_80 == ensemble_pred_opt)
        
        results["Stacking Ensemble (Opt)"] = {"F1": ens_opt_f1, "Kappa": ens_opt_kappa, "Accuracy": ens_opt_acc, "Split": "80/20"}
        reports["Stacking Ensemble (Opt)"] = classification_report(y_test_80, ensemble_pred_opt, target_names=ensemble_encoder.classes_)
        confusion_matrices["Stacking Ensemble (Opt)"] = confusion_matrix(y_test_80, ensemble_pred_opt)
        class_f1_scores["Stacking Ensemble (Opt)"] = f1_score(y_test_80, ensemble_pred_opt, average=None)
        
        print("Stacking Ensemble and all 5 base classifiers evaluated successfully!")
    except Exception as e:
        print(f"Error evaluating Stacking Ensemble: {e}")
        import traceback
        traceback.print_exc()

    # -------------------------------------------------------------
    # MODEL 4: PyTorch MLP (70/15/15 Split)
    # -------------------------------------------------------------
    print("\nEvaluating PyTorch MLP Baseline model...")
    try:
        mlp_path = os.path.join(models_dir, "best_mlp_model.pth")
        if os.path.exists(mlp_path):
            # Since standard scaling was fit on training 70% data, let's fit a scaling object for the 70/15/15 split
            # matching mlp_baseline.py split exactly to make predictions comparable
            scaler_mlp = StandardScaler()
            # Split train_val (85%) and test (15%) - use raw data
            X_tr_val_mlp_raw, X_test_15_mlp_raw, y_tr_val_le, y_test_15_le = train_test_split(
                X_26_raw, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
            )
            # Split train_val into train (82.35%) and val (17.65%)
            X_train_70, _, _, _ = train_test_split(
                X_tr_val_mlp_raw, y_tr_val_le, test_size=0.1765, random_state=42, stratify=y_tr_val_le
            )
            
            # Impute train_70 and test_15 using train-only medians
            imputer_mlp = SimpleImputer(strategy="median")
            X_train_70_imputed = imputer_mlp.fit_transform(X_train_70)
            X_test_15_mlp_imputed = imputer_mlp.transform(X_test_15_mlp_raw)
            
            scaler_mlp.fit(X_train_70_imputed)
            X_test_15_scaled = scaler_mlp.transform(X_test_15_mlp_imputed)
            
            # Load PyTorch model
            mlp_model = EmotionMLP(input_dim=26, num_classes=len(classes))
            mlp_model.load_state_dict(torch.load(mlp_path, map_location=torch.device('cpu')))
            mlp_model.eval()
            
            with torch.no_grad():
                data_t = torch.tensor(X_test_15_scaled, dtype=torch.float32)
                outputs = mlp_model(data_t)
                _, preds_t = torch.max(outputs, 1)
                mlp_preds = preds_t.numpy()
                
            mlp_f1 = f1_score(y_test_15_le, mlp_preds, average="weighted")
            mlp_kappa = cohen_kappa_score(y_test_15_le, mlp_preds)
            mlp_acc = np.mean(y_test_15_le == mlp_preds)
            
            results["PyTorch MLP Baseline"] = {"F1": mlp_f1, "Kappa": mlp_kappa, "Accuracy": mlp_acc, "Split": "70/15/15"}
            reports["PyTorch MLP Baseline"] = classification_report(y_test_15_le, mlp_preds, target_names=classes)
            confusion_matrices["PyTorch MLP Baseline"] = confusion_matrix(y_test_15_le, mlp_preds)
            class_f1_scores["PyTorch MLP Baseline"] = f1_score(y_test_15_le, mlp_preds, average=None)
            
            print("PyTorch MLP Baseline evaluated successfully!")
        else:
            print("PyTorch MLP weight file best_mlp_model.pth not found in models/.")
    except Exception as e:
        print(f"Error evaluating PyTorch MLP: {e}")
        import traceback
        traceback.print_exc()

    # -------------------------------------------------------------
    # PRINT RESULTS TABLE
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("                      MODEL EVALUATION METRICS COMPARISON")
    print("" + "=" * 80)
    print(f"{'Model Name':<28} | {'Split':<8} | {'Weighted F1':<11} | {'Cohen Kappa':<11} | {'Accuracy':<8}")
    print("-" * 80)
    for model_name, metrics in sorted(results.items(), key=lambda x: x[1]["F1"], reverse=True):
        print(f"{model_name:<28} | {metrics['Split']:<8} | {metrics['F1']:.6f}    | {metrics['Kappa']:.6f}    | {metrics['Accuracy']:.6f}")
    print("=" * 80)
    
    # -------------------------------------------------------------
    # SAVE TEXT REPORT TO FILE
    # -------------------------------------------------------------
    report_file_path = os.path.join(project_root, "models_review_report.txt")
    with open(report_file_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("      SPEECH EMOTION RECOGNITION - SYSTEMATIC MODEL PERFORMANCE REVIEW\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("=== SUMMARY METRICS TABLE ===\n\n")
        f.write(f"{'Model Name':<28} | {'Split':<8} | {'Weighted F1':<11} | {'Cohen Kappa':<11} | {'Accuracy':<8}\n")
        f.write("-" * 80 + "\n")
        for model_name, metrics in sorted(results.items(), key=lambda x: x[1]["F1"], reverse=True):
            f.write(f"{model_name:<28} | {metrics['Split']:<8} | {metrics['F1']:.6f}    | {metrics['Kappa']:.6f}    | {metrics['Accuracy']:.6f}\n")
        f.write("=" * 80 + "\n\n")
        
        for model_name, rep in reports.items():
            f.write(f"=== {model_name} Detailed Classification Report ===\n")
            f.write(rep + "\n")
            f.write("-" * 80 + "\n\n")
            
    print(f"Saved performance report to {report_file_path}")
    
    # -------------------------------------------------------------
    # GENERATE PLOTS
    # -------------------------------------------------------------
    # Plot 1: Model Comparison Bar Chart
    plt.figure(figsize=(12, 6))
    sorted_models = sorted(results.items(), key=lambda x: x[1]["F1"], reverse=True)
    model_names_plot = [m[0] for m in sorted_models]
    f1_plot = [m[1]["F1"] for m in sorted_models]
    kappa_plot = [m[1]["Kappa"] for m in sorted_models]
    
    x = np.arange(len(model_names_plot))
    width = 0.35
    
    plt.bar(x - width/2, f1_plot, width, label="Weighted F1-Score", color="#3498db")
    plt.bar(x + width/2, kappa_plot, width, label="Cohen's Kappa", color="#2c3e50")
    plt.xticks(x, model_names_plot, rotation=45, ha="right")
    plt.ylabel("Score")
    plt.ylim(0.6, 1.0)
    plt.title("Performance Comparison Across All Serialized Models")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "review_model_comparison.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved model comparison plot to: {os.path.join(figures_dir, 'review_model_comparison.png')}")
    
    # Plot 2: Confusion Matrices for Top 4 Models
    top_4_models = ["LGBM Optuna Tuned", "Stacking Ensemble (Opt)", "Stacking Ensemble (Std)", "Stack Base - LightGBM"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for i, name in enumerate(top_4_models):
        if name in confusion_matrices:
            sns.heatmap(
                confusion_matrices[name], 
                annot=True, 
                fmt="d", 
                cmap="Blues", 
                xticklabels=classes, 
                yticklabels=classes, 
                ax=axes[i], 
                cbar=False
            )
            axes[i].set_title(f"{name}\nConfusion Matrix")
            axes[i].set_xlabel("Predicted")
            axes[i].set_ylabel("True")
        else:
            axes[i].text(0.5, 0.5, f"{name} not available", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "review_confusion_matrices.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrices to: {os.path.join(figures_dir, 'review_confusion_matrices.png')}")
    
    # Plot 3: Per-class F1-score comparison for LGBM Standalone (Opt) vs Stacking Ensemble (Opt)
    plt.figure(figsize=(10, 5))
    if "LGBM Optuna Tuned" in class_f1_scores and "Stacking Ensemble (Opt)" in class_f1_scores:
        lgb_f1s = class_f1_scores["LGBM Optuna Tuned"]
        stack_f1s = class_f1_scores["Stacking Ensemble (Opt)"]
        x_cls = np.arange(len(classes))
        plt.bar(x_cls - width/2, lgb_f1s, width, label="LGBM Optuna Tuned", color="#3498db")
        plt.bar(x_cls + width/2, stack_f1s, width, label="Stacking Ensemble (Optimized)", color="#2ecc71")
        plt.xticks(x_cls, classes)
        plt.ylabel("F1-Score")
        plt.ylim(0.7, 1.0)
        plt.title("Per-Class F1-Score Comparison: Optuna LGBM vs. Stacking Ensemble")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "review_class_f1_comparison.png"), dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved per-class F1 comparison to: {os.path.join(figures_dir, 'review_class_f1_comparison.png')}")

if __name__ == "__main__":
    main()
