# ===========================================================================
# Speech Emotion Recognition (SER) Advanced Stacking Ensemble Pipeline
# ===========================================================================
# This script trains a state-of-the-art stacking ensemble combining XGBoost,
# LightGBM, CatBoost, and a Multi-Layer Perceptron (MLP) base model using
# an optimized Extra Trees meta-classifier.
#
# Key Improvements implemented:
# 1. 5-Fold Stratified CV (up from 3-Fold) for high-quality OOF generation.
# 2. MLP Base Model integration for neural feature diversity.
# 3. Optimized Extra Trees Meta-Classifier.
# 4. Global figures and models outputs.

import json
import os
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
import torch

warnings.filterwarnings("ignore")

# Global Constants & Hardware Configuration
RANDOM_STATE = 42
CUDA_AVAILABLE = torch.cuda.is_available()
print(f"CUDA Available (PyTorch): {CUDA_AVAILABLE}")

# Path Configuration
_base = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_base) if os.path.basename(_base).lower() == "src" else _base
ALL_EMOTIONS_CSV = os.path.normpath(os.path.join(_project_root, "dataset", "all_emotions.csv"))
if not os.path.isfile(ALL_EMOTIONS_CSV):
    fallback_csv = os.path.normpath(os.path.join(_project_root, "all_emotions.csv"))
    if os.path.isfile(fallback_csv):
        ALL_EMOTIONS_CSV = fallback_csv

print(f"Dataset path: {ALL_EMOTIONS_CSV}")
print(f"File exists: {os.path.isfile(ALL_EMOTIONS_CSV)}")

# Data Loading & Imputation
df = pd.read_csv(ALL_EMOTIONS_CSV)
target_col = "label" if "label" in df.columns else "Label"

df_cleaned = df.dropna(subset=[target_col]).copy()
df_cleaned = df_cleaned[df_cleaned[target_col].astype(str).str.strip().str.lower() != "nan"]

FEATURE_COLS = [col for col in df_cleaned.columns if col not in [target_col]]
print(f"Number of features selected dynamically: {len(FEATURE_COLS)}")

# Impute features with median
for col in FEATURE_COLS:
    s = pd.to_numeric(df_cleaned[col], errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    med = s.median()
    if pd.isna(med):
        med = 0.0
    df_cleaned[col] = s.fillna(med)

X = df_cleaned[FEATURE_COLS].values
y_label = df_cleaned[target_col].astype(str).str.strip().values

# Label Encoding
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_label)

print(f"Cleaned X shape: {X.shape}")
print(f"Classes: {encoder.classes_}")

# Stratified Train-Test Split and Standard Scaling (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train Shape: {X_train_scaled.shape} | Test Shape: {X_test_scaled.shape}")

# Model Configuration & Parameters
params_path = os.path.join(_project_root, "best_params.json")
best_params = {}
if os.path.isfile(params_path):
    try:
        with open(params_path, "r") as f:
            best_params = json.load(f)
        print("Loaded best_params.json successfully.")
    except Exception as e:
        print("Error loading best_params.json:", e)

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
    num_class=len(encoder.classes_),
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
    num_class=len(encoder.classes_),
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

# MLP Base Classifier (Neural feature representation)
mlp_model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    alpha=0.001,
    learning_rate_init=0.001,
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=RANDOM_STATE
)

# 5-Fold Stratified Cross-Validation for OOF Predictions
print("\n--- Training Out-Of-Fold models for Stacking Meta-Features (5-Fold CV) ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
xgb_oof = np.zeros((len(X_train_scaled), len(encoder.classes_)))
lgb_oof = np.zeros((len(X_train_scaled), len(encoder.classes_)))
cb_oof = np.zeros((len(X_train_scaled), len(encoder.classes_)))
mlp_oof = np.zeros((len(X_train_scaled), len(encoder.classes_)))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train)):
    print(f"  Processing Fold {fold + 1}...")
    X_tr, y_tr = X_train_scaled[train_idx], y_train[train_idx]
    X_va, y_va = X_train_scaled[val_idx], y_train[val_idx]

    # Re-instantiate fold models
    xgb_m = xgb.XGBClassifier(**xgb_params, random_state=RANDOM_STATE, n_jobs=-1, eval_metric="mlogloss", objective="multi:softprob", num_class=len(encoder.classes_), device="cuda" if CUDA_AVAILABLE else "cpu")
    lgb_m = lgb.LGBMClassifier(**lgb_params, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, objective="multiclass", num_class=len(encoder.classes_), device="gpu" if CUDA_AVAILABLE else "cpu")
    cb_m = CatBoostClassifier(**cb_params, loss_function="MultiClass", random_seed=RANDOM_STATE, thread_count=-1, verbose=False, task_type="GPU" if CUDA_AVAILABLE else "CPU")
    mlp_m = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=0.001, learning_rate_init=0.001, max_iter=300, early_stopping=True, validation_fraction=0.1, random_state=RANDOM_STATE)

    # Fit and predict probabilities
    xgb_m.fit(X_tr, y_tr)
    lgb_m.fit(X_tr, y_tr)
    cb_m.fit(X_tr, y_tr)
    mlp_m.fit(X_tr, y_tr)

    xgb_oof[val_idx] = xgb_m.predict_proba(X_va)
    lgb_oof[val_idx] = lgb_m.predict_proba(X_va)
    cb_oof[val_idx] = cb_m.predict_proba(X_va)
    mlp_oof[val_idx] = mlp_m.predict_proba(X_va)

print("OOF predictions generated successfully.")

# Stacking Meta-Classifier (Extra Trees with Tuned Depth)
print("\n--- Training Stacking Meta-Classifier (Extra Trees) ---")
oof_meta_features = np.hstack([xgb_oof, lgb_oof, cb_oof, mlp_oof])

meta_model = ExtraTreesClassifier(n_estimators=400, max_depth=7, random_state=RANDOM_STATE, n_jobs=-1)
meta_model.fit(oof_meta_features, y_train)
print("Stacking Meta-Classifier trained successfully.")

# Final Model Training on Full Training Set
print("\n--- Fitting final models on full scaled training set ---")
xgb_model.fit(X_train_scaled, y_train)
lgb_model.fit(X_train_scaled, y_train)
cb_model.fit(X_train_scaled, y_train)
mlp_model.fit(X_train_scaled, y_train)

# Generate test set base probabilities
xgb_proba = xgb_model.predict_proba(X_test_scaled)
lgb_proba = lgb_model.predict_proba(X_test_scaled)
cb_proba = cb_model.predict_proba(X_test_scaled)
mlp_proba = mlp_model.predict_proba(X_test_scaled)

# Concat test meta-features
test_meta_features = np.hstack([xgb_proba, lgb_proba, cb_proba, mlp_proba])

# Run meta-classifier predictions
ensemble_pred = meta_model.predict(test_meta_features)

# Base models predictions for comparison
xgb_pred = np.argmax(xgb_proba, axis=1)
lgb_pred = np.argmax(lgb_proba, axis=1)
cb_pred = np.argmax(cb_proba, axis=1)
mlp_pred = np.argmax(mlp_proba, axis=1)

print("Final predictions computed.")

# Performance Evaluation
xgb_f1 = f1_score(y_test, xgb_pred, average="weighted")
xgb_kappa = cohen_kappa_score(y_test, xgb_pred)

lgb_f1 = f1_score(y_test, lgb_pred, average="weighted")
lgb_kappa = cohen_kappa_score(y_test, lgb_pred)

cb_f1 = f1_score(y_test, cb_pred, average="weighted")
cb_kappa = cohen_kappa_score(y_test, cb_pred)

mlp_f1 = f1_score(y_test, mlp_pred, average="weighted")
mlp_kappa = cohen_kappa_score(y_test, mlp_pred)

ensemble_f1 = f1_score(y_test, ensemble_pred, average="weighted")
ensemble_kappa = cohen_kappa_score(y_test, ensemble_pred)
ensemble_report = classification_report(y_test, ensemble_pred, target_names=encoder.classes_, output_dict=True)

print("\n=== XGBoost Test Metrics ===")
print(classification_report(y_test, xgb_pred, target_names=encoder.classes_))
print(f"Weighted F1: {xgb_f1:.4f}")

print("\n=== LightGBM Test Metrics ===")
print(classification_report(y_test, lgb_pred, target_names=encoder.classes_))
print(f"Weighted F1: {lgb_f1:.4f}")

print("\n=== CatBoost Test Metrics ===")
print(classification_report(y_test, cb_pred, target_names=encoder.classes_))
print(f"Weighted F1: {cb_f1:.4f}")

print("\n=== MLP Classifier Test Metrics ===")
print(classification_report(y_test, mlp_pred, target_names=encoder.classes_))
print(f"Weighted F1: {mlp_f1:.4f}")

print("\n=== Stacking Ensemble (XGB + LGB + CB + MLP -> Extra Trees) ===")
print(classification_report(y_test, ensemble_pred, target_names=encoder.classes_))
print(f"Stacking Ensemble Weighted F1: {ensemble_f1:.4f}")
print(f"Stacking Ensemble Cohen Kappa: {ensemble_kappa:.4f}")

# Visualizing Confusion Matrices (5 Subplots Side-by-Side)
fig, axes = plt.subplots(1, 5, figsize=(24, 4.8))
predictions = [
    (xgb_pred, "XGBoost Matrix"),
    (lgb_pred, "LightGBM Matrix"),
    (cb_pred, "CatBoost Matrix"),
    (mlp_pred, "MLP Neural Matrix"),
    (ensemble_pred, "Stacking Ensemble Matrix")
]

for idx, (pred, title) in enumerate(predictions):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=encoder.classes_, yticklabels=encoder.classes_, ax=axes[idx], cbar=False)
    axes[idx].set_title(title)
    axes[idx].set_xlabel("Predicted")
    axes[idx].set_ylabel("True")

plt.tight_layout()
figures_dir = os.path.join(_project_root, "figures")
os.makedirs(figures_dir, exist_ok=True)
plt.savefig(os.path.join(figures_dir, "stacking_confusion.png"), dpi=200, bbox_inches="tight")
plt.close()

# Performance Comparison Chart
models = ["XGBoost", "LightGBM", "CatBoost", "MLP Net", "Stacking Ensemble"]
f1_scores = [xgb_f1, lgb_f1, cb_f1, mlp_f1, ensemble_f1]
kappas = [xgb_kappa, lgb_kappa, cb_kappa, mlp_kappa, ensemble_kappa]

x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(10, 5))
plt.bar(x - width / 2, f1_scores, width, label="Weighted F1", color="skyblue")
plt.bar(x + width / 2, kappas, width, label="Cohen Kappa", color="steelblue")
plt.xticks(x, models)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Model Comparison: Advanced Stacking Pipeline")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "stacking_model_comparison.png"), dpi=200, bbox_inches="tight")
plt.close()

# Per-Class F1-Scores
class_f1 = [ensemble_report[label]["f1-score"] for label in encoder.classes_]
plt.figure(figsize=(8, 5))
sns.barplot(x=list(encoder.classes_), y=class_f1, color="steelblue")
plt.ylim(0, 1)
plt.ylabel("F1-score")
plt.xlabel("Emotion Class")
plt.title("Stacking Ensemble Per-Class F1-Score")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "stacking_class_f1.png"), dpi=200, bbox_inches="tight")
plt.close()

# Save Trained Models and Preprocessors
print("\nSerializing production artifacts for live backend...")
models_dir = os.path.join(_project_root, 'models')
os.makedirs(models_dir, exist_ok=True)

joblib.dump(xgb_model, os.path.join(models_dir, 'ser_xgb_model.joblib'))
joblib.dump(lgb_model, os.path.join(models_dir, 'ser_lgb_model.joblib'))
joblib.dump(cb_model,  os.path.join(models_dir, 'ser_cb_model.joblib'))
joblib.dump(mlp_model, os.path.join(models_dir, 'ser_mlp_model.joblib'))
joblib.dump(meta_model, os.path.join(models_dir, 'ser_meta_model.joblib'))
joblib.dump(scaler,    os.path.join(models_dir, 'ser_ensemble_scaler.joblib'))
joblib.dump(encoder,   os.path.join(models_dir, 'ser_ensemble_encoder.joblib'))
print("SUCCESS: Base models, MLP, preprocessors, and stacking meta-classifier saved to disk!")

# Save Stacking Report
report_path = os.path.join(_project_root, 'stacking_report.txt')
with open(report_path, 'w') as f:
    f.write('=== Speech Emotion Recognition Advanced Stacking Ensemble Training Report ===\n\n')
    f.write('=== XGBoost Weighted F1 ===\n')
    f.write(f'{xgb_f1:.4f}\n\n')
    f.write('=== LightGBM Weighted F1 ===\n')
    f.write(f'{lgb_f1:.4f}\n\n')
    f.write('=== CatBoost Weighted F1 ===\n')
    f.write(f'{cb_f1:.4f}\n\n')
    f.write('=== MLP Base Weighted F1 ===\n')
    f.write(f'{mlp_f1:.4f}\n\n')
    f.write('=== Stacking Ensemble (XGB + LGB + CB + MLP -> Extra Trees) ===\n')
    f.write(classification_report(y_test, ensemble_pred, target_names=encoder.classes_))
    f.write(f'\nEnsemble Weighted F1: {ensemble_f1:.4f}\n')
    f.write(f'Ensemble Cohen Kappa: {ensemble_kappa:.4f}\n')
print(f'Saved performance report to {report_path}')
