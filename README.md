# Speech Emotion Recognition (SER) Pipeline

This project builds a robust **Speech Emotion Recognition (SER)** machine learning pipeline to classify 6 distinct emotion classes (*anger, disgust, fear, happy, neutral, sad*) using pre-extracted tabular acoustic features.

---

## 📂 Project Structure

```
Speech-Emotion-Recognition/
│
├── dataset/
│   └── all_emotions.csv          # Feature dataset (54,485 audio samples)
│
├── figures/                      # Generated performance plots (Confusion matrices, comparisons)
│
├── notebooks/                    # Active Jupyter Notebooks (.ipynb)
│   ├── train_lightgbm.ipynb      # Standalone LightGBM training notebook
│   └── train_stacking_ensemble.ipynb # Stacking ensemble training notebook
│
├── src/                          # Active Python scripts (.py)
│   ├── train_lightgbm.py         # Standalone LightGBM model training script
│   └── review_models.py          # Systematic model performance evaluation script
│
├── models/                       # Saved / serialized model checkpoints
│   ├── ser_cb_model.joblib
│   ├── ser_ensemble_encoder.joblib
│   ├── ser_ensemble_scaler.joblib
│   ├── ser_ensemble_thresholds.joblib
│   ├── ser_lgb_model.joblib
│   ├── ser_lgb_standalone_encoder.joblib
│   ├── ser_lgb_standalone_imputer.joblib
│   ├── ser_lgb_standalone_model.joblib
│   ├── ser_lgb_standalone_scaler.joblib
│   ├── ser_lgb_standalone_thresholds.joblib
│   ├── ser_meta_model.joblib
│   ├── ser_mlp_model.joblib
│   ├── ser_optuna_encoder.joblib
│   ├── ser_optuna_lightgbm.joblib
│   ├── ser_optuna_scaler.joblib
│   ├── ser_rf_model.joblib
│   ├── ser_xgb_model.joblib
│   └── best_mlp_model.pth
│
├── best_model/                   # 🏆 Serialized assets for the best-performing model (LGBM Optuna Tuned)
│   ├── ser_optuna_lightgbm.joblib
│   ├── ser_optuna_scaler.joblib
│   └── ser_optuna_encoder.joblib
│
├── archive/                      # Historical and legacy files
│   ├── notebooks/                # Archived baseline/exploratory notebooks
│   └── src/                      # Archived CLI source scripts
│
├── requirements.txt              # Standard project dependencies
├── lightgbm_report.txt           # Latest LightGBM training report
├── stacking_report.txt           # Latest stacking ensemble validation report
├── models_review_report.txt      # Comprehensive review report for all checkpoints
└── .gitignore                    # Local cache ignore rules
```

---

## 📊 Dataset & Feature Selection

The dataset contains **54,485 cleaned audio samples** mapped to 6 emotions. The baseline features consist of **48 acoustic descriptors** extracted from raw audio signals, including pitch/fundamental frequency (F0), zero-crossing rate (ZCR), energy, spectral centroid, spectral flux, and MFCCs (Mel-Frequency Cepstral Coefficients).

### Feature Selection
To optimize training speed and prevent overfitting, we ran a feature importance analysis using the Random Forest classifier. All features with importance scores lower than `ZCR_mean` (`0.014965`) were pruned.

This reduced the feature count from **48** to **26 highly-informative features**:
*   **Pitch/F0:** `F0_mean`, `F0_std`, `F0_range`
*   **Energy:** `Energy_ mean`, `Energy_ std`
*   **Zero Crossing Rate (ZCR):** `ZCR_mean`, `ZCR_std`
*   **Spectral Descriptors:** `Spectral_centroid_mean`, `Spectral_centroid_std`, `Spectral_flux_mean`
*   **Mel-Frequency Cepstral Coefficients (MFCCs):** 
    *   *Means:* `MFCC_C0_mean`, `MFCC_C1_mean`, `MFCC_C2_mean`, `MFCC_C3_mean`, `MFCC_C5_mean`, `MFCC_C7_mean`, `MFCC_C10_mean`
    *   *Stds:* `MFCC_C0_std`, `MFCC_C1_std`, `MFCC_C2_std`, `MFCC_C3_std`, `MFCC_C5_std`, `MFCC_C7_std`
    *   *Deltas (Stds):* `Delta_MFCC_C0_std`, `Delta_MFCC_C2_std`, `Delta_MFCC_C3_std`

---

## 📈 Model Performance (Random Forest Baseline)

On an unseen test set of **10,897 samples**, the optimized Random Forest baseline achieves:
*   **Weighted F1-Score:** `0.8320` (83.2%)
*   **Cohen's Kappa:** `0.7974` (Strong agreement)

This is the **best-documented baseline** in the repository. The MLP script is included as a neural-network baseline, but its results are not reported in the README.

## 📈 Model Performance & Project Progress (Advanced & Ensemble Models)

The project progressed from single baseline models to a highly optimized **Stacking Ensemble** combining **XGBoost**, **LightGBM**, **CatBoost**, **MLP**, and **Random Forest** using an **Extra Trees meta-classifier** and **Powell-based decision threshold optimization**:

### Stacking Ensemble Setup (Ultimate Configuration)
* **Base Models Stacked**: XGBoost, LightGBM, CatBoost, MLP Classifier, Random Forest Classifier.
* **CV Configuration**: 10-Fold Stratified Cross Validation.
* **Meta-Classifier**: Optimized Extra Trees.
* **Decision Optimization**: Scipy Powell optimizer to calibrate class-specific prediction boundaries.

### Comparative Performance (Test Set)
| Model | Weighted F1 | Cohen's Kappa | Split | Status |
| :--- | :---: | :---: | :---: | :--- |
| **LGBM Optuna Tuned** | **0.9342** | **0.9209** | 80/20 | **Best Performer** |
| **LGBM Standalone (Optimized)** | **0.8808** | **0.8566** | 90/10 | **Active Production** |
| **Stacking Ensemble (Optimized)** | 0.8715 | 0.8453 | 80/20 | Tuned Ensemble |
| **Stacking Ensemble (Standard)** | 0.8709 | 0.8444 | 80/20 | Ensemble Baseline |
| **Stack Base - LightGBM** | 0.8677 | 0.8407 | 80/20 | Base Model |
| **Stack Base - XGBoost** | 0.8632 | 0.8353 | 80/20 | Base Model |
| **Stack Base - CatBoost** | 0.8317 | 0.7972 | 80/20 | Base Model |
| **Stack Base - MLP (sklearn)** | 0.8235 | 0.7876 | 80/20 | Base Model |
| **PyTorch MLP Baseline** | 0.7787 | 0.7331 | 70/15/15 | Deep Learning Baseline |
| **Stack Base - Random Forest** | 0.7620 | 0.7122 | 80/20 | Base Model |

### Class Breakdown (LGBM Optuna Tuned F1-Scores)
* **Anger:** `0.96` *(Easiest to classify)*
* **Sad:** `0.94`
* **Fear:** `0.93`
* **Disgust:** `0.93`
* **Happy:** `0.92`
* **Neutral:** `0.91` *(Hardest to classify, highly improved from baseline)*

---

## 🚀 Installation & Setup

1. **Install Dependencies:**
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Explore & Run Jupyter Notebooks (Recommended):**
   Model training can be executed interactively inside the `notebooks/` folder:
   ```bash
   jupyter notebook notebooks/train_lightgbm.ipynb
   ```
   Other notebooks include `train_stacking_ensemble.ipynb` for stacking models.

3. **Train the Production Model via CLI:**
   ```bash
   py src/train_lightgbm.py
   ```

4. **Evaluate Saved Models:**
   Evaluate and compare all serialized model checkpoints on test splits:
   ```bash
   py src/review_models.py
   ```
