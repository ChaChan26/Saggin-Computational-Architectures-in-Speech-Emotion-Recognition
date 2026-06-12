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
├── notebooks/                    # All exploratory and baseline Jupyter Notebooks (.ipynb)
│   ├── checking_csvfile.ipynb
│   ├── visualize_data.ipynb
│   ├── random_forest.ipynb
│   ├── advanced_ml_models.ipynb
│   ├── ensemble_advanced.ipynb
│   ├── production_ensemble_pipeline.ipynb
│   ├── production_ensemble_pipeline_fixed.ipynb
│   ├── production_optuna_pipeline.ipynb
│   ├── train_final_ensemble.ipynb
│   └── train_stacking_ensemble.ipynb
│
├── src/                          # All executable Python scripts (.py)
│   ├── tune_pipeline.py              # Hyperparameter tuning script via Optuna
│   ├── train_lightgbm.py             # Single LightGBM model training script
│   ├── train_final_ensemble.py       # Optimally weighted ensemble training script
│   ├── train_stacking_ensemble.py    # Stacking ensemble classifier training script (best model)
│   ├── neutral_improve.py            # Model training & threshold moving for neutral class
│   ├── mlp_baseline.py               # PyTorch Multi-layer Perceptron baseline script
│   ├── ensemble_advanced.py          # Soft voting ensemble script
│   └── extracted_ensemble_pipeline.py # Production deployment / inference pipeline script
│
├── models/                       # Saved / serialized model checkpoints
│   ├── ser_cb_model.joblib
│   ├── ser_ensemble_encoder.joblib
│   ├── ser_ensemble_scaler.joblib
│   ├── ser_lgb_model.joblib
│   ├── ser_meta_model.joblib
│   ├── ser_mlp_model.joblib
│   ├── ser_rf_model.joblib
│   ├── ser_optuna_encoder.joblib
│   ├── ser_optuna_lightgbm.joblib
│   ├── ser_optuna_scaler.joblib
│   ├── ser_xgb_model.joblib
│   └── best_mlp_model.pth
│
├── requirements.txt              # Standard project dependencies
├── stacking_report.txt           # Latest stacking ensemble validation report
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
| Model | Weighted F1 | Cohen's Kappa |
| :--- | :---: | :---: |
| **Stacking Ensemble (Optimized)** | **0.8745** | **0.8488** |
| **Stacking Ensemble (Standard)** | 0.8745 | 0.8488 |
| **LightGBM** | 0.8704 | 0.8439 |
| **XGBoost** | 0.8632 | 0.8353 |
| **CatBoost** | 0.8355 | 0.8019 |
| **MLP Neural Net** | 0.8235 | 0.7876 |
| **Random Forest Baseline** | 0.7597 | 0.7094 |

### Class Breakdown (Optimized Ensemble F1-Scores)
* **Anger:** `0.94` *(Easiest to classify)*
* **Sad:** `0.88`
* **Disgust:** `0.87`
* **Fear:** `0.87`
* **Happy:** `0.86`
* **Neutral:** `0.83` *(Hardest to classify, significantly improved from baseline)*

---

## 🚀 Installation & Setup

1. **Install Dependencies:**
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Explore & Run Jupyter Notebooks (Recommended):**
   All model training can be executed interactively inside the `notebooks/` folder:
   ```bash
   jupyter notebook notebooks/train_stacking_ensemble.ipynb
   ```
   Other notebooks include `train_final_ensemble.ipynb` for weighted voting and `production_optuna_pipeline.ipynb` for tuning.

3. **Train the Stacking Ensemble via CLI:**
   ```bash
   python src/train_stacking_ensemble.py
   ```

4. **Evaluate Saved Models:**
   ```bash
   python src/review_ensemble.py
   ```
