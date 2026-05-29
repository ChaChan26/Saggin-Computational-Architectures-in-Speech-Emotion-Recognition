# Speech Emotion Recognition (SER) Pipeline 

This repository implements a high-performance, GPU-accelerated **Speech Emotion Recognition (SER)** machine learning pipeline. It classifies acoustic feature vectors extracted from audio recordings into one of 6 target emotional states: **anger, disgust, fear, happy, neutral, and sad**.

---

## 📂 Project Structure

```
Saggin-Computational-Architectures-in-Speech-Emotion-Recognition-main/
│
├── dataset/
│   └── all_emotions.csv          # Pre-extracted acoustic feature dataset (54,485 samples)
│
├── checking_csvfile.ipynb        # Data sanity checks, null handling, and type corrections
├── visualize_data.ipynb          # Exploratory Data Analysis (EDA), distribution plots, and class balances
├── random_forest.ipynb           # Baseline Random Forest classifier, hyperparameter grid search, and feature selection
├── advanced_ml_models.ipynb      # Optuna hyperparameter tuning for XGBoost, LightGBM, and SVM
├── ensemble_advanced.py          # Baseline XGBoost + LightGBM weighted soft-voting ensemble
├── train_final_ensemble.py       # Final high-accuracy XGBoost + LightGBM + CatBoost OOF ensemble (GPU)
├── tune_pipeline.py              # Optuna hyperparameter tuning on all 48 features (GPU)
├── mlp_baseline.py               # Deep Learning baseline (PyTorch MLP) with early stopping
├── requirements.txt              # Project library dependencies
└── README.md                     # Comprehensive technical documentation
```

---

## 📊 Feature Glossary (48 Acoustic Descriptors)

The dataset contains **54,485 audio samples** with **48 acoustic features** extracted from speech signals using windowed framing. Below is the glossary of the acoustic descriptors:

| Feature Name Group | Count | Description & Emotion Correlation |
| :--- | :---: | :--- |
| **Pitch (`F0_mean`, `F0_std`, `F0_range`)** | 3 | Fundamental Frequency ($F_0$). Represents vocal fold vibration rate. High pitch mean/variability indicates high arousal (anger, fear), while flat pitch indicates low arousal (sad, neutral). |
| **Energy (`Energy_ mean`, `Energy_ std`)** | 2 | Signal amplitude/loudness. Crucial for distinguishing high-intensity emotions (anger, happy) from low-intensity ones (sad, neutral). |
| **Zero Crossing Rate (`ZCR_mean`, `ZCR_std`)** | 2 | Frequency of sign changes in the audio wave. Measures signal noisiness and spectral complexity; useful for separating voiced and unvoiced speech segments. |
| **Spectral Centroid (`Spectral_centroid_mean`, `Spectral_centroid_std`)** | 2 | The "center of mass" of the spectral distribution. Higher values correspond to "brighter" sounds (anger, fear), while lower values correspond to "darker" sounds. |
| **Spectral Flux (`Spectral_flux_mean`)** | 1 | The rate of change of the spectral power spectrum over time. Captures speech rhythm and spectral transitions. |
| **MFCC Means (`MFCC_C0_mean` to `MFCC_C12_mean`)** | 13 | Mel-Frequency Cepstral Coefficients (coefficients 0 to 12). Represents the envelope of the vocal tract spectrum. |
| **MFCC Stds (`MFCC_C0_std` to `MFCC_C12_std`)** | 13 | Standard deviation of the MFCCs across time frames. Captures the temporal stability of the vocal tract shape. |
| **Delta MFCCs (`Delta_MFCC_C0_std` to `Delta_MFCC_C5_std`)** | 12 | First-order derivatives (differential trajectories) of the MFCCs. Represents the speed of spectral transitions, which correlates with articulation speed under emotional states. |

---

##  Blending Weights & Soft-Voting Mathematics

The high accuracy of the final ensemble model (**`0.8719` Test F1-Score**) is achieved using a **Weighted Soft-Voting Ensemble** optimized via **Out-of-Fold (OOF)** cross-validation.

### 1. Soft-Voting Probability Formula
For a given input sample $x$, each model outputs a probability vector over the $C$ emotion classes. The ensemble calculates a weighted average of these probabilities:

$$\hat{P}(c \mid x) = \frac{\sum_{m \in M} w_m \cdot P_m(c \mid x)}{\sum_{m \in M} w_m}$$

where:
*   $M = \{\text{XGBoost}, \text{LightGBM}, \text{CatBoost}\}$
*   $w_m$ is the blending weight assigned to model $m$ (satisfying $\sum w_m = 1.0$)
*   $P_m(c \mid x)$ is the probability predicted by model $m$ for class $c$.

The final prediction is selected as the argmax of the blended probabilities:

$$\hat{y} = \arg\max_{c \in \{1,\dots,C\}} \hat{P}(c \mid x)$$

### 2. Out-of-Fold (OOF) Weight Optimization
Directly optimizing weights on the test set leads to **data leakage** and overfitting. To solve this, we:
1.  Partition the training set into 3 stratified folds.
2.  Train the base models on 2 folds and predict probabilities on the remaining 1 validation fold.
3.  Concatenate these validation predictions across all folds to build **Out-of-Fold (OOF) prediction arrays** ($X_{\text{OOF}}$) of shape $(N_{\text{train}}, C)$ representing unbiased performance on unseen data.
4.  Run a grid search over possible weight combinations to maximize the weighted F1-score on the OOF arrays:
    *   **Optimal XGBoost Weight ($w_{\text{xgb}}$)**: `0.40`
    *   **Optimal LightGBM Weight ($w_{\text{lgb}}$)**: `0.42`
    *   **Optimal CatBoost Weight ($w_{\text{cb}}$)**: `0.18`

---

## 📈 Detailed Benchmark Performance

All models were evaluated on an independent test set of **10,897 samples (20% split)**.

### Model Metrics Table

| Model / Feature Configuration | Feature Count | Test Weighted F1 | Test Cohen's Kappa |
| :--- | :---: | :---: | :---: |
| PyTorch MLP (Deep Learning Baseline) | 26 | `0.7787` | `0.7331` |
| Random Forest (ML Baseline) | 26 | `0.8320` | `0.7974` |
| XGBoost (Single Fit) | 26 | `0.8495` | `0.8188` |
| LightGBM (Single Fit) | 26 | `0.8506` | `0.8201` |
| **Weighted Ensemble (XGB + LGB)** | **26** | **`0.8536`** | **`0.8236`** |
| | | | |
| CatBoost (Single Fit) | 48 | `0.8355` | `0.8019` |
| XGBoost (Single Fit) | 48 | `0.8632` | `0.8353` |
| LightGBM (Single Fit) | 48 | `0.8704` | `0.8439` |
| **Optimal Weighted Ensemble (XGB+LGB+CB)** | **48** | **`0.8719`** 🚀 | **`0.8458`** 🚀 |

### Class Breakdown for Optimal Ensemble
```text
              precision    recall  f1-score   support

       anger       0.94      0.92      0.93      1863
     disgust       0.84      0.87      0.85      1863
        fear       0.86      0.86      0.86      1863
       happy       0.85      0.84      0.84      1863
     neutral       0.81      0.82      0.82      1583
         sad       0.88      0.87      0.87      1862

    accuracy                           0.86     10897
   macro avg       0.86      0.86      0.86     10897
weighted avg       0.86      0.86      0.86     10897
```

---

## 🛠️ Hyperparameter Configurations

Below are the exact hyperparameter sets tuned via Optuna on the 48-feature dataset:

```python
# XGBoost Config
xgb_params = {
    "n_estimators": 468,
    "max_depth": 10,
    "learning_rate": 0.17549140891728818,
    "subsample": 0.9690394070981359,
    "colsample_bytree": 0.7725519831966194,
    "gamma": 1.0492767129301485e-08,
    "eval_metric": "mlogloss",
    "objective": "multi:softprob",
}

# LightGBM Config
lgb_params = {
    "n_estimators": 499,
    "max_depth": 11,
    "num_leaves": 67,
    "learning_rate": 0.24625126683753454,
    "subsample": 0.6909971314141472,
    "colsample_bytree": 0.7554088091076056,
    "objective": "multiclass",
}

# CatBoost Config
cb_params = {
    "iterations": 500,
    "depth": 8,
    "learning_rate": 0.15,
    "l2_leaf_reg": 3.0,
    "loss_function": "MultiClass",
}
```

---

## 🚀 Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

> [!NOTE]
> All scripts contain **Automatic GPU/CPU Detection** using PyTorch:
> ```python
> import torch
> CUDA_AVAILABLE = torch.cuda.is_available()
> ```
> If a CUDA-enabled GPU is detected, models automatically compile on the GPU. Otherwise, they fallback to standard multi-threaded CPU execution.

### 2. Run the Main Training & Evaluation Pipeline
Trains models, optimizes blending weights, evaluates the test set, and outputs confusion matrix heatmaps to the `figures/` folder:
```bash
python train_final_ensemble.py
```

### 3. Run Hyperparameter Tuning
Runs Optuna searches to find alternative parameters on the full 48 features:
```bash
python tune_pipeline.py
```

### 4. Run PyTorch Neural Network Baseline
Trains the neural network baseline using an early stopping criteria:
```bash
python mlp_baseline.py
```
