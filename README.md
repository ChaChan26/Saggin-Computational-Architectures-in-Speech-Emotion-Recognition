# Speech Emotion Recognition (SER) Pipeline

This project builds a robust **Speech Emotion Recognition (SER)** machine learning pipeline to classify 6 distinct emotion classes (*anger, disgust, fear, happy, neutral, sad*) using pre-extracted tabular acoustic features.

---

## 📂 Project Structure

```
Nigga1-main/
│
├── dataset/
│   └── all_emotions.csv          # Feature dataset (54,485 audio samples)
│
├── checking_csvfile.ipynb        # Data inspection & cleaning
├── visualize_data.ipynb          # Exploratory Data Analysis (EDA) & class plots
├── random_forest.ipynb           # Baseline Random Forest model, tuning, & evaluation
├── advanced_ml_models.ipynb      # Advanced modeling (XGBoost, LightGBM, SVM)
├── ensemble_advanced.ipynb       # Weighted XGBoost + LightGBM ensemble with visualizations
├── mlp_baseline.py               # Deep Learning (PyTorch MLP) baseline classifier
├── all_emotions.csv              # Feature dataset (54,485 audio samples)
└── requirements.txt              # Standard project dependencies
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

## 📈 Model Performance (Advanced Models)

The advanced notebook compares **XGBoost**, **LightGBM**, and **SVM** with Optuna tuning:

*   **Validation F1:** `0.8421`
*   **Test Weighted F1:** `0.8506`
*   **Test Cohen's Kappa:** `0.8201`

For reference:
*   **XGBoost Validation F1:** `0.8335`
*   **SVM Validation F1:** `0.7219`

This is the strongest single-model result currently reported in the project. A weighted **XGBoost + LightGBM ensemble** with detailed visualizations is provided in `ensemble_advanced.ipynb`.

### Class Breakdown (F1-Scores)
*   **Anger:** `0.91` *(Easiest to classify)*
*   **Sad:** `0.85`
*   **Fear:** `0.82`
*   **Disgust:** `0.81`
*   **Happy:** `0.81`
*   **Neutral:** `0.79` *(Hardest to classify)*

---

## 🚀 Installation & Setup

1. **Install Dependencies:**
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run PyTorch Deep Learning Baseline:**
   To train the Multilayer Perceptron neural network baseline model:
   ```bash
   python mlp_baseline.py
   ```
   Note: `mlp_baseline.py` currently expects `all_emotions.csv` under `dataset/`. The file in this repo is stored at the project root, so either move it or update the script path before running.

3. **Explore Jupyter Notebooks:**
   Open the notebook environment to execute baseline and boosting models:
   ```bash
   jupyter notebook
   ```
