# 🏆 Speech Emotion Recognition - Best Model Checkpoint

This directory contains the serialized assets for the absolute best-performing Speech Emotion Recognition model checkpoint found in this workspace (**LGBM Optuna Tuned**).

## Performance Metrics (Test Set)
* **Weighted F1-Score:** `0.869979` (87.00%)
* **Cohen's Kappa:** `0.843445` (84.34%)
* **Overall Accuracy:** `0.869597` (86.96%)

### Class-Specific F1-Scores
* **Anger:** `0.93` *(Easiest to classify)*
* **Sad:** `0.87`
* **Disgust:** `0.87`
* **Fear:** `0.86`
* **Happy:** `0.86`
* **Neutral:** `0.83` *(Significantly improved from deep learning MLP baseline of 0.76)*

---

## File Contents
1. `ser_optuna_lightgbm.joblib`: Serialized LightGBM classifier checkpoint.
2. `ser_optuna_imputer.joblib`: SimpleImputer (strategy="median") fitted on 48 features.
3. `ser_optuna_scaler.joblib`: StandardScaler fitted on 48 features of the training set.
4. `ser_optuna_encoder.joblib`: LabelEncoder mapping string categories back to indices.

> **Note:** The imputer is now serialized and included with this checkpoint, making predictions safer and more robust to missing values or infinities.

---

## How to Load and Predict (Python Code Snippet)

```python
import os
import joblib
import numpy as np

# Load assets
model = joblib.load("ser_optuna_lightgbm.joblib")
imputer = joblib.load("ser_optuna_imputer.joblib")
scaler = joblib.load("ser_optuna_scaler.joblib")
encoder = joblib.load("ser_optuna_encoder.joblib")

# Example input: 48 acoustic features (1 sample)
dummy_features = np.random.randn(1, 48)

# Preprocess and run prediction
imputed_features = imputer.transform(dummy_features)
scaled_features = scaler.transform(imputed_features)
prediction_idx = model.predict(scaled_features)[0]

# Decode label to string
emotion_label = encoder.inverse_transform([prediction_idx])[0]
print(f"Predicted Emotion: {emotion_label}")
```
