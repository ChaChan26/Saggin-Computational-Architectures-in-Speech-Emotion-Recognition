# 🏆 Speech Emotion Recognition - Best Model Checkpoint

This directory contains the serialized assets for the absolute best-performing Speech Emotion Recognition model checkpoint found in this workspace (**LGBM Optuna Tuned**).

## Performance Metrics (Test Set)
* **Weighted F1-Score:** `0.934237` (93.42%)
* **Cohen's Kappa:** `0.920890` (92.09%)
* **Overall Accuracy:** `0.934110` (93.41%)

### Class-Specific F1-Scores
* **Anger:** `0.96` *(Easiest to classify)*
* **Sad:** `0.94`
* **Fear:** `0.93`
* **Disgust:** `0.93`
* **Happy:** `0.92`
* **Neutral:** `0.91` *(Highly improved from baselines)*

---

## File Contents
1. `ser_optuna_lightgbm.joblib`: Serialized LightGBM classifier checkpoint.
2. `ser_optuna_scaler.joblib`: StandardScaler fitted on 48 features of the training set.
3. `ser_optuna_encoder.joblib`: LabelEncoder mapping string categories back to indices.

---

## How to Load and Predict (Python Code Snippet)

```python
import os
import joblib
import numpy as np

# Load assets
model = joblib.load("ser_optuna_lightgbm.joblib")
scaler = joblib.load("ser_optuna_scaler.joblib")
encoder = joblib.load("ser_optuna_encoder.joblib")

# Example input: 48 acoustic features (1 sample)
dummy_features = np.random.randn(1, 48)

# Preprocess and run prediction
scaled_features = scaler.transform(dummy_features)
prediction_idx = model.predict(scaled_features)[0]

# Decode label to string
emotion_label = encoder.inverse_transform([prediction_idx])[0]
print(f"Predicted Emotion: {emotion_label}")
```
