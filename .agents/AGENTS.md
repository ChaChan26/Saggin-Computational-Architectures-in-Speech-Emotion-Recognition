# Workspace Guidelines: Speech Emotion Recognition (SER) Pipeline

## Project Workflow Reference

### 1. Data Structure
* Raw Dataset: [dataset/all_emotions.csv](file:///c:/Emotion/Saggin-Computational-Architectures-in-Speech-Emotion-Recognition/dataset/all_emotions.csv) (54,485 samples, 48 features).

### 2. Best Model Checkpoint
* Folder: [best_model/](file:///c:/Emotion/Saggin-Computational-Architectures-in-Speech-Emotion-Recognition/best_model)
* Model: **LGBM Optuna Tuned** (87.00% Weighted F1, 84.34% Cohen Kappa).
* Preprocessing: `SimpleImputer(strategy="median")` → **StandardScaler** (not MinMax) → `LabelEncoder` (on 48 features).
* Split: Stratified **80/20 split** (`test_size=0.20`, `random_state=42`).

### 3. Hyperparameter Configuration
* Parameters for the model are archived in [archive/best_lightgbm_params.json](file:///c:/Emotion/Saggin-Computational-Architectures-in-Speech-Emotion-Recognition/archive/best_lightgbm_params.json) and configured in the root [best_params.json](file:///c:/Emotion/Saggin-Computational-Architectures-in-Speech-Emotion-Recognition/best_params.json).

### 4. Active Scripts & Notebooks
* Training Pipeline: [src/train_lightgbm.py](file:///c:/Emotion/Saggin-Computational-Architectures-in-Speech-Emotion-Recognition/src/train_lightgbm.py) & [notebooks/train_lightgbm.ipynb](file:///c:/Emotion/Saggin-Computational-Architectures-in-Speech-Emotion-Recognition/notebooks/train_lightgbm.ipynb) (configured to use the 80/20 split and optimal params).
* Evaluation script: [src/review_models.py](file:///c:/Emotion/Saggin-Computational-Architectures-in-Speech-Emotion-Recognition/src/review_models.py) (runs systematic evaluation on test splits).

### 5. Running the Pipeline
* To retrain the best model (87%): Run [notebooks/train_lightgbm.ipynb](file:///c:/Emotion/Saggin-Computational-Architectures-in-Speech-Emotion-Recognition/notebooks/train_lightgbm.ipynb) or execute `py src/train_lightgbm.py`.
* To evaluate checkpoints: Execute `py src/review_models.py`.
