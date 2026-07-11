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

## Latency Optimization Guidelines
1. **Numerical Feature Caching:** The inference wrapper uses an $O(1)$ dictionary-based cache with rounded feature tuples. NaN values are mapped to a `'NaN'` string sentinel to prevent key collisions with valid `0.0` measurements.
2. **Single-Model Inference:** The pipeline uses a single standalone Optuna-tuned LightGBM model (<2 ms latency). Fallback ensembles have been removed. Do not re-introduce concurrent base model execution unless the accuracy gain exceeds +1% F1.

## Development Log Rule
* **DEVLOG.md:** A persistent development journal exists at [DEVLOG.md](file:///c:/Emotion/Saggin-Computational-Architectures-in-Speech-Emotion-Recognition/DEVLOG.md).
* **Rule:** At the end of every coding session that produces a Git commit, append a new entry to `DEVLOG.md` above the `<!-- NEXT ENTRY GOES HERE -->` marker. Use the format: `## [YYYY-MM-DD] — commit hash — One-line summary`, followed by a brief description of what changed and why.
