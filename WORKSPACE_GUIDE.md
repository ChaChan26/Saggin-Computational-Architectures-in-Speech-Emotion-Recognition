# Speech Emotion Recognition Workspace Guide

This guide documents the directory structure and the status of all files in this repository. All historical, legacy, or experimental files have been separated into the `archive/` folder to keep the workspace clean.

---

## 🟢 Active & Production-Ready Files (Trained)
These files represent the current optimized machine learning pipeline and its serialized assets.

| File / Folder Path | Type | Status | Description |
| :--- | :---: | :---: | :--- |
| 📁 [**`src/`**](file:///c:/Users/User/Downloads/Saggin-Computational-Architectures-in-Speech-Emotion-Recognition/src/) | Directory | **Active** | Core source code directory. |
| ├── [**`train_lightgbm.py`**](file:///c:/Users/User/Downloads/Saggin-Computational-Architectures-in-Speech-Emotion-Recognition/src/train_lightgbm.py) | Python Script | **Active (Trained)** | Command-line script to train the standalone LightGBM model on a 90/10 split. Trains in < 6 seconds. |
| 📁 [**`notebooks/`**](file:///c:/Users/User/Downloads/Saggin-Computational-Architectures-in-Speech-Emotion-Recognition/notebooks/) | Directory | **Active** | Interactive Jupyter notebooks directory. |
| ├── [**`train_lightgbm.ipynb`**](file:///c:/Users/User/Downloads/Saggin-Computational-Architectures-in-Speech-Emotion-Recognition/notebooks/train_lightgbm.ipynb) | Notebook | **Active (Trained)** | Interactive version of the standalone LightGBM pipeline with markdown guides and inline plots. |
| 📁 [**`models/`**](file:///c:/Users/User/Downloads/Saggin-Computational-Architectures-in-Speech-Emotion-Recognition/models/) | Directory | **Active (Output)** | Serialized pipeline artifacts for backend API deployment. |
| ├── `ser_lgb_standalone_model.joblib` | Binary Asset | **Trained** | Serialized LightGBM classifier. |
| ├── `ser_lgb_standalone_imputer.joblib` | Binary Asset | **Fitted** | Median feature imputer. |
| ├── `ser_lgb_standalone_scaler.joblib` | Binary Asset | **Fitted** | Standard scaler. |
| ├── `ser_lgb_standalone_encoder.joblib` | Binary Asset | **Fitted** | Target label encoder. |
| 📁 [**`figures/`**](file:///c:/Users/User/Downloads/Saggin-Computational-Architectures-in-Speech-Emotion-Recognition/figures/) | Directory | **Active (Output)** | Output visualizations. |
| ├── `lightgbm_confusion.png` | Heatmap | **Generated** | Test set confusion matrix showing true vs. predicted emotion distributions. |
| ├── `lightgbm_class_f1.png` | Bar Chart | **Generated** | F1-scores broken down by class. |
| 📄 [**`lightgbm_report.txt`**](file:///c:/Users/User/Downloads/Saggin-Computational-Architectures-in-Speech-Emotion-Recognition/lightgbm_report.txt) | Text File | **Active (Output)** | Metrics log for the latest training run (F1-score: `0.8768`). |
| 📄 [**`best_params.json`**](file:///c:/Users/User/Downloads/Saggin-Computational-Architectures-in-Speech-Emotion-Recognition/best_params.json) | JSON File | **Active (Config)** | Optimal hyperparameters used for model configurations. |
| 📁 `dataset/` | Directory | **Active (Data)** | Contains raw dataset `all_emotions.csv` with 48 acoustic features. |
| 📄 `requirements.txt` | Config File | **Active** | List of Python dependencies (lightgbm, scikit-learn, joblib, seaborn, torch). |

---

## 🔴 Obsolete & Legacy Files (Archived)
These files are kept for historical reference but are **no longer used** in production. They have been moved to the `archive/` folder.

| File / Folder Path | Type | Status | Original Purpose / Why it is Obsolete |
| :--- | :---: | :---: | :--- |
| 📁 [**`archive/notebooks/`**](file:///c:/Users/User/Downloads/Saggin-Computational-Architectures-in-Speech-Emotion-Recognition/archive/notebooks/) | Directory | **Obsolete** | Old experiment notebooks. |
| ├── `train_stacking_ensemble.ipynb` | Notebook | **Obsolete** | Notebook version of the complex 10-fold Stacking Ensemble model. |
| ├── `train_final_ensemble.ipynb` | Notebook | **Obsolete** | Weighted voting ensemble model baseline. |
| ├── `production_ensemble_pipeline.ipynb` | Notebook | **Obsolete** | Older ensemble pipeline version. |
| ├── `production_ensemble_pipeline_fixed.ipynb` | Notebook | **Obsolete** | Older ensemble pipeline version with bugfixes. |
| ├── `production_optuna_pipeline.ipynb` | Notebook | **Obsolete** | Tuning hyperparameter search using Optuna. |
| ├── `advanced_ml_models.ipynb` | Notebook | **Obsolete** | Exploratory tests with advanced classifiers. |
| ├── `checking_csvfile.ipynb` | Notebook | **Obsolete** | Debug script to inspect the integrity of the raw CSV. |
| ├── `ensemble_advanced.ipynb` | Notebook | **Obsolete** | Earlier experimental advanced ensemble notebook. |
| ├── `random_forest.ipynb` | Notebook | **Obsolete** | Random Forest baseline model exploration. |
| ├── `visualize_data.ipynb` | Notebook | **Obsolete** | Exploratory data visualization. |
| 📁 [**`archive/src/`**](file:///c:/Users/User/Downloads/Saggin-Computational-Architectures-in-Speech-Emotion-Recognition/archive/src/) | Directory | **Obsolete** | Old CLI source scripts. |
| ├── `train_stacking_ensemble.py` | Python Script | **Obsolete** | Complex Stacking Ensemble script (slow, complex deployment). |
| ├── `train_final_ensemble.py` | Python Script | **Obsolete** | Old final weighted ensemble CLI. |
| ├── `tune_pipeline.py` | Python Script | **Obsolete** | CLI parameter tuning runner using Optuna. |
| ├── `mlp_baseline.py` | Python Script | **Obsolete** | Multilayer Perceptron PyTorch baseline script. |
| ├── `neutral_improve.py` | Python Script | **Obsolete** | Target class threshold optimization for neutral class baseline. |
| ├── `ensemble_advanced.py` | Python Script | **Obsolete** | Legacy advanced ensemble helper. |
| ├── `extracted_ensemble_pipeline.py` | Python Script | **Obsolete** | Intermediate state of the ensembling pipeline. |
| 📄 `archive/best_lightgbm_params.json` | JSON File | **Obsolete** | Legacy parameters (replaced by `best_params.json`). |
| 📄 `archive/ensemble_weights.json` | JSON File | **Obsolete** | Voting weights for the legacy ensemble. |
| 📄 `archive/stacking_report.txt` | Text File | **Obsolete** | Metrics report for the legacy Stacking Ensemble model. |
