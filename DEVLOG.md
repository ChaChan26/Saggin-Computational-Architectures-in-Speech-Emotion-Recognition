# DEVLOG — Speech Emotion Recognition Pipeline

> A living development journal. Each session appends a new entry below.  
> **Format:** `## [Date] — Commit Hash — One-line Summary`

---

## [2026-06-21] — `583593f` — Pre-review checkpoint

Saved all raw work before structured code review. Included the first trained LightGBM model, workspace rules, Jupyter notebook, and initial review figures.

---

## [2026-06-21] — `ea9f984` — Fix imputation leakage, clean imports

Fixed data leakage bug where the imputer was fitted on the full dataset (including test samples). Cleaned unused imports. Updated `.gitignore` to exclude large binary artifacts.

---

## [2026-06-21] — `f90d1ad` — README documentation pass

Updated README to reflect renamed report file. Consolidated model comparison tables. Added model card to `best_model/README.md`.

---

## [2026-06-21] — `1fb6f1c` — Clarify preprocessing docs

Corrected workspace guidelines: preprocessing uses `StandardScaler` (not `MinMaxScaler`), preceded by `SimpleImputer(strategy="median")`.

---

## [2026-06-26] — `189e7ae` — Fix evaluation leakage, production checkpoint

Discovered and fixed evaluation leakage in train/test splits. Retrained standalone Optuna-tuned LightGBM on clean 80/20 stratified split. Serialized final production checkpoint.

**Result:** 87.00% Weighted F1, 84.34% Cohen Kappa.

---

## [2026-06-30] — `7ece697` — Boosted training pipeline & Optuna tuning

Added feature engineering utilities, class-specific threshold optimization (Scipy Powell), and parallel Optuna hyperparameter tuning. Saved first 122 trials to `optuna_study.db`.

---

## [2026-06-30] — `67ed020` — Review pipeline refactor & workspace guide

Cleaned up model review evaluation for consistent reporting. Created `WORKSPACE_GUIDE.md`. Re-serialized production model assets with consistent preprocessing.

---

## [2026-07-11] — `0a9837a` — Cache optimization, ensemble pruning, SSL reference

Ran a 5-persona Native LLM Council review. Key findings:
- 90% F1 is mathematically impossible on the 48 tabular features.
- Linear cosine cache was a latency tax (3–8 ms > model's 1–2 ms).
- Shadow cache bug: low-confidence predictions overwrote ensemble results.
- 5 concurrent models caused OpenMP core contention.

Changes:
- Replaced $O(N)$ cosine cache with $O(1)$ dictionary lookup.
- Pruned base models from 5 to 2 (XGB + LGB).
- Removed ExtraTrees meta-classifier (~20 MB saved).
- Created `src/extract_ssl_embeddings.py` (Wav2Vec 2.0 reference pipeline).

---

## [2026-07-11] — `dc80345` — Fix NaN collision, thread leak, crash safety

Second council review found three hidden bugs:
- `np.nan_to_num` mapped NaN→0.0, causing cache key collisions. Fixed with `'NaN'` string sentinel.
- `ThreadPoolExecutor` never shut down → thread leak. Added `close()` method.
- Missing model predictions caused `TypeError` on `None`. Added safety fallback.

---

## [2026-07-11] — `93b321a` — Strip fallbacks, standalone LightGBM only

Removed the entire fallback ensemble (0.15% F1 gain not worth 60 MB RAM + thread complexity):
- Deleted `ParallelPredictor` class, `asyncio`, `ThreadPoolExecutor`.
- Pipeline now: Cache Check → Preprocess → LightGBM → Cache → Return.
- Net −119 lines deleted. RAM ~40 MB (down from ~110 MB). Latency <2 ms.

---

<!-- NEXT ENTRY GOES HERE -->
