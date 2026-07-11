import os
import time
import asyncio
import joblib
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from concurrent.futures import ThreadPoolExecutor

class FeatureCache:
    """
    An O(1) in-memory feature cache for numerical vectors.
    Uses rounded features as a dictionary key to achieve sub-microsecond lookups.
    """
    def __init__(self, decimals: int = 4):
        self.decimals = decimals
        # Keys can contain strings ('NaN'), floats (rounded), or infs
        self.cache: Dict[Tuple[Any, ...], Tuple[str, np.ndarray]] = {}
        self.keys_fifo: List[Tuple[Any, ...]] = []

    def _make_key(self, features: np.ndarray) -> Tuple[Any, ...]:
        feat_flat = features.flatten()
        return tuple(
            'NaN' if np.isnan(x) else
            float('inf') if np.isposinf(x) else
            float('-inf') if np.isneginf(x) else
            round(float(x), self.decimals)
            for x in feat_flat
        )

    def get(self, features: np.ndarray) -> Optional[Tuple[str, np.ndarray, float]]:
        """
        Lookup features in cache.
        Returns (label, probability_distribution, similarity_score=1.0) if hit, else None.
        """
        key = self._make_key(features)
        if key in self.cache:
            label, prob = self.cache[key]
            return label, prob, 1.0
        return None

    def set(self, features: np.ndarray, label: str, prob: np.ndarray):
        """Add predictions to cache. Limits cache size to prevent memory leaks."""
        key = self._make_key(features)
        if key in self.cache:
            return
            
        if len(self.cache) >= 1000:
            oldest_key = self.keys_fifo.pop(0)
            self.cache.pop(oldest_key, None)
            
        self.cache[key] = (label, prob.copy())
        self.keys_fifo.append(key)

class ParallelPredictor:
    """
    Executes base model predictions concurrently using a thread pool.
    Releases the Python GIL (C-based ML libraries) for true parallel CPU execution.
    """
    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.models: Dict[str, Any] = {}
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._load_base_models()

    def _load_base_models(self):
        # Pruned base models list (only LGB and XGB)
        model_files = {
            "lgb": "ser_lgb_model.joblib",
            "xgb": "ser_xgb_model.joblib"
        }
        for name, filename in model_files.items():
            path = os.path.join(self.models_dir, filename)
            if os.path.isfile(path):
                try:
                    self.models[name] = joblib.load(path)
                except Exception as e:
                    print(f"Warning: Could not load base model {name} from {path}: {e}")

    async def _predict_single_model(self, name: str, model: Any, X_scaled: np.ndarray) -> np.ndarray:
        """Runs model prediction inside the thread pool."""
        loop = asyncio.get_running_loop()
        # predict_proba is thread-safe and releases the GIL in C++
        return await loop.run_in_executor(self.executor, model.predict_proba, X_scaled)

    async def predict_all_parallel(self, X_scaled: np.ndarray) -> Dict[str, np.ndarray]:
        """Runs prediction for all base models in parallel."""
        tasks = {}
        for name, model in self.models.items():
            tasks[name] = asyncio.create_task(self._predict_single_model(name, model, X_scaled))
            
        if not tasks:
            return {}
            
        # Run concurrently
        await asyncio.gather(*tasks.values())
        return {name: task.result() for name, task in tasks.items()}

    def close(self):
        """Shutdown the thread pool executor to prevent memory/thread leaks."""
        self.executor.shutdown(wait=True)

class LatencyOptimizedInference:
    """
    Orchestrates the entire latency-optimized Speech Emotion Recognition inference pipeline.
    Uses Caching -> Fast Model Prediction -> Confidence Routing -> Fallback.
    """
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.best_model_dir = os.path.join(project_root, "best_model")
        self.models_dir = os.path.join(project_root, "models")
        
        self.cache = FeatureCache(decimals=4)
        self.parallel_predictor = ParallelPredictor(self.models_dir)
        
        # Load best production model assets
        self.model = joblib.load(os.path.join(self.best_model_dir, "ser_optuna_lightgbm.joblib"))
        self.imputer = joblib.load(os.path.join(self.best_model_dir, "ser_optuna_imputer.joblib"))
        self.scaler = joblib.load(os.path.join(self.best_model_dir, "ser_optuna_scaler.joblib"))
        self.encoder = joblib.load(os.path.join(self.best_model_dir, "ser_optuna_encoder.joblib"))
        
        # Load ensemble assets for fallback scaling and decoding (meta-model removed)
        self.ensemble_scaler = joblib.load(os.path.join(self.models_dir, "ser_ensemble_scaler.joblib"))
        self.ensemble_encoder = joblib.load(os.path.join(self.models_dir, "ser_ensemble_encoder.joblib"))
        
    def predict_sequential_tools(self, X_raw: np.ndarray) -> Dict[str, Any]:
        """Runs standard sequential prediction (Un-optimized reference)."""
        start = time.perf_counter()
        
        # 1. Preprocess
        X_imputed = self.imputer.transform(X_raw)
        X_scaled = self.scaler.transform(X_imputed)
        
        # 2. Sequential prediction across base models
        base_preds = {}
        X_ens_scaled = self.ensemble_scaler.transform(X_imputed)
        for name, model in self.parallel_predictor.models.items():
            base_preds[name] = model.predict_proba(X_ens_scaled)
            
        # 3. Final model prediction
        proba = self.model.predict_proba(X_scaled)
        pred_idx = np.argmax(proba, axis=1)
        label = self.encoder.inverse_transform(pred_idx)[0]
        
        elapsed = time.perf_counter() - start
        return {
            "label": label,
            "probability": proba[0],
            "latency_ms": elapsed * 1000,
            "cache_hit": False,
            "method": "sequential_unoptimized"
        }

    async def predict_optimized(self, X_raw: np.ndarray, confidence_threshold: float = 0.85) -> Dict[str, Any]:
        """
        Runs optimized prediction pipeline:
        1. Check numerical feature cache (~0.005ms)
        2. Clean/Scale features
        3. Fast route: Predict using LightGBM. If confidence > threshold, return early (~1-2ms)
        4. If confidence is low, run base models (LGBM, XGBoost) in parallel and average probabilities
        """
        start = time.perf_counter()
        
        # --- LAYER 1: NUMERICAL CACHING ---
        cached = self.cache.get(X_raw)
        if cached:
            label, prob, sim = cached
            elapsed = time.perf_counter() - start
            return {
                "label": label,
                "probability": prob,
                "latency_ms": elapsed * 1000,
                "cache_hit": True,
                "method": "cached_match"
            }
            
        # --- PREPROCESSING ---
        X_imputed = self.imputer.transform(X_raw)
        X_scaled = self.scaler.transform(X_imputed)
        
        # --- LAYER 2: CONFIDENCE ROUTING (Fast-path model prediction) ---
        proba = self.model.predict_proba(X_scaled)[0]
        max_prob = float(np.max(proba))
        pred_idx = int(np.argmax(proba))
        label = self.encoder.inverse_transform([pred_idx])[0]
        
        if max_prob >= confidence_threshold:
            # High confidence - cache and return early
            self.cache.set(X_raw, label, proba)
            elapsed = time.perf_counter() - start
            return {
                "label": label,
                "probability": proba,
                "latency_ms": elapsed * 1000,
                "cache_hit": False,
                "method": "confidence_fast_path"
            }
            
        # --- LAYER 3: PARALLEL BASE MODEL EXECUTION (Low confidence fallback) ---
        X_ens_scaled = self.ensemble_scaler.transform(X_imputed)
        parallel_results = await self.parallel_predictor.predict_all_parallel(X_ens_scaled)
        
        # Extract individual model probabilities
        lgb_proba = parallel_results.get("lgb")
        xgb_proba = parallel_results.get("xgb")
        
        # Safety fallback if either model prediction failed to return
        if lgb_proba is None or xgb_proba is None:
            elapsed = time.perf_counter() - start
            self.cache.set(X_raw, label, proba)
            return {
                "label": label,
                "probability": proba,
                "latency_ms": elapsed * 1000,
                "cache_hit": False,
                "method": "parallel_fallback_failed"
            }
            
        lgb_proba = np.atleast_2d(lgb_proba)[0]
        xgb_proba = np.atleast_2d(xgb_proba)[0]
        
        # Simple average ensemble prediction
        ensemble_proba = (lgb_proba + xgb_proba) / 2.0
        ensemble_pred_idx = np.argmax(ensemble_proba)
        ensemble_label = self.ensemble_encoder.inverse_transform([ensemble_pred_idx])[0]
        
        # Cache this final prediction (only once)
        self.cache.set(X_raw, ensemble_label, ensemble_proba)
        
        elapsed = time.perf_counter() - start
        return {
            "label": ensemble_label,
            "probability": ensemble_proba,
            "latency_ms": elapsed * 1000,
            "cache_hit": False,
            "method": "parallel_fallback_ensemble"
        }

    def close(self):
        """Shutdown the underlying parallel executor to prevent resource leaks."""
        self.parallel_predictor.close()
