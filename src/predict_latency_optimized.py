import os
import time
import joblib
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

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

class LatencyOptimizedInference:
    """
    Orchestrates the entire latency-optimized Speech Emotion Recognition inference pipeline.
    Uses Caching -> Fast Model Prediction (LightGBM).
    """
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.best_model_dir = os.path.join(project_root, "best_model")
        
        self.cache = FeatureCache(decimals=4)
        
        # Load best production model assets
        self.model = joblib.load(os.path.join(self.best_model_dir, "ser_optuna_lightgbm.joblib"))
        self.imputer = joblib.load(os.path.join(self.best_model_dir, "ser_optuna_imputer.joblib"))
        self.scaler = joblib.load(os.path.join(self.best_model_dir, "ser_optuna_scaler.joblib"))
        self.encoder = joblib.load(os.path.join(self.best_model_dir, "ser_optuna_encoder.joblib"))
        
    def predict_sequential_tools(self, X_raw: np.ndarray) -> Dict[str, Any]:
        """Runs standard prediction (Un-optimized reference)."""
        start = time.perf_counter()
        
        # 1. Preprocess
        X_imputed = self.imputer.transform(X_raw)
        X_scaled = self.scaler.transform(X_imputed)
        
        # 2. Final model prediction
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

    async def predict_optimized(self, X_raw: np.ndarray, confidence_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Runs optimized prediction pipeline:
        1. Check numerical feature cache (~0.005ms)
        2. Clean/Scale features
        3. Predict using Standalone LightGBM (~1-2ms)
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
        
        # --- LAYER 2: STANDALONE LIGHTGBM PREDICTION ---
        proba = self.model.predict_proba(X_scaled)[0]
        pred_idx = int(np.argmax(proba))
        label = self.encoder.inverse_transform([pred_idx])[0]
        
        # Cache this prediction
        self.cache.set(X_raw, label, proba)
        
        elapsed = time.perf_counter() - start
        return {
            "label": label,
            "probability": proba,
            "latency_ms": elapsed * 1000,
            "cache_hit": False,
            "method": "standalone_lightgbm"
        }
