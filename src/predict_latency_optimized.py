import os
import time
import math
import joblib
import threading
import warnings
import asyncio
import numpy as np
from typing import Dict, Any, Tuple, Optional, List, Union

class FeatureCache:
    """
    A thread-safe, latency-optimized in-memory feature cache for numerical vectors.
    Uses rounded features as a dictionary key to achieve sub-microsecond lookups.
    """
    def __init__(self, decimals: int = 4, max_size: int = 1000):
        self.decimals = decimals
        self.max_size = max_size
        # Keys contain strings ('NaN', 'inf', '-inf') or rounded floats
        self.cache: Dict[Tuple[Any, ...], Tuple[str, np.ndarray]] = {}
        self.keys_fifo: List[Tuple[Any, ...]] = []
        self.lock = threading.Lock()

    def _make_key(self, features: np.ndarray) -> Tuple[Any, ...]:
        # Flatten and cast to standard float array
        feat_flat = features.flatten().astype(float)
        
        # Check if all elements are finite (no nan/inf)
        if np.isfinite(feat_flat).all():
            # Fast path: vectorized rounding and list conversion (C-speed)
            return tuple(np.round(feat_flat, self.decimals).tolist())
            
        # Slow path fallback: handle non-finite values using fast Python math scalars
        return tuple(
            'NaN' if math.isnan(x) else
            float('inf') if math.isinf(x) and x > 0 else
            float('-inf') if math.isinf(x) else
            round(x, self.decimals)
            for x in feat_flat.tolist()
        )

    def get(self, features: np.ndarray) -> Optional[Tuple[str, np.ndarray, float]]:
        """
        Lookup features in cache.
        Returns (label, probability_distribution, similarity_score=1.0) if hit, else None.
        """
        key = self._make_key(features)
        with self.lock:
            if key in self.cache:
                label, prob = self.cache[key]
                return label, prob.copy(), 1.0
        return None

    def set(self, features: np.ndarray, label: str, prob: np.ndarray):
        """Add predictions to cache. Limits cache size to prevent memory leaks."""
        key = self._make_key(features)
        with self.lock:
            if key in self.cache:
                return
                
            # Limit cache size (FIFO eviction)
            if len(self.cache) >= self.max_size:
                if self.keys_fifo:
                    oldest_key = self.keys_fifo.pop(0)
                    self.cache.pop(oldest_key, None)
                    
            self.cache[key] = (label, prob.copy())
            self.keys_fifo.append(key)

class LatencyOptimizedInference:
    """
    Orchestrates the entire latency-optimized Speech Emotion Recognition inference pipeline.
    Uses Thread-Safe Caching -> Fast Model Prediction (LightGBM).
    Supports both single-sample and batch predictions in sync and async modes.
    """
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.best_model_dir = os.path.join(project_root, "best_model")
        self.cache = FeatureCache(decimals=4, max_size=1000)
        
        # Load best production model assets with clear error diagnostics
        try:
            self.model = joblib.load(os.path.join(self.best_model_dir, "ser_optuna_lightgbm.joblib"))
            self.imputer = joblib.load(os.path.join(self.best_model_dir, "ser_optuna_imputer.joblib"))
            self.scaler = joblib.load(os.path.join(self.best_model_dir, "ser_optuna_scaler.joblib"))
            self.encoder = joblib.load(os.path.join(self.best_model_dir, "ser_optuna_encoder.joblib"))
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Required model assets not found in {self.best_model_dir}. "
                "Please run 'py src/train_lightgbm.py' first to train and serialize the production assets."
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to load serialized assets from {self.best_model_dir}. "
                f"This may be due to library version inconsistencies: {e}"
            ) from e

        # Patch SimpleImputer version differences if necessary
        if not hasattr(self.imputer, '_fill_dtype'):
            self.imputer._fill_dtype = self.imputer.statistics_.dtype

    def predict(self, X_raw: np.ndarray, confidence_threshold: Optional[float] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Synchronously predicts emotion for single or batch features using cache and LightGBM.
        
        Args:
            X_raw: Numpy array of shape (48,), (1, 48) for single, or (N, 48) for batch.
            confidence_threshold: Retained for backward compatibility.
            
        Returns:
            A single result dictionary or list of dictionaries.
        """
        start = time.perf_counter()
        
        # 1. Normalize dimensions
        X_arr = np.asarray(X_raw)
        is_single = False
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
            is_single = True
        elif X_arr.ndim == 2 and X_arr.shape[0] == 1:
            is_single = True
            
        N = X_arr.shape[0]
        results = [None] * N
        miss_indices = []
        
        # 2. Check Cache
        for i in range(N):
            row = X_arr[i:i+1]
            cached = self.cache.get(row)
            if cached:
                label, prob, _ = cached
                results[i] = {
                    "label": label,
                    "probability": prob,
                    "latency_ms": 0.0,  # Updated globally at the end
                    "cache_hit": True,
                    "method": "cached_match"
                }
            else:
                miss_indices.append(i)
                
        # 3. Model Prediction for Misses (Vectorized Batch)
        if miss_indices:
            try:
                X_miss = X_arr[miss_indices]
                
                # Replace infinities with NaN to allow imputation and scaling
                X_miss_clean = np.where(np.isinf(X_miss), np.nan, X_miss)
                
                # Preprocess batch
                X_imputed = self.imputer.transform(X_miss_clean)
                X_scaled = self.scaler.transform(X_imputed)
                
                # Predict batch (suppressing warnings about feature names mismatch)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    proba_miss = self.model.predict_proba(X_scaled)
                    
                pred_indices = np.argmax(proba_miss, axis=1)
                labels_miss = self.encoder.inverse_transform(pred_indices)
                
                # Cache results and populate output
                for k, original_idx in enumerate(miss_indices):
                    label = labels_miss[k]
                    proba = proba_miss[k]
                    
                    self.cache.set(X_arr[original_idx:original_idx+1], label, proba)
                    
                    results[original_idx] = {
                        "label": label,
                        "probability": proba,
                        "latency_ms": 0.0,  # Updated globally at the end
                        "cache_hit": False,
                        "method": "standalone_lightgbm"
                    }
            except Exception as e:
                # Handle unexpected model inference errors gracefully
                err_msg = f"Inference pipeline failure: {e}"
                for original_idx in miss_indices:
                    results[original_idx] = {
                        "label": "unknown",
                        "probability": np.zeros(len(self.encoder.classes_)),
                        "latency_ms": 0.0,
                        "cache_hit": False,
                        "method": "failed",
                        "error": err_msg
                    }
                    
        # 4. Finalize Latency Metrics
        elapsed = time.perf_counter() - start
        total_latency_ms = elapsed * 1000.0
        
        # If batch prediction, distribute average latency or use total latency
        latency_val = total_latency_ms if is_single else (total_latency_ms / N)
        for res in results:
            res["latency_ms"] = latency_val
            
        if is_single:
            return results[0]
        return results

    async def predict_optimized(self, X_raw: np.ndarray, confidence_threshold: Optional[float] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Asynchronously predicts emotion by running the CPU-bound prediction in a thread pool.
        This complies with MIND-FLOW rules to keep the UI thread completely responsive.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return await loop.run_in_executor(None, self.predict, X_raw, confidence_threshold)

    def predict_sequential_tools(self, X_raw: np.ndarray) -> Dict[str, Any]:
        """Runs standard prediction on a single sample (Un-optimized reference)."""
        start = time.perf_counter()
        X_arr = np.asarray(X_raw)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
            
        X_clean = np.where(np.isinf(X_arr), np.nan, X_arr)
        
        # 1. Preprocess
        X_imputed = self.imputer.transform(X_clean)
        X_scaled = self.scaler.transform(X_imputed)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            proba = self.model.predict_proba(X_scaled)
            
        pred_idx = np.argmax(proba, axis=1)
        label = self.encoder.inverse_transform(pred_idx)[0]
        
        elapsed = time.perf_counter() - start
        return {
            "label": label,
            "probability": proba[0],
            "latency_ms": elapsed * 1000.0,
            "cache_hit": False,
            "method": "sequential_unoptimized"
        }
