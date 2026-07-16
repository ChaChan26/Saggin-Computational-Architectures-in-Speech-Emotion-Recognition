import os
import sys
import numpy as np
import pandas as pd
import threading
import warnings
import time

# Add project root to python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.predict_latency_optimized import LatencyOptimizedInference

def resolve_csv_path(base_dir: str) -> str:
    search_paths = [
        os.path.normpath(os.path.join(base_dir, "dataset", "all_emotions.csv")),
        os.path.normpath(os.path.join(base_dir, "all_emotions.csv")),
        os.path.normpath(os.path.join(os.path.dirname(base_dir), "dataset", "all_emotions.csv")),
        os.path.normpath(os.path.join(os.path.dirname(base_dir), "all_emotions.csv")),
    ]
    for path in search_paths:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"Could not find all_emotions.csv. Searched in: {search_paths}")

def run_tests():
    print("=" * 80)
    print("      INFERENCE PIPELINE ROBUSTNESS & PERFORMANCE VALIDATION SUITE")
    print("=" * 80)
    
    # 1. Initialize pipeline
    print("1. Initializing LatencyOptimizedInference...")
    try:
        pipeline = LatencyOptimizedInference(project_root)
        print("   [SUCCESS] Loaded all model assets successfully.")
    except Exception as e:
        print(f"   [FAIL] Could not load model assets: {e}")
        return

    # 2. Load a sample row
    csv_path = resolve_csv_path(project_root)
    df = pd.read_csv(csv_path)
    target_col = "label" if "label" in df.columns else "Label"
    feature_cols = [col for col in df.columns if col not in [target_col]]
    row = df.dropna(subset=[target_col]).iloc[0]
    sample_vec = pd.to_numeric(row[feature_cols], errors="coerce").values.reshape(1, -1)
    sample_vec = np.nan_to_num(sample_vec, nan=0.0)
    
    # 3. Test warning suppression
    print("\n2. Testing Warning Suppression...")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = pipeline.predict(sample_vec)
        # Check if any user warnings from lightgbm were raised
        lgb_warnings = [warning for warning in w if "lightgbm" in str(warning.message).lower() or "feature names" in str(warning.message).lower()]
        if len(lgb_warnings) == 0:
            print("   [SUCCESS] No LightGBM feature names warnings raised during prediction.")
        else:
            print(f"   [WARNING] Mismatch warnings detected: {[str(warn.message) for warn in lgb_warnings]}")

    # 4. Test cache hit and cache miss logic
    print("\n3. Testing Cache Hit / Miss Logic...")
    pipeline.cache.cache.clear()
    pipeline.cache.keys_fifo.clear()
    
    res1 = pipeline.predict(sample_vec)
    assert not res1["cache_hit"], "First run should be a cache miss"
    assert res1["method"] == "standalone_lightgbm", "First run should use LightGBM model"
    
    res2 = pipeline.predict(sample_vec)
    assert res2["cache_hit"], "Second run should be a cache hit"
    assert res2["method"] == "cached_match", "Second run should use cache"
    assert res1["label"] == res2["label"], "Cached label must match predicted label"
    assert np.allclose(res1["probability"], res2["probability"]), "Cached probability must match predicted probability"
    print("   [SUCCESS] Cache hit and miss logic verified.")

    # 5. Test NaN and Inf handling in cache key generator
    print("\n4. Testing NaN and Inf Handling in Cache Key Generator...")
    vec_nan1 = sample_vec.copy()
    vec_nan1[0, 0] = np.nan
    vec_nan2 = sample_vec.copy()
    vec_nan2[0, 0] = np.nan
    
    pipeline.cache.cache.clear()
    pipeline.cache.keys_fifo.clear()
    
    res_nan1 = pipeline.predict(vec_nan1)
    res_nan2 = pipeline.predict(vec_nan2)
    assert res_nan2["cache_hit"], "NaN vector should hit cache on second prediction"
    print("   [SUCCESS] NaN vector successfully cached and matched.")
    
    vec_inf = sample_vec.copy()
    vec_inf[0, 0] = np.inf
    res_inf1 = pipeline.predict(vec_inf)
    res_inf2 = pipeline.predict(vec_inf)
    assert res_inf2["cache_hit"], "Inf vector should hit cache on second prediction"
    print("   [SUCCESS] Inf vector successfully cached and matched.")

    # 6. Test Single-sample vs Batch Equivalence
    print("\n5. Testing Single-sample vs Batch Equivalence...")
    rows = df.dropna(subset=[target_col]).iloc[10:15]
    batch_vec = rows[feature_cols].apply(pd.to_numeric, errors="coerce").values
    batch_vec = np.nan_to_num(batch_vec, nan=0.0)
    
    pipeline.cache.cache.clear()
    pipeline.cache.keys_fifo.clear()
    
    single_results = []
    for i in range(5):
        single_results.append(pipeline.predict(batch_vec[i:i+1]))
        
    pipeline.cache.cache.clear()
    pipeline.cache.keys_fifo.clear()
    batch_results = pipeline.predict(batch_vec)
    
    assert len(batch_results) == 5, "Batch prediction should return 5 results"
    for i in range(5):
        assert batch_results[i]["label"] == single_results[i]["label"], f"Row {i} label mismatch"
        assert np.allclose(batch_results[i]["probability"], single_results[i]["probability"]), f"Row {i} probability mismatch"
    print("   [SUCCESS] Batch and single predictions are mathematically identical.")

    # 7. Test Thread Safety
    print("\n6. Testing Thread Safety...")
    errors = []
    def worker():
        try:
            for _ in range(50):
                idx = np.random.randint(0, len(batch_vec))
                vec = batch_vec[idx:idx+1]
                res = pipeline.predict(vec)
                assert "label" in res
        except Exception as ex:
            errors.append(ex)
            
    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
        
    if not errors:
        print("   [SUCCESS] Run complete. 10 parallel threads, 500 requests total, 0 errors.")
    else:
        print(f"   [FAIL] Thread safety test failed with {len(errors)} errors: {errors[0]}")
        return

    # 8. Latency Benchmark
    print("\n7. Latency Benchmarks:")
    pipeline.cache.cache.clear()
    t_start = time.perf_counter()
    pipeline.predict(sample_vec)
    t_miss = (time.perf_counter() - t_start) * 1000
    
    t_start = time.perf_counter()
    pipeline.predict(sample_vec)
    t_hit = (time.perf_counter() - t_start) * 1000
    
    large_batch_vec = df.dropna(subset=[target_col]).iloc[:1000][feature_cols].apply(pd.to_numeric, errors="coerce").values
    large_batch_vec = np.nan_to_num(large_batch_vec, nan=0.0)
    pipeline.cache.cache.clear()
    t_start = time.perf_counter()
    pipeline.predict(large_batch_vec)
    t_batch_total = (time.perf_counter() - t_start) * 1000
    t_batch_per_sample = t_batch_total / len(large_batch_vec)
    
    print(f"   - Single Sample Cache Miss Latency: {t_miss:.4f} ms")
    print(f"   - Single Sample Cache Hit Latency:  {t_hit:.4f} ms")
    print(f"   - Vectorized Batch Prediction:       {t_batch_total:.2f} ms for {len(large_batch_vec)} samples ({t_batch_per_sample:.4f} ms/sample)")
    print("=" * 80)

if __name__ == "__main__":
    run_tests()
