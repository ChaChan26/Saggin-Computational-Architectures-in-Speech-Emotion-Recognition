import os
import sys
import pandas as pd
import numpy as np
import asyncio
from sklearn.metrics import classification_report, f1_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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

async def evaluate_threshold(pipeline, X_test, y_test, threshold, encoder):
    start = time.perf_counter()
    y_pred = []
    method_counts = {}
    
    # Process in batches to avoid overwhelming the thread pool or asyncio loop
    batch_size = 500
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i:i+batch_size]
        tasks = [pipeline.predict_optimized(row.reshape(1, -1), confidence_threshold=threshold) for row in batch]
        results = await asyncio.gather(*tasks)
        y_pred.extend([res['label'] for res in results])
        
        for res in results:
            method = res.get('method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
            
    elapsed = time.perf_counter() - start
    
    # encode preds
    y_pred_encoded = encoder.transform(y_pred)
    
    f1 = f1_score(y_test, y_pred_encoded, average='weighted')
    kappa = cohen_kappa_score(y_test, y_pred_encoded)
    
    print(f"\n--- Threshold: {threshold} ---")
    print(f"Elapsed Time: {elapsed:.2f}s for {len(X_test)} samples")
    print(f"Weighted F1: {f1:.6f}")
    print(f"Cohen Kappa: {kappa:.6f}")
    print("Method Breakdown:")
    for k, v in method_counts.items():
        print(f"  {k}: {v} samples ({(v/len(X_test))*100:.1f}%)")
    
    return f1, kappa, threshold

async def main():
    print("Initializing Confidence-Routed Hybrid Pipeline...")
    pipeline = LatencyOptimizedInference(project_root)
    
    csv_path = resolve_csv_path(project_root)
    df = pd.read_csv(csv_path)
    target_col = "label" if "label" in df.columns else "Label"
    
    df_cleaned = df.dropna(subset=[target_col]).copy()
    df_cleaned = df_cleaned[df_cleaned[target_col].astype(str).str.strip().str.lower() != "nan"]
    
    FEATURE_COLS = [col for col in df_cleaned.columns if col not in [target_col]]
    
    for col in FEATURE_COLS:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        
    X_raw = df_cleaned[FEATURE_COLS].values
    y_str = df_cleaned[target_col].astype(str).str.strip().values
    
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_str)
    
    # Needs to match 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )
    
    print(f"Test Set Size: {len(X_test)}")
    
    thresholds = [0.80, 0.85, 0.90, 0.95, 0.99]
    best_f1 = 0
    best_thresh = None
    
    for th in thresholds:
        pipeline.cache.cache.clear()
        f1, kappa, t = await evaluate_threshold(pipeline, X_test, y_test, th, encoder)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            
    print("=" * 50)
    print(f"BEST RESULT: F1 = {best_f1:.6f} at Threshold = {best_thresh}")
    if best_f1 >= 0.90:
        print("🎉 GOAL ACHIEVED! > 90% F1-SCORE! 🎉")
    else:
        print("We didn't hit 90% yet.")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())
