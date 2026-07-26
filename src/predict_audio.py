"""
Core Speech Emotion Recognition (SER) Prediction Pipeline.

Loads Optuna-tuned LightGBM model assets from best_model/ and runs end-to-end inference
on raw audio files (.wav, .mp3, .flac) or pre-extracted 48-feature arrays.
"""

import os
import sys
import time
import argparse
import joblib
import numpy as np
from typing import Dict, Any, Union

# Ensure project root is in python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.extract_audio_features import extract_48_acoustic_features, FEATURE_COLS_48

class SERAudioPredictor:
    """
    Production-grade predictor for Speech Emotion Recognition.
    Handles imputer -> scaler -> LightGBM -> encoder pipeline.
    """

    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = os.path.join(PROJECT_ROOT, "best_model")

        self.model_dir = model_dir
        self._load_assets()

    def _load_assets(self):
        """Loads serialized model, imputer, scaler, and label encoder."""
        model_path = os.path.join(self.model_dir, "ser_optuna_lightgbm.joblib")
        imputer_path = os.path.join(self.model_dir, "ser_optuna_imputer.joblib")
        scaler_path = os.path.join(self.model_dir, "ser_optuna_scaler.joblib")
        encoder_path = os.path.join(self.model_dir, "ser_optuna_encoder.joblib")

        for path in [model_path, imputer_path, scaler_path, encoder_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing required model asset: {path}")

        self.model = joblib.load(model_path)
        self.imputer = joblib.load(imputer_path)
        self.scaler = joblib.load(scaler_path)
        self.encoder = joblib.load(encoder_path)
        self.classes = list(self.encoder.classes_)

    def predict_features(self, X_raw: np.ndarray) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Runs prediction directly on a 48-element acoustic feature vector (single or batch).
        """
        t0 = time.perf_counter()

        X_arr = np.asarray(X_raw, dtype=np.float64)
        is_single = False
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
            is_single = True
        elif X_arr.ndim == 2 and X_arr.shape[0] == 1:
            is_single = True

        if X_arr.shape[1] != 48:
            raise ValueError(f"Expected 48 acoustic features, got shape {X_arr.shape}")

        # Clean NaNs and Infinities
        X_clean = np.where(np.isinf(X_arr), np.nan, X_arr)

        # Pipeline: Impute -> Scale -> Predict
        X_imputed = self.imputer.transform(X_clean)
        X_scaled = self.scaler.transform(X_imputed)

        probas = self.model.predict_proba(X_scaled)
        pred_indices = np.argmax(probas, axis=1)
        predicted_labels = self.encoder.inverse_transform(pred_indices)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        results = []
        for i in range(len(probas)):
            proba_dict = {
                self.classes[j]: float(probas[i][j]) for j in range(len(self.classes))
            }
            results.append({
                "predicted_label": str(predicted_labels[i]),
                "confidence": float(probas[i][pred_indices[i]]),
                "probabilities": proba_dict,
                "latency_ms": elapsed_ms / len(probas)
            })

        return results[0] if is_single else results

    def predict_audio(
        self,
        audio_input: Union[str, bytes, np.ndarray],
        sample_rate: int = 16000
    ) -> Dict[str, Any]:
        """
        Extracts 48 acoustic features from raw audio and predicts emotion.
        """
        t0 = time.perf_counter()

        # Step 1: Feature Extraction
        features = extract_48_acoustic_features(audio_input, sample_rate=sample_rate)
        dsp_time_ms = (time.perf_counter() - t0) * 1000.0

        # Step 2: Model Inference
        result = self.predict_features(features)
        total_time_ms = (time.perf_counter() - t0) * 1000.0

        if isinstance(result, dict):
            result["dsp_latency_ms"] = dsp_time_ms
            result["total_latency_ms"] = total_time_ms
        return result

def main():
    parser = argparse.ArgumentParser(description="Speech Emotion Recognition (SER) Prediction CLI")
    parser.add_argument("audio_path", nargs="?", help="Path to raw audio file (.wav, .mp3, .flac)")
    args = parser.parse_args()

    predictor = SERAudioPredictor()
    print(f"Loaded SER Predictor ({len(predictor.classes)} classes: {predictor.classes})")

    if args.audio_path:
        print(f"\nAnalyzing audio file: {args.audio_path}...")
        res = predictor.predict_audio(args.audio_path)
        print(f"\n=== PREDICTION RESULT ===")
        print(f"Predicted Emotion:  {res['predicted_label'].upper()}")
        print(f"Confidence:         {res['confidence']*100:.2f}%")
        print(f"DSP Latency:        {res['dsp_latency_ms']:.2f} ms")
        print(f"Inference Latency:  {res['latency_ms']:.2f} ms")
        print(f"Total Latency:      {res['total_latency_ms']:.2f} ms")
        print("\nProbability Breakdown:")
        for emotion, prob in res['probabilities'].items():
            bar = '█' * int(prob * 30)
            print(f"  {emotion:<10}: {prob*100:6.2f}% {bar}")
    else:
        print("\nCLI Usage: python src/predict_audio.py <path_to_audio_file>")
        print("Running quick self-test on synthetic sine wave audio signal...")
        sr = 16000
        t = np.linspace(0, 2.0, sr * 2) # 2s 440Hz sine wave
        synth_audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        res = predictor.predict_audio(synth_audio)
        print(f"Self-test prediction: {res['predicted_label'].upper()} ({res['confidence']*100:.2f}% confidence) in {res['total_latency_ms']:.2f} ms")

if __name__ == "__main__":
    main()
