"""
Unit tests for Speech Emotion Recognition (SER) core audio feature extractor and prediction pipeline.
"""

import os
import sys
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.extract_audio_features import extract_48_acoustic_features, FEATURE_COLS_48
from src.predict_audio import SERAudioPredictor

def test_feature_extraction_shape_and_finite():
    # 2 seconds of 16kHz synthetic audio
    sr = 16000
    t = np.linspace(0, 2.0, sr * 2)
    audio_signal = (np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)

    features = extract_48_acoustic_features(audio_signal, sample_rate=sr)

    assert features.shape == (1, 48), f"Expected shape (1, 48), got {features.shape}"
    assert not np.isnan(features).any(), "Extracted features contain NaN values!"
    assert not np.isinf(features).any(), "Extracted features contain Inf values!"
    assert len(FEATURE_COLS_48) == 48

def test_silent_audio_extraction():
    silence = np.zeros(16000, dtype=np.float32)
    features = extract_48_acoustic_features(silence)

    assert features.shape == (1, 48)
    assert not np.isnan(features).any()

def test_predictor_initialization_and_inference():
    predictor = SERAudioPredictor()
    assert len(predictor.classes) == 6
    expected_emotions = {'anger', 'disgust', 'fear', 'happy', 'neutral', 'sad'}
    assert set(predictor.classes) == expected_emotions

    # Test direct feature prediction
    dummy_features = np.random.randn(1, 48)
    res = predictor.predict_features(dummy_features)

    assert "predicted_label" in res
    assert res["predicted_label"] in expected_emotions
    assert "confidence" in res
    assert 0.0 <= res["confidence"] <= 1.0
    assert "probabilities" in res
    assert len(res["probabilities"]) == 6
    assert abs(sum(res["probabilities"].values()) - 1.0) < 1e-4

def test_audio_pipeline_prediction():
    predictor = SERAudioPredictor()
    sr = 16000
    t = np.linspace(0, 1.5, int(sr * 1.5))
    audio_signal = np.sin(2 * np.pi * 300 * t).astype(np.float32)

    res = predictor.predict_audio(audio_signal)

    assert "predicted_label" in res
    assert "dsp_latency_ms" in res
    assert "total_latency_ms" in res
    assert res["total_latency_ms"] > 0.0

def test_batch_feature_prediction():
    predictor = SERAudioPredictor()
    batch_features = np.random.randn(5, 48)
    results = predictor.predict_features(batch_features)

    assert isinstance(results, list)
    assert len(results) == 5
    for item in results:
        assert "predicted_label" in item
        assert "confidence" in item
        assert "probabilities" in item

def test_non_16k_sample_rate_prediction():
    predictor = SERAudioPredictor()
    sr = 44100 # 44.1kHz audio stream
    t = np.linspace(0, 1.0, sr)
    audio_signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    res = predictor.predict_audio(audio_signal, sample_rate=44100)
    assert "predicted_label" in res

