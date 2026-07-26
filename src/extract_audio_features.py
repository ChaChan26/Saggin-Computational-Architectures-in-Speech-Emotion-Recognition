"""
Core Audio Feature Extractor for Speech Emotion Recognition (SER) Pipeline.

Extracts the exact 48 tabular acoustic features matching dataset/all_emotions.csv
and FEATURE_COLS_48 in src/feature_engineering.py.
"""

import io
import os
from typing import Union, Tuple
import numpy as np
import librosa
import soundfile as sf

FEATURE_COLS_48 = [
    'F0_mean', 'F0_std', 'F0_range', 'Energy_ mean', 'Energy_ std',
    'ZCR_mean', 'ZCR_std', 'Spectral_centroid_mean', 'Spectral_centroid_std',
    'Spectral_flux_mean', 'MFCC_C0_mean', 'MFCC_C1_mean', 'MFCC_C2_mean',
    'MFCC_C3_mean', 'MFCC_C4_mean', 'MFCC_C5_mean', 'MFCC_C6_mean',
    'MFCC_C7_mean', 'MFCC_C8_mean', 'MFCC_C9_mean', 'MFCC_C10_mean',
    'MFCC_C11_mean', 'MFCC_C12_mean', 'MFCC_C0_std', 'MFCC_C1_std',
    'MFCC_C2_std', 'MFCC_C3_std', 'MFCC_C4_std', 'MFCC_C5_std',
    'MFCC_C6_std', 'MFCC_C7_std', 'MFCC_C8_std', 'MFCC_C9_std',
    'MFCC_C10_std', 'MFCC_C11_std', 'MFCC_C12_std', 'Delta_MFCC_C0_mean',
    'Delta_MFCC_C1_mean', 'Delta_MFCC_C2_mean', 'Delta_MFCC_C3_mean',
    'Delta_MFCC_C4_mean', 'Delta_MFCC_C5_mean', 'Delta_MFCC_C0_std',
    'Delta_MFCC_C1_std', 'Delta_MFCC_C2_std', 'Delta_MFCC_C3_std',
    'Delta_MFCC_C4_std', 'Delta_MFCC_C5_std'
]

def load_audio_signal(
    audio_input: Union[str, bytes, np.ndarray],
    target_sr: int = 16000
) -> Tuple[np.ndarray, int]:
    """
    Loads raw audio signal from file path, raw bytes, or numpy array.
    Converts stereo to mono and resamples to target_sr (default 16kHz).
    """
    if isinstance(audio_input, np.ndarray):
        y = audio_input.copy()
        sr = target_sr
    elif isinstance(audio_input, bytes):
        buffer = io.BytesIO(audio_input)
        try:
            y, sr = sf.read(buffer)
        except Exception:
            buffer.seek(0)
            y, sr = librosa.load(buffer, sr=target_sr, mono=True)
    elif isinstance(audio_input, str):
        if not os.path.exists(audio_input):
            raise FileNotFoundError(f"Audio file not found: {audio_input}")
        try:
            y, sr = sf.read(audio_input)
        except Exception:
            y, sr = librosa.load(audio_input, sr=target_sr, mono=True)
    else:
        raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

    # Ensure 1D float array (mono)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    y = y.astype(np.float32)

    # Resample if required
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return y, sr

def extract_48_acoustic_features(
    audio_input: Union[str, bytes, np.ndarray],
    sample_rate: int = 16000
) -> np.ndarray:
    """
    Extracts 48 acoustic descriptors from an audio signal.
    Returns: numpy array of shape (1, 48) aligned with FEATURE_COLS_48.
    """
    y, sr = load_audio_signal(audio_input, target_sr=sample_rate)

    # Handle empty or near-zero silent audio
    if len(y) == 0 or np.max(np.abs(y)) < 1e-7:
        return np.zeros((1, 48), dtype=np.float64)

    # 1. Fundamental Frequency (F0) using pyin / yin
    try:
        f0, _, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'), # ~65 Hz
            fmax=librosa.note_to_hz('C7'), # ~2093 Hz
            sr=sr
        )
    except Exception:
        f0 = None

    if f0 is not None:
        f0_valid = f0[~np.isnan(f0)]
    else:
        f0_valid = np.array([], dtype=np.float64)

    if len(f0_valid) > 0:
        f0_mean = float(np.mean(f0_valid))
        f0_std = float(np.std(f0_valid))
        f0_range = float(np.ptp(f0_valid))
    else:
        f0_mean = 0.0
        f0_std = 0.0
        f0_range = 0.0

    # 2. RMS Energy
    rms = librosa.feature.rms(y=y)[0]
    energy_mean = float(np.mean(rms))
    energy_std = float(np.std(rms))

    # 3. Zero Crossing Rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    zcr_mean = float(np.mean(zcr))
    zcr_std = float(np.std(zcr))

    # 4. Spectral Centroid & Flux
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spec_cent_mean = float(np.mean(spec_cent))
    spec_cent_std = float(np.std(spec_cent))

    # Spectral Flux (Onset envelope strength mean)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    spec_flux_mean = float(np.mean(onset_env)) if len(onset_env) > 0 else 0.0

    # 5. MFCCs C0-C12 (13 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13) # Shape (13, T)
    mfcc_means = np.mean(mfcc, axis=1).tolist() # 13 values
    mfcc_stds = np.std(mfcc, axis=1).tolist()   # 13 values

    # 6. Delta MFCCs C0-C5 (first 6 coefficients)
    delta_mfcc = librosa.feature.delta(mfcc[:6]) # Shape (6, T)
    delta_means = np.mean(delta_mfcc, axis=1).tolist() # 6 values
    delta_stds = np.std(delta_mfcc, axis=1).tolist()   # 6 values

    # Assemble 48 feature vector in exact order
    features = [
        f0_mean, f0_std, f0_range,
        energy_mean, energy_std,
        zcr_mean, zcr_std,
        spec_cent_mean, spec_cent_std,
        spec_flux_mean,
        *mfcc_means,
        *mfcc_stds,
        *delta_means,
        *delta_stds
    ]

    assert len(features) == 48, f"Extracted {len(features)} features, expected 48"

    feat_arr = np.array(features, dtype=np.float64).reshape(1, -1)
    # Clean infinities or NaNs if any remain
    return np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0)
