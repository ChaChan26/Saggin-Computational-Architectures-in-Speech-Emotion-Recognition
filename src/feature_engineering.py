import numpy as np
import pandas as pd

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

def engineer_features(X: np.ndarray) -> np.ndarray:
    """
    Engineers new acoustic/prosodic features row-wise to prevent any data leakage.
    Input: X of shape (N, 48)
    Output: X_new of shape (N, 48 + 20)
    """
    df = pd.DataFrame(X, columns=FEATURE_COLS_48)
    
    # 1. Ratios (std to mean, etc.)
    df['F0_ratio'] = df['F0_std'] / (df['F0_mean'] + 1e-5)
    df['F0_range_ratio'] = df['F0_range'] / (df['F0_mean'] + 1e-5)
    df['Energy_ratio'] = df['Energy_ std'] / (df['Energy_ mean'] + 1e-5)
    df['ZCR_ratio'] = df['ZCR_std'] / (df['ZCR_mean'] + 1e-5)
    df['Spectral_centroid_ratio'] = df['Spectral_centroid_std'] / (df['Spectral_centroid_mean'] + 1e-5)
    
    # 2. Cross-domain interactions
    df['F0_energy_interaction'] = df['F0_mean'] * df['Energy_ mean']
    df['F0_zcr_interaction'] = df['F0_mean'] * df['ZCR_mean']
    df['Spectral_energy_interaction'] = df['Spectral_centroid_mean'] * df['Energy_ mean']
    
    # 3. MFCC Summaries
    mfcc_mean_cols = [f'MFCC_C{i}_mean' for i in range(13)]
    mfcc_std_cols = [f'MFCC_C{i}_std' for i in range(13)]
    df['MFCC_mean_all'] = df[mfcc_mean_cols].mean(axis=1)
    df['MFCC_std_all'] = df[mfcc_std_cols].mean(axis=1)
    
    # 4. Delta MFCC Summaries
    delta_mean_cols = [f'Delta_MFCC_C{i}_mean' for i in range(6)]
    delta_std_cols = [f'Delta_MFCC_C{i}_std' for i in range(6)]
    df['Delta_MFCC_mean_all'] = df[delta_mean_cols].mean(axis=1)
    df['Delta_MFCC_std_all'] = df[delta_std_cols].mean(axis=1)
    
    # 5. Energy Log Contrast
    df['Energy_contrast'] = np.log1p(df['Energy_ mean'])
    
    # 6. Spectral Shape
    df['Spectral_shape'] = df['Spectral_centroid_mean'] / (df['Spectral_flux_mean'] + 1e-5)
    
    # 7. Delta-to-Static Ratios
    for i in range(6):
        df[f'Delta_static_ratio_C{i}'] = df[f'Delta_MFCC_C{i}_std'] / (df[f'MFCC_C{i}_std'] + 1e-5)
        
    return df.values
