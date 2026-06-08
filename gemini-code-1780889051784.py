import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

def main():
    # 1. Production Pathing
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'dataset', 'all_emotions.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Production dataset not found at {data_path}.")
    
    print(f"Loading production training data from: {data_path}...")
    df = pd.read_csv(data_path)
    
    # 2. Strict Feature Space
    features = [
        'F0_mean', 'F0_std', 'F0_range', 
        'Energy_ mean', 'Energy_ std', 
        'ZCR_mean', 'ZCR_std', 
        'Spectral_centroid_mean', 'Spectral_centroid_std', 'Spectral_flux_mean',
        'MFCC_C0_mean', 'MFCC_C1_mean', 'MFCC_C2_mean', 'MFCC_C3_mean', 'MFCC_C5_mean', 'MFCC_C7_mean', 'MFCC_C10_mean',
        'MFCC_C0_std', 'MFCC_C1_std', 'MFCC_C2_std', 'MFCC_C3_std', 'MFCC_C5_std', 'MFCC_C7_std',
        'Delta_MFCC_C0_std', 'Delta_MFCC_C2_std', 'Delta_MFCC_C3_std'
    ]
    
    X = df[features]
    
    # 3. Target Verification
    target_options = ['emotion', 'Emotion', 'label', 'class', 'target']
    target_col = next((opt for opt in target_options if opt in df.columns), df.columns[-1])
    y = df[target_col]
    
    # 4. Label Encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y.astype(str))
    
    # 5. Three-Way Stratified Data Split (90 Train / 5 Val / 5 Test)
    # First split: Isolate 90% for training, 10% for temporary holding
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=0.10, random_state=42, stratify=y_encoded
    )
    
    # Second split: Cut the 10% holding set exactly in half to yield 5% Val and 5% Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    print(f"Data split complete.")
    print(f"  Training:   {X_train.shape[0]} samples (90%)")
    print(f"  Validation: {X_val.shape[0]} samples (5%)")
    print(f"  Testing:    {X_test.shape[0]} samples (5%)")
    
    # 6. Feature Scaling (StandardScaler)
    print("\nApplying Standard Scaling...")
    scaler = StandardScaler()
    
    # Fit strictly on training data to prevent data leakage, then transform all sets
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 7. Dataset Balancing
    unique_classes = np.unique(y_train)
    computed_weights = compute_class_weight(
        class_weight='balanced', 
        classes=unique_classes, 
        y=y_train
    )
    class_weight_dict = dict(zip(unique_classes, computed_weights))

    # 8. Model configuration 
    print("\nConfiguring LightGBM production classifier...")
    model = lgb.LGBMClassifier(
        n_estimators=1000, # Increased strictly because early stopping will catch it
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        class_weight=class_weight_dict,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    # 9. Training with Early Stopping via Validation Set
    print("Fitting model across scaled acoustic feature spaces...")
    model.fit(
        X_train_scaled, 
        y_train,
        eval_set=[(X_val_scaled, y_val)],
        eval_metric='multi_logloss',
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True)
        ]
    )
    
    # 10. Production Serialization (Now includes the Scaler)
    print("\nSerializing production artifacts...")
    model_filename = os.path.join(script_dir, 'ser_production_model.joblib')
    encoder_filename = os.path.join(script_dir, 'ser_label_encoder.joblib')
    scaler_filename = os.path.join(script_dir, 'ser_scaler.joblib') # New artifact
    
    joblib.dump(model, model_filename)
    joblib.dump(le, encoder_filename)
    joblib.dump(scaler, scaler_filename)
    print("Saved Model, Encoder, and Scaler artifacts.")
    
    # 11. Multi-class Pipeline Evaluation on 5% Test Set
    print("\n=================== PRODUCTION TEST REPORT ===================")
    y_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred, target_names=le.classes_, digits=4))

if __name__ == '__main__':
    main()