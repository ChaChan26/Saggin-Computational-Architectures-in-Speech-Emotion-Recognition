import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

def main():
    # 1. Production Pathing: Locate dataset relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    project_root = os.path.dirname(script_dir) if os.path.basename(script_dir).lower() in ["src", "notebooks"] else script_dir
    data_path = os.path.join(project_root, 'dataset', 'all_emotions.csv')
    
    if not os.path.exists(data_path):
        data_path = os.path.join(project_root, 'all_emotions.csv')
        
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Production dataset not found at {data_path}. Ensure 'all_emotions.csv' "
            f"is inside the 'dataset/' folder."
        )
    
    print(f"Loading production training data from: {data_path}...")
    df = pd.read_csv(data_path)
    
    # 2. Strict Feature Space: The 26 highly-informative acoustic descriptors
    features = [
        'F0_mean', 'F0_std', 'F0_range', 
        'Energy_ mean', 'Energy_ std', 
        'ZCR_mean', 'ZCR_std', 
        'Spectral_centroid_mean', 'Spectral_centroid_std', 'Spectral_flux_mean',
        'MFCC_C0_mean', 'MFCC_C1_mean', 'MFCC_C2_mean', 'MFCC_C3_mean', 'MFCC_C5_mean', 'MFCC_C7_mean', 'MFCC_C10_mean',
        'MFCC_C0_std', 'MFCC_C1_std', 'MFCC_C2_std', 'MFCC_C3_std', 'MFCC_C5_std', 'MFCC_C7_std',
        'Delta_MFCC_C0_std', 'Delta_MFCC_C2_std', 'Delta_MFCC_C3_std'
    ]
    
    # Verify feature space presence
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise KeyError(f"Missing required acoustic features from CSV columns: {missing_features}")
        
    X = df[features]
    
    # 3. Dynamic target label detection
    target_options = ['emotion', 'Emotion', 'label', 'class', 'target']
    target_col = None
    for option in target_options:
        if option in df.columns:
            target_col = option
            break
            
    if target_col is None:
        target_col = df.columns[-1]
        print(f"Target Column Fallback: Selected default column '{target_col}' as target class.")
    else:
        print(f"Target column verified: Using column '{target_col}' for emotion tracking.")
        
    y = df[target_col]
    
    # 4. Multi-class Label Encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y.astype(str))
    
    # 5. Stratified 80/20 data partition to preserve class distributions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"Data split complete. Training: {X_train.shape[0]} samples | Testing: {X_test.shape[0]} samples.")
    
    # 6. Natural Dataset Balancing (No artificial multipliers to prevent real-world false positives)
    unique_classes = np.unique(y_train)
    computed_weights = compute_class_weight(
        class_weight='balanced', 
        classes=unique_classes, 
        y=y_train
    )
    class_weight_dict = dict(zip(unique_classes, computed_weights))
    
    print("\nCalculated Dataset Class Weights:")
    for cls_idx, weight in class_weight_dict.items():
        print(f"  {le.classes_[cls_idx]}: {weight:.4f}")

    # 7. Model configuration using tuned hyper-parameters
    print("\nConfiguring LightGBM production classifier...")
    model = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        class_weight=class_weight_dict,  # Uses data-driven balanced weights only
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    # 8. Training loop
    print("Fitting model across acoustic feature spaces...")
    model.fit(X_train, y_train)
    
    # 9. Production Serialization: Exporting model and encoder artifacts
    print("\nSerializing production artifacts...")
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_filename = os.path.join(models_dir, 'ser_production_model.joblib')
    encoder_filename = os.path.join(models_dir, 'ser_label_encoder.joblib')
    
    joblib.dump(model, model_filename)
    joblib.dump(le, encoder_filename)
    print(f"Saved model artifact to: {model_filename}")
    print(f"Saved label encoder to: {encoder_filename}")
    
    # 10. Multi-class Pipeline Evaluation (Standard Baseline)
    print("\nGenerating model evaluation metrics across validation splits...")
    y_pred = model.predict(X_test)
    
    print("\n=================== PRODUCTION BASELINE REPORT ===================")
    print(classification_report(y_test, y_pred, target_names=le.classes_, digits=4))
    
    # 11. Showcase Production Runtime Inference Logic (Threshold Moving Example)
    print("\n=== LIVE RUNTIME INFERENCE SIMULATION (Threshold Moving) ===")
    print("Simulating how a live FastAPI/Flask backend checks raw probabilities instead of blind prediction...")
    
    # Get the raw probability matrix for all test samples
    raw_probabilities = model.predict_proba(X_test)
    
    # Find index for neutral class
    cleaned_classes = [str(c).lower().strip() for c in le.classes_]
    if 'neutral' in cleaned_classes:
        neutral_idx = cleaned_classes.index('neutral')
        
        # Production Runtime Rule: If the model is at least 35% confident a user is neutral, 
        # default to neutral instead of letting close low-arousal classes take over.
        CUSTOM_NEUTRAL_THRESHOLD = 0.35
        
        simulated_predictions = []
        for prob_distribution in raw_probabilities:
            # If neutral probability crosses our custom runtime threshold, intercept and select it
            if prob_distribution[neutral_idx] >= CUSTOM_NEUTRAL_THRESHOLD:
                simulated_predictions.append(neutral_idx)
            else:
                # Otherwise, select the standard mathematical maximum probability
                simulated_predictions.append(np.argmax(prob_distribution))
                
        print(f"\n=================== RUNTIME THRESHOLD MOVING REPORT ({CUSTOM_NEUTRAL_THRESHOLD * 100}% Neutral Rule) ===================")
        print(classification_report(y_test, simulated_predictions, target_names=le.classes_, digits=4))
    else:
        print("Skipping threshold simulation: 'neutral' class label not found.")

if __name__ == '__main__':
    main()