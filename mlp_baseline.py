import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, f1_score, cohen_kappa_score

# -------------------------------------------------------------
# 1. Dataset Class for PyTorch
# -------------------------------------------------------------
class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -------------------------------------------------------------
# 2. MLP Architecture
# -------------------------------------------------------------
class EmotionMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(EmotionMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)

# -------------------------------------------------------------
# 3. Data Processing & Main Execution Pipeline
# -------------------------------------------------------------
def main():
    # Detect GPU / CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Path configuration
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.normpath(os.path.join(base_dir, "dataset", "all_emotions.csv"))
    if not os.path.exists(csv_path):
        csv_path = os.path.normpath(os.path.join(base_dir, "all_emotions.csv"))
        
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Please check your folder structure.")
        
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset shape: {df.shape}")
    
    # Target column extraction (case-insensitive check)
    target_col = "label"
    if target_col not in df.columns and "Label" in df.columns:
        target_col = "Label"
        
    # Clean null targets
    df_cleaned = df.dropna(subset=[target_col]).copy()
    df_cleaned = df_cleaned[df_cleaned[target_col].astype(str).str.strip().str.lower() != "nan"]
    
    # Feature columns matching notebook setup
    feature_cols = [
        "F0_mean", "F0_std", "F0_range", "Energy_ mean", "Energy_ std", 
        "ZCR_mean", "ZCR_std", "Spectral_centroid_mean", "Spectral_centroid_std", 
        "Spectral_flux_mean", "MFCC_C0_mean", "MFCC_C1_mean", "MFCC_C2_mean", 
        "MFCC_C3_mean", "MFCC_C5_mean", "MFCC_C7_mean", "MFCC_C10_mean", 
        "MFCC_C0_std", "MFCC_C1_std", "MFCC_C2_std", "MFCC_C3_std", 
        "MFCC_C5_std", "MFCC_C7_std", "Delta_MFCC_C0_std", "Delta_MFCC_C2_std", 
        "Delta_MFCC_C3_std"
    ]
    
    # Impute missing values with column median
    for col in feature_cols:
        s = pd.to_numeric(df_cleaned[col], errors="coerce")
        s = s.replace([np.inf, -np.inf], np.nan)
        med = s.median()
        if pd.isna(med):
            med = 0.0
        df_cleaned[col] = s.fillna(med)
        
    X = df_cleaned[feature_cols].values
    y = df_cleaned[target_col].astype(str).str.strip().values
    
    # Label Encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    print(f"Classes: {le.classes_}")
    
    # Stratified Train/Val/Test Split (70% Train, 15% Val, 15% Test)
    random_state = 42
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y_encoded, test_size=0.15, random_state=random_state, stratify=y_encoded
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1765, random_state=random_state, stratify=y_train_val
    )
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Train Shape: {X_train_scaled.shape} | Val Shape: {X_val_scaled.shape} | Test Shape: {X_test_scaled.shape}")
    
    # DataLoader creation
    batch_size = 128
    train_dataset = EmotionDataset(X_train_scaled, y_train)
    val_dataset = EmotionDataset(X_val_scaled, y_val)
    test_dataset = EmotionDataset(X_test_scaled, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model instantiation
    model = EmotionMLP(input_dim=len(feature_cols), num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Training Loop parameters
    epochs = 50
    best_val_loss = float("inf")
    patience = 7
    patience_counter = 0
    best_model_path = os.path.join(base_dir, "best_mlp_model.pth")
    
    print("\n--- Starting Training ---")
    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_loss = 0.0
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * data.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * data.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == targets.data)
                
        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break
                
    # -------------------------------------------------------------
    # 4. Evaluation Phase
    # -------------------------------------------------------------
    print("\n--- Evaluating on Test Set ---")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        
    model.eval()
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(targets.numpy())
            
    # Calculate performance metrics
    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)
    
    print("\n=== Test Set Classification Report ===")
    print(classification_report(test_targets, test_preds, target_names=le.classes_))
    
    f1w = f1_score(test_targets, test_preds, average="weighted")
    kappa = cohen_kappa_score(test_targets, test_preds)
    
    print(f"Final Test F1 (weighted): {f1w:.4f}")
    print(f"Final Test Cohen Kappa:   {kappa:.4f}")

if __name__ == "__main__":
    main()
