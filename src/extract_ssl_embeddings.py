import os
import sys
import logging
import time
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
import lightgbm as lgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("SSL_Embeddings_Pipeline")

# Lazy imports for audio libraries to avoid hard crashes if not installed
def load_audio_libraries():
    try:
        import librosa
        import soundfile
        import transformers
        return librosa, soundfile, transformers
    except ImportError as e:
        logger.error(
            f"Failed to import required libraries. Make sure to run: "
            f"pip install transformers librosa soundfile. Error: {e}"
        )
        raise

class SSLEmbeddingExtractor:
    """
    Handles feature extraction from raw audio waveforms using pre-trained SSL models
    like Wav2Vec 2.0 or HuBERT.
    """
    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h", device: Optional[str] = None):
        librosa, soundfile, transformers = load_audio_libraries()
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing SSL Model '{model_name}' on device '{self.device}'...")
        
        try:
            self.processor = transformers.Wav2Vec2Processor.from_pretrained(model_name)
            self.model = transformers.Wav2Vec2Model.from_pretrained(model_name).to(self.device)
            self.model.eval()
            logger.info("SSL Model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def extract_from_waveform(self, waveform: np.ndarray, sampling_rate: int = 16000) -> np.ndarray:
        """
        Extracts a single 768-dimensional mean-pooled embedding vector from an audio waveform.
        """
        librosa, _, _ = load_audio_libraries()
        # 1. Resample to 16kHz if necessary (expected by standard speech foundation models)
        if sampling_rate != 16000:
            logger.debug(f"Resampling audio from {sampling_rate}Hz to 16000Hz...")
            waveform = librosa.resample(waveform, orig_sr=sampling_rate, target_sr=16000)
            
        # Ensure single dimension (mono)
        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=0)

        # 2. Process and extract representations
        inputs = self.processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_values)
            # last_hidden_state shape: [batch_size=1, sequence_length, hidden_dimension=768]
            hidden_states = outputs.last_hidden_state
            # Apply global mean pooling over sequence dimension to extract dynamic temporal features
            mean_pooled = torch.mean(hidden_states, dim=1)
            
        return mean_pooled.squeeze(0).cpu().numpy()

    def extract_from_file(self, file_path: str) -> Optional[np.ndarray]:
        """
        Loads an audio file and extracts its mean-pooled embedding vector.
        """
        librosa, _, _ = load_audio_libraries()
        if not os.path.isfile(file_path):
            logger.error(f"File not found: {file_path}")
            return None
            
        try:
            # Load as mono at native sampling rate
            y, sr = librosa.load(file_path, sr=None, mono=True)
            return self.extract_from_waveform(y, sr)
        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")
            return None

class PyTorchMLP(nn.Module):
    """
    Lightweight downstream classification neural network for 768-dim embeddings.
    """
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)

class EmbeddingDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_pytorch_mlp(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, num_classes: int, epochs: int = 15) -> PyTorchMLP:
    """Trains a simple PyTorch MLP on the extracted embeddings."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PyTorchMLP(input_dim=768, num_classes=num_classes).to(device)
    
    train_dataset = EmbeddingDataset(X_train, y_train)
    test_dataset = EmbeddingDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    logger.info("Training PyTorch MLP on embeddings...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_X)
            
        # Validation Check
        model.eval()
        with torch.no_grad():
            test_X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            val_logits = model(test_X_tensor)
            val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()
            val_f1 = f1_score(y_test, val_preds, average="weighted")
            
        logger.info(f"  Epoch {epoch+1:02d}/{epochs} - Train Loss: {total_loss/len(X_train):.4f} - Val Weighted F1: {val_f1:.4f}")
        
    return model

def train_lightgbm_on_embeddings(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, label_encoder: LabelEncoder):
    """Trains LightGBM classifier on the extracted embeddings."""
    logger.info("Training LightGBM Classifier on embeddings...")
    
    # Simple default hyperparameters for high-dimensional embeddings
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    params = {
        "objective": "multiclass",
        "num_class": len(label_encoder.classes_),
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "feature_fraction": 0.8,
        "verbose": -1,
        "random_state": 42
    }
    
    start = time.perf_counter()
    gbm = lgb.train(
        params,
        train_data,
        num_boost_round=150,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
    )
    logger.info(f"LightGBM trained in {time.perf_counter() - start:.2f}s")
    
    y_pred_probs = gbm.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    print("\n=== LightGBM Embedding Classifier Evaluation ===")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    logger.info(f"Validation Weighted F1: {f1_score(y_test, y_pred, average='weighted'):.4f}")
    return gbm

def run_synthetic_test():
    """
    Self-contained verification test that runs dummy audio inputs through
    the extractor and models to verify code correctness.
    """
    logger.info("=========================================")
    logger.info("   RUNNING SYNTHETIC PIPELINE SELF-TEST  ")
    logger.info("=========================================")
    
    # 1. Initialize Extractor
    try:
        extractor = SSLEmbeddingExtractor(model_name="facebook/wav2vec2-base-960h")
    except Exception as e:
        logger.warning(
            f"Skipping synthetic self-test run because transformers or model files "
            f"cannot be resolved in this sandbox: {e}"
        )
        return
        
    # 2. Generate Synthetic Audio (3 samples of 2 seconds of white noise at 16kHz)
    logger.info("Generating synthetic mock audio waveforms...")
    np.random.seed(42)
    mock_waveforms = [np.random.randn(32000).astype(np.float32) for _ in range(10)]
    mock_labels = ["anger", "sad", "neutral", "happy", "disgust", "fear", "anger", "sad", "neutral", "happy"]
    
    # 3. Extract Embeddings
    logger.info("Running waveforms through Wav2Vec 2.0 Extractor...")
    embeddings = []
    for i, wav in enumerate(mock_waveforms):
        emb = extractor.extract_from_waveform(wav, sampling_rate=16000)
        embeddings.append(emb)
        logger.info(f"  Sample {i+1}/10 Extracted Shape: {emb.shape}")
        
    X = np.stack(embeddings)
    assert X.shape == (10, 768), f"Expected shape (10, 768), got {X.shape}"
    logger.info("Feature extraction verified. Shape is correct.")
    
    # 4. Train classifiers on mock features
    logger.info("Encoding labels...")
    le = LabelEncoder()
    y = le.fit_transform(mock_labels)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    
    # Train mock models
    train_lightgbm_on_embeddings(X_train, X_val, y_train, y_val, le)
    train_pytorch_mlp(X_train, X_val, y_train, y_val, num_classes=len(le.classes_), epochs=3)
    
    logger.info("Synthetic pipeline self-test passed successfully!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_synthetic_test()
    else:
        logger.info(
            "Speech Emotion Recognition SSL Pipeline Utility.\n"
            "Usage:\n"
            "  1. Run self-test: py src/extract_ssl_embeddings.py --test\n"
            "  2. Import SSLEmbeddingExtractor in your training scripts once raw WAVs are available."
        )
