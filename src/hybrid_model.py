"""
hybrid_model.py — Hybrid Quantum-Classical Model
==================================================
Combines:
    1. Classical CNN (MobileNetV2) for feature extraction
    2. PCA for dimensionality reduction (1280 → 4)
    3. PennyLane VQC for quantum classification

Pipeline:
    Image → MobileNetV2 → 1280-dim → PCA → 4-dim → VQC → Prediction
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
import os

from src.config import (
    PCA_COMPONENTS, N_QUBITS, QUANTUM_LR,
    MODEL_DIR
)
from src.quantum_circuit import QuantumLayer


def extract_features(feature_model, data_generator):
    """
    Extract CNN features from all images using the feature extractor model.
    
    Args:
        feature_model: Keras model that outputs 1280-dim features
        data_generator: Keras data generator (train/val/test)
        
    Returns:
        features: numpy array of shape (n_samples, 1280)
        labels: numpy array of shape (n_samples,)
    """
    features_list = []
    labels_list = []
    
    # Reset generator
    data_generator.reset()
    n_batches = len(data_generator)
    
    print(f"  Extracting features from {data_generator.samples} images...")
    
    for i in range(n_batches):
        batch_images, batch_labels = data_generator[i]
        batch_features = feature_model.predict(batch_images, verbose=0)
        features_list.append(batch_features)
        labels_list.append(batch_labels)
    
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    
    # Handle case where generator might return more than actual samples
    features = features[:data_generator.samples]
    labels = labels[:data_generator.samples]
    
    print(f"  Extracted features shape: {features.shape}")
    return features, labels


def fit_pca_scaler(train_features):
    """
    Fit PCA and StandardScaler on training features.
    
    Steps:
    1. Standardize features (zero mean, unit variance)
    2. Reduce from 1280 to 4 dimensions using PCA
    3. Calculate min and max bounds for quantum angle encoding
    
    Args:
        train_features: Training features, shape (n_train, 1280)
        
    Returns:
        scaler: Fitted StandardScaler
        pca: Fitted PCA transformer
        min_vals: Minimum values of PCA features
        max_vals: Maximum values of PCA features
    """
    # Standardize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(train_features)
    
    # PCA reduction
    pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
    reduced = pca.fit_transform(features_scaled)
    
    # Calculate min/max ONLY on training data
    min_vals = reduced.min(axis=0)
    max_vals = reduced.max(axis=0)
    
    explained = sum(pca.explained_variance_ratio_) * 100
    print(f"[PCA] Reduced 1280 → {PCA_COMPONENTS} dimensions")
    print(f"      Explained variance: {explained:.1f}%")
    
    return scaler, pca, min_vals, max_vals


def transform_features(features, scaler, pca, min_vals, max_vals):
    """
    Transform features using fitted scaler and PCA.
    Then scale to [0, π] range for quantum angle encoding using TRAINING bounds.
    
    Args:
        features: Raw CNN features, shape (n, 1280)
        scaler: Fitted StandardScaler
        pca: Fitted PCA
        min_vals: Minimum training bounds
        max_vals: Maximum training bounds
        
    Returns:
        quantum_features: Shape (n, PCA_COMPONENTS), range [0, π]
    """
    # Standardize
    scaled = scaler.transform(features)
    
    # PCA
    reduced = pca.transform(scaled)
    
    # Scale to [0, π] using strict training bounds to prevent leakage
    for col in range(reduced.shape[1]):
        col_min = min_vals[col]
        col_max = max_vals[col]
        
        if col_max - col_min > 0:
            # Scale and clip to prevent out-of-bounds angles from unseen test data
            scaled_col = (reduced[:, col] - col_min) / (col_max - col_min)
            scaled_col = np.clip(scaled_col, 0.0, 1.0)
            reduced[:, col] = scaled_col * np.pi
        else:
            reduced[:, col] = 0.0
            
    return reduced.astype(np.float32)


def build_hybrid_model():
    """
    Build the hybrid quantum-classical model for training.
    
    Uses a custom QuantumLayer that wraps the PennyLane circuit
    as a differentiable Keras layer.
    
    Returns:
        hybrid_model: Keras model with quantum layer
    """
    # Build Keras model
    inputs = tf.keras.Input(shape=(PCA_COMPONENTS,), name='pca_features')
    
    # Quantum layer outputs expectation value in [-1, 1]
    x = QuantumLayer(n_qubits=N_QUBITS, name="quantum_vqc")(inputs)
    
    # Map from [-1, 1] to [0, 1] using (x + 1) / 2
    # This acts as our probability output
    outputs = layers.Lambda(
        lambda x: (x + 1.0) / 2.0,
        name='probability_output'
    )(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='PCOS_Hybrid_Quantum')
    
    # Compile with binary crossentropy
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=QUANTUM_LR),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    total_params = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    print(f"[HYBRID] Quantum model built with {total_params} trainable parameters")
    
    return model


def save_pca_scaler(scaler, pca, min_vals, max_vals):
    """Save fitted PCA, scaler, and bounds for inference."""
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODEL_DIR, "pca.pkl"), 'wb') as f:
        pickle.dump(pca, f)
    with open(os.path.join(MODEL_DIR, "pca_bounds.pkl"), 'wb') as f:
        pickle.dump({"min_vals": min_vals, "max_vals": max_vals}, f)
    print("[SAVE] Scaler, PCA, and bounds saved to outputs/models/")


def load_pca_scaler():
    """Load fitted PCA, scaler, and bounds."""
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "pca.pkl"), 'rb') as f:
        pca = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "pca_bounds.pkl"), 'rb') as f:
        bounds = pickle.load(f)
    return scaler, pca, bounds["min_vals"], bounds["max_vals"]
