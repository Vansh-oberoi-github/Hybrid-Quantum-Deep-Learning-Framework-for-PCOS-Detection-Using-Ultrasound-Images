"""
train.py — Training Pipeline for Classical & Hybrid Models
============================================================
Handles:
    1. Classical model: Two-phase transfer learning
    2. Hybrid model: Feature extraction → PCA → VQC training
    3. Training history logging and model checkpoint saving
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)

from src.config import (
    LEARNING_RATE, FINE_TUNE_LR, EPOCHS_FROZEN, EPOCHS_FINETUNE,
    QUANTUM_EPOCHS, QUANTUM_BATCH_SIZE, MODEL_DIR, RESULTS_DIR
)
from src.classical_model import (
    build_classical_model, unfreeze_for_finetuning,
    build_feature_extractor, get_model_summary
)
from src.hybrid_model import (
    extract_features, fit_pca_scaler, transform_features,
    build_hybrid_model, save_pca_scaler
)


def get_callbacks(model_name="classical"):
    """
    Create training callbacks.
    
    Args:
        model_name: Prefix for saved model files
        
    Returns:
        List of Keras callbacks
    """
    callbacks = [
        # Save best model based on validation accuracy
        ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, f"{model_name}_best.keras"),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        # Stop if no improvement for 5 epochs
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    return callbacks


def train_classical_model(train_gen, val_gen):
    """
    Train the classical CNN model in two phases:
        Phase 1: Frozen backbone (fast, learns classification head)
        Phase 2: Fine-tune backbone (slow, adapts to ultrasound domain)
    
    Args:
        train_gen: Training data generator
        val_gen: Validation data generator
        
    Returns:
        model: Trained model
        base_model: MobileNetV2 backbone (for fine-tuning reference)
        history: Combined training history
    """
    print("\n" + "=" * 60)
    print("  TRAINING CLASSICAL CNN MODEL")
    print("=" * 60)
    
    # ── Phase 1: Frozen backbone ──
    print("\n📌 Phase 1: Training classification head (backbone frozen)")
    model, base_model = build_classical_model(freeze_backbone=True)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    get_model_summary(model)
    
    history1 = model.fit(
        train_gen,
        epochs=EPOCHS_FROZEN,
        validation_data=val_gen,
        callbacks=get_callbacks("classical_phase1"),
        verbose=1
    )
    
    # ── Phase 2: Fine-tune backbone ──
    print(f"\n📌 Phase 2: Fine-tuning backbone (last {30} layers unfrozen)")
    unfreeze_for_finetuning(model, base_model)
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    history2 = model.fit(
        train_gen,
        epochs=EPOCHS_FINETUNE,
        validation_data=val_gen,
        callbacks=get_callbacks("classical_best"),
        verbose=1
    )
    
    # Save final model
    model.save(os.path.join(MODEL_DIR, "classical_final.keras"))
    print(f"\n✅ Classical model saved to {MODEL_DIR}/classical_final.keras")
    
    # Combine histories
    combined_history = {}
    for key in history1.history:
        combined_history[key] = history1.history[key] + history2.history[key]
    
    # Save history
    # Convert numpy values to Python floats for JSON serialization
    history_json = {k: [float(v) for v in vals] for k, vals in combined_history.items()}
    with open(os.path.join(RESULTS_DIR, "classical_history.json"), 'w') as f:
        json.dump(history_json, f, indent=2)
    
    return model, base_model, combined_history


def train_hybrid_model(classical_model, train_gen, val_gen, test_gen):
    """
    Train the hybrid quantum-classical model:
        1. Extract CNN features from trained classical backbone
        2. Fit PCA on training features
        3. Transform all features to quantum-ready format
        4. Train VQC using Adam optimizer
    
    Args:
        classical_model: Trained classical CNN model
        train_gen, val_gen, test_gen: Data generators
        
    Returns:
        hybrid_model: Trained hybrid model
        history: Training history
        pca_data: Dict with transformed features and labels
    """
    print("\n" + "=" * 60)
    print("  TRAINING HYBRID QUANTUM-CLASSICAL MODEL")
    print("=" * 60)
    
    # ── Step 1: Extract CNN features ──
    print("\n📌 Step 1: Extracting CNN features using trained backbone...")
    feature_model = build_feature_extractor(classical_model)
    
    print("\n  Training features:")
    train_features, train_labels = extract_features(feature_model, train_gen)
    print("  Validation features:")
    val_features, val_labels = extract_features(feature_model, val_gen)
    print("  Test features:")
    test_features, test_labels = extract_features(feature_model, test_gen)
    
    # ── Step 2: PCA dimensionality reduction ──
    print("\n📌 Step 2: PCA dimensionality reduction...")
    scaler, pca, min_vals, max_vals = fit_pca_scaler(train_features)
    save_pca_scaler(scaler, pca, min_vals, max_vals)
    
    # Transform all splits
    train_quantum = transform_features(train_features, scaler, pca, min_vals, max_vals)
    val_quantum   = transform_features(val_features, scaler, pca, min_vals, max_vals)
    test_quantum  = transform_features(test_features, scaler, pca, min_vals, max_vals)
    
    print(f"  Quantum features shape: {train_quantum.shape}")
    
    # ── Step 3: Build and train VQC ──
    print("\n📌 Step 3: Training Variational Quantum Circuit...")
    hybrid_model = build_hybrid_model()
    
    # Train the quantum model
    history = hybrid_model.fit(
        train_quantum, train_labels,
        epochs=QUANTUM_EPOCHS,
        batch_size=QUANTUM_BATCH_SIZE,
        validation_data=(val_quantum, val_labels),
        callbacks=[
            EarlyStopping(
                monitor='val_loss',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(MODEL_DIR, "hybrid_best.keras"),
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            )
        ],
        verbose=1
    )
    
    # Save model and history
    hybrid_model.save(os.path.join(MODEL_DIR, "hybrid_final.keras"))
    
    history_json = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(os.path.join(RESULTS_DIR, "hybrid_history.json"), 'w') as f:
        json.dump(history_json, f, indent=2)
    
    print(f"\n✅ Hybrid quantum model saved to {MODEL_DIR}/hybrid_final.keras")
    
    # Package data for evaluation
    pca_data = {
        "train_features": train_quantum, "train_labels": train_labels,
        "val_features": val_quantum,     "val_labels": val_labels,
        "test_features": test_quantum,   "test_labels": test_labels,
    }
    
    return hybrid_model, history.history, pca_data
