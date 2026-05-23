"""
classical_model.py — Classical CNN Baseline using MobileNetV2
==============================================================
Architecture:
    Input (224x224x3)
    → MobileNetV2 (pre-trained ImageNet, frozen initially)
    → GlobalAveragePooling2D → 1280 features
    → Dense(256, relu) + Dropout(0.3)
    → Dense(1, sigmoid) → PCOS / Not PCOS

Training Strategy:
    Phase 1: Train only top layers (10 epochs, lr=1e-4)
    Phase 2: Fine-tune last 30 MobileNetV2 layers (20 epochs, lr=1e-5)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2

from src.config import (
    IMG_SIZE, DROPOUT_RATE, DENSE_UNITS,
    FINE_TUNE_LAYERS
)


def build_classical_model(freeze_backbone=True):
    """
    Build the classical baseline model using MobileNetV2 transfer learning.
    
    Args:
        freeze_backbone: If True, freeze all MobileNetV2 layers.
                        Set False for fine-tuning phase.
    
    Returns:
        model: Compiled Keras model
    """
    # ── Step 1: Load pre-trained MobileNetV2 ──
    # include_top=False removes the original classification head
    # We only want the feature extraction layers
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,           # Remove classification head
        weights='imagenet'           # Pre-trained on ImageNet
    )
    
    # Freeze/unfreeze backbone
    base_model.trainable = not freeze_backbone
    
    # ── Step 2: Build classification head ──
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='input_image')
    
    # Pass through backbone
    x = base_model(inputs, training=False)  # training=False keeps BatchNorm in inference mode
    
    # Global Average Pooling: (7,7,1280) → (1280,)
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Dense classification layers
    x = layers.Dense(DENSE_UNITS, activation='relu', name='dense_256')(x)
    x = layers.Dropout(DROPOUT_RATE, name='dropout')(x)
    
    # Output: single neuron with sigmoid for binary classification
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    # ── Step 3: Create model ──
    model = Model(inputs=inputs, outputs=outputs, name='PCOS_Classical_CNN')
    
    return model, base_model


def unfreeze_for_finetuning(model, base_model, num_layers=FINE_TUNE_LAYERS):
    """
    Unfreeze the last `num_layers` of the backbone for fine-tuning.
    This allows the model to adapt pre-trained features to ultrasound images.
    
    Args:
        model: The full model
        base_model: The MobileNetV2 backbone
        num_layers: Number of layers to unfreeze from the end
    """
    # Unfreeze backbone
    base_model.trainable = True
    
    # Freeze all layers except the last `num_layers`
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    
    # Count trainable parameters
    trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    print(f"[FINETUNE] Unfroze last {num_layers} layers. Trainable params: {trainable:,}")


def build_feature_extractor(model):
    """
    Build a feature extractor model from the trained classical model.
    Outputs the 1280-dim feature vector (before the Dense layers).
    
    This is used to extract features for the quantum classifier.
    
    Args:
        model: Trained classical model
        
    Returns:
        feature_model: Model that outputs 1280-dim features
    """
    # Get the GlobalAveragePooling layer output
    feature_layer = model.get_layer('global_avg_pool')
    feature_model = Model(
        inputs=model.input,
        outputs=feature_layer.output,
        name='Feature_Extractor'
    )
    
    print(f"[FEATURES] Feature extractor outputs shape: {feature_model.output_shape}")
    return feature_model


def get_model_summary(model):
    """Print a clean model summary."""
    print("\n" + "=" * 60)
    print("  CLASSICAL MODEL ARCHITECTURE")
    print("=" * 60)
    model.summary(print_fn=lambda x: print(f"  {x}"))
    print("=" * 60)
