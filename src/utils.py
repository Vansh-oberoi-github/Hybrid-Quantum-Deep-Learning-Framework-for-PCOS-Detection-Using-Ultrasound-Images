"""
utils.py — Helper Utilities
==============================
Shared utility functions for the PCOS detection project.
"""

import os
import random
import numpy as np
import tensorflow as tf

from src.config import RANDOM_SEED


def set_seed(seed=RANDOM_SEED):
    """
    Set random seeds for full reproducibility.
    Ensures consistent results across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # For GPU determinism (may slow down training slightly)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
    print(f"[SEED] Random seed set to {seed} for reproducibility")


def check_gpu():
    """
    Check GPU availability and print device info.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[GPU] Found {len(gpus)} GPU(s):")
        for gpu in gpus:
            print(f"      {gpu.name}")
        # Enable memory growth to avoid OOM
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
    else:
        print("[GPU] No GPU found — using CPU only")
        print("      Classical training will be slower")
        print("      Quantum training is CPU-based regardless")
    
    return len(gpus) > 0


def print_banner(title, char="=", width=60):
    """Print a formatted section banner."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def count_parameters(model):
    """Count and display model parameters."""
    total = sum(tf.keras.backend.count_params(w) for w in model.weights)
    trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable = total - trainable
    
    print(f"  Total params:         {total:>12,}")
    print(f"  Trainable params:     {trainable:>12,}")
    print(f"  Non-trainable params: {non_trainable:>12,}")
    
    return total, trainable
