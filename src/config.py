"""
config.py — Global Configuration & Hyperparameters
====================================================
Centralizes ALL settings so you never have to hunt through code to change a value.
Change anything here and it propagates everywhere.
"""

import os

# ─────────────────────────────────────────────
# 1. PATHS
# ─────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Raw dataset locations (your existing data)
DATASET_1_DIR = os.path.join(PROJECT_ROOT, "Dtataset", "dataset_1", "archive", "PCOS")
DATASET_2_DIR = os.path.join(PROJECT_ROOT, "Dtataset", "Dataset_2", "archive", "data")

# Curated dataset (created by our pipeline)
CURATED_DATA_DIR = os.path.join(PROJECT_ROOT, "data_curated")

# Output directories
OUTPUT_DIR    = os.path.join(PROJECT_ROOT, "outputs")
MODEL_DIR     = os.path.join(OUTPUT_DIR, "models")
PLOT_DIR      = os.path.join(OUTPUT_DIR, "plots")
RESULTS_DIR   = os.path.join(OUTPUT_DIR, "results")

# Create output dirs if they don't exist
for d in [CURATED_DATA_DIR, MODEL_DIR, PLOT_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────
# 2. DATA SETTINGS
# ─────────────────────────────────────────────
# Low-Data Regime (Starves models of training data to prove Quantum Advantage)
LOW_DATA_MODE = False
LOW_DATA_TRAIN_SAMPLES = 20  # Number of training images PER CLASS (40 total)

IMG_SIZE           = 224          # Input image size (224x224 for MobileNetV2)
MIN_FILE_SIZE_KB   = 3            # Skip images smaller than 3 KB (likely corrupted)
MAX_IMAGES_PER_CLASS = 1500       # Cap per class for faster training (~3000 total)
TRAIN_RATIO        = 0.70         # 70% train
VAL_RATIO          = 0.15         # 15% validation
TEST_RATIO         = 0.15         # 15% test
RANDOM_SEED        = 42           # For reproducibility

# ─────────────────────────────────────────────
# 3. AUGMENTATION SETTINGS
# ─────────────────────────────────────────────
ROTATION_RANGE     = 20           # ±20 degrees
HORIZONTAL_FLIP    = True
ZOOM_RANGE         = 0.15         # ±15% zoom
BRIGHTNESS_RANGE   = (0.8, 1.2)   # ±20% brightness
SHEAR_RANGE        = 0.1          # Slight shear

# ─────────────────────────────────────────────
# 4. CLASSICAL MODEL SETTINGS
# ─────────────────────────────────────────────
BACKBONE           = "MobileNetV2"   # Options: MobileNetV2, ResNet50, EfficientNetB0
BATCH_SIZE         = 32
LEARNING_RATE      = 1e-4
FINE_TUNE_LR       = 1e-5            # Lower LR for fine-tuning backbone
EPOCHS_FROZEN      = 10              # Epochs with frozen backbone
EPOCHS_FINETUNE    = 20              # Epochs for fine-tuning
FINE_TUNE_LAYERS   = 30              # Unfreeze last N layers during fine-tune
DROPOUT_RATE       = 0.3
DENSE_UNITS        = 256

# ─────────────────────────────────────────────
# 5. QUANTUM MODEL SETTINGS
# ─────────────────────────────────────────────
N_QUBITS           = 4               # Number of qubits in VQC
N_VQC_LAYERS       = 2               # Depth of variational circuit
PCA_COMPONENTS     = 4               # Must match N_QUBITS for angle encoding
QUANTUM_LR         = 0.01            # Learning rate for quantum parameters
QUANTUM_EPOCHS     = 50              # Training epochs for VQC
QUANTUM_BATCH_SIZE = 16              # Smaller batches for quantum (CPU-bound)

# ─────────────────────────────────────────────
# 6. CLASS LABELS
# ─────────────────────────────────────────────
CLASS_NAMES = ["Not Infected", "Infected"]  # 0 = healthy, 1 = PCOS
