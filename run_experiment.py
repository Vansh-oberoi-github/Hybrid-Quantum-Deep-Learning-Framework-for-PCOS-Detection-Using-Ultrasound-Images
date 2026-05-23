"""
run_experiment.py — Main Entry Point
=======================================
Runs the full Hybrid Quantum Deep Learning pipeline for PCOS Detection.

Usage:
    python run_experiment.py                  # Full pipeline
    python run_experiment.py --mode curate    # Only curate data
    python run_experiment.py --mode classical # Train classical only
    python run_experiment.py --mode hybrid    # Train hybrid only  
    python run_experiment.py --mode evaluate  # Evaluate saved models
    python run_experiment.py --mode gradcam   # Generate Grad-CAM only

Author: Hybrid Quantum PCOS Detection Project
"""

import os
import sys
import argparse
import json
import time

# Fix Windows console encoding for special characters
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress verbose TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from src.utils import set_seed, check_gpu, print_banner
from src.config import MODEL_DIR, RESULTS_DIR, PLOT_DIR, CURATED_DATA_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hybrid Quantum Deep Learning for PCOS Detection"
    )
    parser.add_argument(
        '--mode', type=str, default='all',
        choices=['all', 'curate', 'classical', 'hybrid', 'evaluate', 'gradcam'],
        help='Which part of the pipeline to run'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()
    
    # -- Setup --
    print_banner("HYBRID QUANTUM DEEP LEARNING FOR PCOS DETECTION", "=")
    set_seed()
    has_gpu = check_gpu()
    
    # =============================================
    # STEP 1: DATA CURATION
    # =============================================
    if args.mode in ['all', 'curate']:
        # Check if curation already done
        train_dir = os.path.join(CURATED_DATA_DIR, "train")
        if os.path.exists(train_dir) and len(os.listdir(os.path.join(train_dir, "infected"))) > 0:
            print("\n[INFO] Curated data already exists. Skipping curation.")
            print(f"       Delete '{CURATED_DATA_DIR}' to re-curate.")
        else:
            print_banner("STEP 1: DATA CURATION")
            from src.data_pipeline import curate_dataset
            curate_dataset()
        
        if args.mode == 'curate':
            print("\n[DONE] Data curation complete. Run again with --mode all to continue.")
            return
    
    # =============================================
    # STEP 2: CREATE DATA GENERATORS
    # =============================================
    if args.mode in ['all', 'classical', 'hybrid', 'evaluate', 'gradcam']:
        print_banner("STEP 2: LOADING DATA")
        from src.data_pipeline import create_data_generators
        train_gen, val_gen, test_gen = create_data_generators()
    
    # =============================================
    # STEP 3: TRAIN CLASSICAL MODEL
    # =============================================
    classical_model = None
    classical_history = None
    
    if args.mode in ['all', 'classical']:
        print_banner("STEP 3: CLASSICAL CNN TRAINING")
        from src.train import train_classical_model
        classical_model, base_model, classical_history = train_classical_model(train_gen, val_gen)
        
        # Plot training history
        from src.evaluate import plot_training_history
        plot_training_history(classical_history, "Classical CNN")
    
    # Load saved classical model if not just trained
    if classical_model is None and args.mode in ['hybrid', 'evaluate', 'gradcam']:
        print("\n  Loading saved classical model...")
        import tensorflow as tf
        model_path = os.path.join(MODEL_DIR, "classical_final.keras")
        if os.path.exists(model_path):
            classical_model = tf.keras.models.load_model(model_path)
            print(f"  Loaded: {model_path}")
        else:
            print(f"  ERROR: No saved classical model found at {model_path}")
            print("  Run with --mode classical first")
            return
    
    # =============================================
    # STEP 4: TRAIN HYBRID QUANTUM MODEL
    # =============================================
    hybrid_model = None
    pca_data = None
    
    if args.mode in ['all', 'hybrid']:
        print_banner("STEP 4: HYBRID QUANTUM MODEL TRAINING")
        from src.train import train_hybrid_model
        hybrid_model, hybrid_history, pca_data = train_hybrid_model(
            classical_model, train_gen, val_gen, test_gen
        )
        
        # Plot training history
        from src.evaluate import plot_training_history
        plot_training_history(hybrid_history, "Hybrid Quantum")
    
    # Load saved hybrid model if not just trained
    if hybrid_model is None and args.mode in ['evaluate']:
        print("\n  Loading saved hybrid model...")
        import tensorflow as tf
        hybrid_path = os.path.join(MODEL_DIR, "hybrid_final.keras")
        if os.path.exists(hybrid_path):
            try:
                from src.quantum_circuit import QuantumLayer
                tf.keras.config.enable_unsafe_deserialization()
                hybrid_model = tf.keras.models.load_model(
                    hybrid_path,
                    custom_objects={"QuantumLayer": QuantumLayer}
                )
                print(f"  Loaded: {hybrid_path}")
            except Exception as e:
                print(f"  ERROR: Failed to load hybrid model: {e}")
                return
        else:
            print(f"  ERROR: No saved hybrid model found")
            print("  Run with --mode hybrid first")
            return
    
    # =============================================
    # STEP 5: EVALUATION & COMPARISON
    # =============================================
    if args.mode in ['all', 'evaluate']:
        print_banner("STEP 5: EVALUATION & COMPARISON")
        from src.evaluate import evaluate_and_compare
        
        if pca_data is None:
            # Re-extract features for evaluation
            from src.hybrid_model import (
                extract_features, load_pca_scaler, transform_features
            )
            from src.classical_model import build_feature_extractor
            
            feature_model = build_feature_extractor(classical_model)
            test_features, test_labels = extract_features(feature_model, test_gen)
            scaler, pca, min_vals, max_vals = load_pca_scaler()
            test_quantum = transform_features(test_features, scaler, pca, min_vals, max_vals)
            
            pca_data = {"test_features": test_quantum, "test_labels": test_labels}
        
        classical_metrics, hybrid_metrics = evaluate_and_compare(
            classical_model, hybrid_model, test_gen, pca_data
        )
    
    # =============================================
    # STEP 6: GRAD-CAM EXPLAINABILITY
    # =============================================
    if args.mode in ['all', 'gradcam']:
        print_banner("STEP 6: GRAD-CAM EXPLAINABILITY")
        from src.gradcam import generate_gradcam_grid
        generate_gradcam_grid(classical_model, test_gen, n_samples=8)
    
    # =============================================
    # DONE!
    # =============================================
    elapsed = time.time() - start_time
    print_banner("EXPERIMENT COMPLETE!", "=")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Models saved to:  {MODEL_DIR}")
    print(f"  Plots saved to:   {PLOT_DIR}")
    print(f"  Results saved to: {RESULTS_DIR}")
    print(f"\n  Check the 'outputs/' folder for all generated files")
    print("=" * 60)


if __name__ == "__main__":
    main()
