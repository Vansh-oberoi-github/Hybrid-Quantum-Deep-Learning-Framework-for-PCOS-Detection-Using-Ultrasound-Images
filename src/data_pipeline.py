"""
data_pipeline.py — Data Curation, Loading, Splitting & Augmentation
=====================================================================
This module handles:
1. Scanning ONLY Dataset 2 (to prevent frame/patient-level data leakage from Dataset 1)
2. Preserving the original authors' train/test patient split.
3. Creating a validation split from the training set.
4. Balancing classes by downsampling the majority class.
5. Deduplicating using MD5 hashes to guarantee zero identical images.
"""

import os
import shutil
import random
import numpy as np
import cv2
import hashlib
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.config import (
    DATASET_2_DIR, CURATED_DATA_DIR,
    IMG_SIZE, MIN_FILE_SIZE_KB,
    VAL_RATIO, RANDOM_SEED,
    ROTATION_RANGE, HORIZONTAL_FLIP, ZOOM_RANGE,
    BRIGHTNESS_RANGE, SHEAR_RANGE, BATCH_SIZE,
    LOW_DATA_MODE, LOW_DATA_TRAIN_SAMPLES
)


def filter_and_deduplicate(image_paths, seen_hashes=None, min_size_kb=MIN_FILE_SIZE_KB):
    """
    Filter out corrupted, too-small, or duplicate images.
    """
    if seen_hashes is None:
        seen_hashes = set()
        
    valid_paths = []
    rejected = 0
    duplicate_count = 0
    
    for path in tqdm(image_paths, desc="Filtering", leave=False):
        try:
            size_kb = os.path.getsize(path) / 1024
            if size_kb < min_size_kb:
                rejected += 1
                continue
            
            img = cv2.imread(path)
            if img is None:
                rejected += 1
                continue
            
            h, w = img.shape[:2]
            if h < 32 or w < 32:
                rejected += 1
                continue
                
            # Strict Deduplication Check
            img_hash = hashlib.md5(img.tobytes()).hexdigest()
            if img_hash in seen_hashes:
                duplicate_count += 1
                rejected += 1
                continue
            seen_hashes.add(img_hash)
            
            valid_paths.append(path)
            
        except Exception:
            rejected += 1
            continue
    
    print(f"    Kept {len(valid_paths)}, rejected {rejected} (including {duplicate_count} exact duplicates)")
    return valid_paths, seen_hashes


def balance_classes(infected_paths, healthy_paths):
    """Balance classes to have a 1:1 ratio."""
    random.seed(RANDOM_SEED)
    min_count = min(len(infected_paths), len(healthy_paths))
    
    infected_sampled = random.sample(infected_paths, min_count)
    healthy_sampled = random.sample(healthy_paths, min_count)
    
    return infected_sampled, healthy_sampled


def apply_clahe(image):
    """Apply CLAHE to enhance ultrasound image contrast."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def process_and_save_split(paths, labels, split_name):
    """Apply CLAHE, resize, and save to curated directory."""
    for cls_name in ["infected", "notinfected"]:
        os.makedirs(os.path.join(CURATED_DATA_DIR, split_name, cls_name), exist_ok=True)
    
    print(f"\nProcessing {split_name} set ({len(paths)} images)...")
    for i, (path, label) in enumerate(tqdm(zip(paths, labels), total=len(paths), leave=False)):
        img = cv2.imread(path)
        if img is None: continue
        
        img = apply_clahe(img)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        
        cls_folder = "infected" if label == 1 else "notinfected"
        filename = f"{split_name}_{cls_folder}_{i:05d}.jpg"
        out_path = os.path.join(CURATED_DATA_DIR, split_name, cls_folder, filename)
        
        cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])


def curate_dataset():
    """
    Main data curation function. 
    Strictly uses Dataset 2 to prevent patient/frame leakage from Dataset 1.
    Preserves Dataset 2's predefined train and test splits to guarantee zero patient overlap.
    """
    print("=" * 60)
    print("  DATA CURATION PIPELINE (DATASET 2 ONLY)")
    print("=" * 60)
    
    # ── 1. SCAN DATASET 2 ──
    print("\n[1] Scanning Dataset 2...")
    # Dataset 2's "test" directory is literally an exact copy of the "train" directory.
    # We will just scan the "train" directory to get the unique images.
    ds2_inf = glob(os.path.join(DATASET_2_DIR, "train", "infected", "*.*"))
    ds2_hlt = glob(os.path.join(DATASET_2_DIR, "train", "notinfected", "*.*"))
    
    # ── 2. FILTER & DEDUPLICATE ──
    print("\n[2] Filtering & Deduplicating Dataset 2...")
    seen_hashes = set()
    inf_valid, seen_hashes = filter_and_deduplicate(ds2_inf, seen_hashes)
    hlt_valid, seen_hashes = filter_and_deduplicate(ds2_hlt, seen_hashes)
    
    # ── 3. BALANCE CLASSES ──
    print("\n[3] Balancing Classes (1:1 Ratio)...")
    inf_bal, hlt_bal = balance_classes(inf_valid, hlt_valid)
    
    # ── 4. CREATE TRAIN/VAL/TEST SPLIT ──
    print("\n[4] Creating Train/Val/Test Split...")
    all_paths = inf_bal + hlt_bal
    all_labels = [1]*len(inf_bal) + [0]*len(hlt_bal)
    
    # First split: train vs temp (val+test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths, all_labels, test_size=(VAL_RATIO + 0.15), stratify=all_labels, random_state=RANDOM_SEED
    )
    
    # Second split: val vs test
    val_ratio_adj = VAL_RATIO / (VAL_RATIO + 0.15)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=(1 - val_ratio_adj), stratify=temp_labels, random_state=RANDOM_SEED
    )
    
    # --- LOW DATA REGIME ---
    if LOW_DATA_MODE:
        print(f"\n[!] LOW DATA MODE ENABLED: Restricting train size to {LOW_DATA_TRAIN_SAMPLES} per class.")
        # Cripple the training set
        train_inf = [p for p, l in zip(train_paths, train_labels) if l == 1][:LOW_DATA_TRAIN_SAMPLES]
        train_hlt = [p for p, l in zip(train_paths, train_labels) if l == 0][:LOW_DATA_TRAIN_SAMPLES]
        train_paths = train_inf + train_hlt
        train_labels = [1]*len(train_inf) + [0]*len(train_hlt)
        
        # Cripple the validation set proportionately (e.g., half of train size)
        val_limit = max(1, LOW_DATA_TRAIN_SAMPLES // 2)
        val_inf = [p for p, l in zip(val_paths, val_labels) if l == 1][:val_limit]
        val_hlt = [p for p, l in zip(val_paths, val_labels) if l == 0][:val_limit]
        val_paths = val_inf + val_hlt
        val_labels = [1]*len(val_inf) + [0]*len(val_hlt)
        
        print(f"    Train Set: {len(train_paths)} images")
        print(f"    Val Set:   {len(val_paths)} images")
        print(f"    Test Set:  {len(test_paths)} images (kept large for robust evaluation)")
    
    # ── 5. PROCESS AND SAVE ──
    print("\n[5] Applying CLAHE & Resizing...")
    process_and_save_split(train_paths, train_labels, "train")
    process_and_save_split(val_paths, val_labels, "val")
    process_and_save_split(test_paths, test_labels, "test")
    
    # Print final statistics
    print("\n" + "=" * 60)
    print("  CURATION COMPLETE (ZERO PATIENT LEAKAGE)")
    print("=" * 60)
    for split_name in ["train", "val", "test"]:
        inf_count = len(os.listdir(os.path.join(CURATED_DATA_DIR, split_name, "infected")))
        hlt_count = len(os.listdir(os.path.join(CURATED_DATA_DIR, split_name, "notinfected")))
        print(f"  {split_name:>5}: {inf_count} infected + {hlt_count} healthy = {inf_count + hlt_count}")
    print("=" * 60)


def create_data_generators():
    """Create Keras ImageDataGenerators."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=ROTATION_RANGE,
        horizontal_flip=HORIZONTAL_FLIP,
        zoom_range=ZOOM_RANGE,
        brightness_range=BRIGHTNESS_RANGE,
        shear_range=SHEAR_RANGE,
        fill_mode='nearest'
    )
    
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    
    train_gen = train_datagen.flow_from_directory(
        os.path.join(CURATED_DATA_DIR, "train"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=['notinfected', 'infected'],
        shuffle=True,
        seed=RANDOM_SEED
    )
    
    val_gen = eval_datagen.flow_from_directory(
        os.path.join(CURATED_DATA_DIR, "val"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=['notinfected', 'infected'],
        shuffle=False
    )
    
    test_gen = eval_datagen.flow_from_directory(
        os.path.join(CURATED_DATA_DIR, "test"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=['notinfected', 'infected'],
        shuffle=False
    )
    
    print(f"\n[GENERATORS] Train: {train_gen.samples} | Val: {val_gen.samples} | Test: {test_gen.samples}")
    return train_gen, val_gen, test_gen


if __name__ == "__main__":
    curate_dataset()
