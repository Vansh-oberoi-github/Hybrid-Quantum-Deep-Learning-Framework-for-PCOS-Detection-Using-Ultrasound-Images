"""
gradcam.py — Grad-CAM Explainability for PCOS Detection
==========================================================
Generates Gradient-weighted Class Activation Mapping (Grad-CAM)
heatmaps to visualize which regions of ultrasound images the
classical CNN model focuses on for PCOS detection.

This is critical for:
    - Clinical interpretability
    - Research validation (does the model look at follicles/cysts?)
    - Publication-quality explainability figures
"""

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import tensorflow as tf

from src.config import IMG_SIZE, PLOT_DIR, CLASS_NAMES


def get_gradcam_heatmap(model, img_array):
    """
    Compute Grad-CAM heatmap for a given image.
    
    Uses a manual forward pass approach compatible with Keras 3:
    1. Pass input through the MobileNetV2 base model
    2. Watch the conv output with GradientTape
    3. Continue through remaining layers manually
    4. Compute gradients and weighted activation maps
    
    Args:
        model: Keras model (our classical CNN)
        img_array: Preprocessed image array, shape (1, 224, 224, 3)
        
    Returns:
        heatmap: Numpy array of shape (7, 7), values in [0, 1]
    """
    # Get base model (MobileNetV2) and classifier layers
    base_model = model.layers[1]  # MobileNetV2 Functional model
    
    with tf.GradientTape() as tape:
        # Forward pass through base model to get conv output
        base_output = base_model(img_array, training=False)  # (1, 7, 7, 1280)
        tape.watch(base_output)
        
        # Continue through remaining layers manually
        x = model.get_layer('global_avg_pool')(base_output)
        x = model.get_layer('dense_256')(x)
        x = model.get_layer('dropout')(x, training=False)
        # Fix: Compute logits to prevent vanishing gradients from sigmoid activation
        output_layer = model.get_layer('output')
        w, b = output_layer.weights
        logits = tf.matmul(x, w) + b
        
        loss = logits[:, 0]
    
    # Compute gradients of output w.r.t. conv features
    grads = tape.gradient(loss, base_output)
    
    if grads is None:
        raise ValueError("Gradients are None")
    
    # Global average pooling of gradients -> channel importance
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weighted sum of feature maps
    heatmap = base_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # ReLU and normalize
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    
    return heatmap.numpy()


def get_scorecam_heatmap(model, img_array, max_N=32):
    """
    Compute Score-CAM heatmap for a given image.
    Score-CAM is gradient-free, providing robust localization by 
    masking the input image with upsampled activation maps.
    
    Args:
        model: Keras model (our classical CNN)
        img_array: Preprocessed image array, shape (1, 224, 224, 3)
        max_N: Max channels to use (sorted by variance) to save computation
        
    Returns:
        heatmap: Numpy array of shape (7, 7), values in [0, 1]
    """
    base_model = model.layers[1]
    
    # 1. Get activations
    activations = base_model(img_array, training=False)[0] # (7, 7, 1280)
    
    # Select top max_N channels based on variance to speed up inference
    if max_N > 0 and activations.shape[-1] > max_N:
        variances = tf.math.reduce_variance(activations, axis=(0, 1))
        top_k_indices = tf.argsort(variances, direction='DESCENDING')[:max_N]
        activations = tf.gather(activations, top_k_indices, axis=-1)
    else:
        max_N = activations.shape[-1]
    
    # 2. Upsample activations to input image size
    input_shape = (img_array.shape[1], img_array.shape[2])
    
    upsampled_activations = tf.image.resize(
        tf.expand_dims(activations, 0), 
        input_shape,
        method='bilinear'
    )[0] # (224, 224, max_N)
    
    # 3. Normalize each upsampled activation map to [0, 1]
    max_vals = tf.reduce_max(upsampled_activations, axis=(0, 1), keepdims=True)
    min_vals = tf.reduce_min(upsampled_activations, axis=(0, 1), keepdims=True)
    norm_activations = (upsampled_activations - min_vals) / (max_vals - min_vals + 1e-8)
    
    # 4. Project onto original image (create masked inputs)
    norm_activations_T = tf.transpose(norm_activations, perm=[2, 0, 1])
    norm_activations_T = tf.expand_dims(norm_activations_T, -1) # (max_N, 224, 224, 1)
    
    masked_images = img_array[0] * norm_activations_T # (max_N, 224, 224, 3)
    
    # 5. Get model confidence scores for each masked image
    scores = []
    batch_size = 16
    for i in range(0, max_N, batch_size):
        batch = masked_images[i:i+batch_size]
        preds = model.predict(batch, verbose=0)
        scores.append(preds[:, 0])
        
    scores = tf.concat(scores, axis=0) # (max_N,)
    
    # 6. Linear combination of original spatial activations with scores
    heatmap = tf.reduce_sum(activations * tf.reshape(scores, (1, 1, max_N)), axis=-1)
    
    # 7. ReLU and normalize
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    
    return heatmap.numpy()


def overlay_heatmap(original_img, heatmap, alpha=0.4):
    """
    Overlay Grad-CAM heatmap on the original image.
    
    Args:
        original_img: Original image array (H, W, 3), range [0, 255]
        heatmap: Grad-CAM heatmap (H_small, W_small), range [0, 1]
        alpha: Transparency of the heatmap overlay
        
    Returns:
        superimposed: Image with heatmap overlay
    """
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed = np.uint8(heatmap_colored * alpha + original_img * (1 - alpha))
    return superimposed


def generate_gradcam_grid(model, test_gen, n_samples=8):
    """
    Generate a grid of Explainability visualizations for sample test images.
    Shows: Original | Grad-CAM (Fixed) | Score-CAM Heatmap | Score-CAM Overlay
    """
    print("\n  Generating Explainability visualizations (Grad-CAM & Score-CAM)...")
    
    test_gen.reset()
    images, labels = next(test_gen)
    
    n = min(n_samples, len(images))
    images = images[:n]
    labels = labels[:n]
    
    predictions = model.predict(images, verbose=0).flatten()
    
    fig, axes = plt.subplots(n, 4, figsize=(18, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    
    column_titles = ['Original Image', 'Grad-CAM (Fixed)', 'Score-CAM Heatmap', 'Score-CAM Overlay']
    for ax, title in zip(axes[0], column_titles):
        ax.set_title(title, fontweight='bold', fontsize=13, pad=10)
    
    success_count = 0
    for i in range(n):
        img = images[i]
        img_array = np.expand_dims(img, axis=0)
        original = np.uint8(img * 255)
        
        true_label = CLASS_NAMES[int(labels[i])]
        pred_label = CLASS_NAMES[int(predictions[i] >= 0.5)]
        confidence = predictions[i] if predictions[i] >= 0.5 else 1 - predictions[i]
        color = 'green' if true_label == pred_label else 'red'
        
        axes[i, 0].imshow(original)
        axes[i, 0].set_ylabel(f'True: {true_label}\nPred: {pred_label}\n({confidence:.1%})',
                             fontsize=10, color=color, fontweight='bold')
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        
        try:
            # 1. Grad-CAM (Fixed logits mapping)
            grad_heatmap = get_gradcam_heatmap(model, img_array)
            grad_overlay = overlay_heatmap(original, grad_heatmap)
            
            axes[i, 1].imshow(grad_overlay)
            axes[i, 1].axis('off')
            
            # 2. Score-CAM (Advanced Visualization)
            score_heatmap = get_scorecam_heatmap(model, img_array)
            score_heatmap_viz = cv2.resize(score_heatmap, (IMG_SIZE, IMG_SIZE))
            score_overlay = overlay_heatmap(original, score_heatmap)
            
            axes[i, 2].imshow(score_heatmap_viz, cmap='jet')
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(score_overlay)
            axes[i, 3].axis('off')
            
            success_count += 1
        except Exception as e:
            print(f"  Warning: Explainers failed for image {i}: {e}")
            for j in range(1, 4):
                axes[i, j].axis('off')
            continue
            
    plt.suptitle('Explainability - Grad-CAM vs Score-CAM (PCOS Detection)',
                fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "explainability_grid.png"),
                bbox_inches='tight', pad_inches=0.3)
    plt.close()
    
    print(f"  Explainability grid saved ({success_count}/{n} images) to {PLOT_DIR}/explainability_grid.png")


def generate_single_gradcam(model, image_path):
    """Generate Grad-CAM for a single image."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_array = np.expand_dims(img_resized / 255.0, axis=0).astype(np.float32)
    
    prediction = model.predict(img_array, verbose=0)[0, 0]
    heatmap = get_gradcam_heatmap(model, img_array)
    overlay = overlay_heatmap(img_resized, heatmap)
    
    return overlay, prediction
