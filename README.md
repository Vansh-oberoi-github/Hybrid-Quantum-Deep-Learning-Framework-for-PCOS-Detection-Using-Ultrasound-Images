# Hybrid Quantum Deep Learning Framework for PCOS Detection 

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?logo=tensorflow&logoColor=white)
![PennyLane](https://img.shields.io/badge/PennyLane-Quantum-000000?logo=qiskit)

A groundbreaking proof-of-concept that integrates **Variational Quantum Circuits (VQC)** with **Classical Convolutional Neural Networks (CNN)** to detect Polycystic Ovary Syndrome (PCOS) from ultrasound imagery. 

This repository provides empirical proof of the **Quantum Low-Data Advantage**, demonstrating that a 24-parameter quantum circuit can outperform a 1.8-million parameter classical network when training data is severely limited, all while maintaining rigorous, zero-leakage data purity.

---

## 🧬 Architecture

The framework utilizes a highly optimized hybrid pipeline:
1. **Classical Feature Extraction**: A pre-trained `MobileNetV2` backbone (with the top classification head removed) acts as the feature extractor, condensing raw ultrasound images into a dense 1280-dimensional feature space.
2. **Dimensionality Reduction**: Principal Component Analysis (PCA) aggressively reduces the feature space from 1280 to 4 dimensions.
3. **Quantum Angle Encoding**: The 4-dimensional classical vectors are mathematically bounded to `[0, π]` using strict training-set limits to prevent statistical data leakage, then encoded into a quantum state via Pauli-Y rotations.
4. **Variational Quantum Circuit (VQC)**: A customized `PennyLane` circuit comprising highly entangled parameterized rotation gates evaluates the quantum state to produce a binary classification (Infected vs. Healthy).

---

## 🚀 The Low-Data Quantum Advantage

In modern medical machine learning, acquiring massive, annotated datasets is extremely expensive. This project was specifically benchmarked in a **Low-Data Regime** to test the expressivity of the quantum layer.

When the training algorithm was artificially starved of data (restricted to merely **40 training images**):
- **Classical CNN Head (~1.8 Million Params)**: Severely overfit the data and degraded to **87.6%** accuracy on the massive 230-image unseen test set.
- **Hybrid Quantum VQC (24 Params)**: Mathematically mapped the features efficiently within its exponential Hilbert space, achieving **97.0%** accuracy, alongside superior Recall, F1-Score, and ROC-AUC.

This proves that Variational Quantum Circuits offer a massive paradigm shift in parameter efficiency and low-data generalizability.

### Parameter Efficiency Comparison
| Architecture | Trainable Classification Parameters | Accuracy (Low-Data) |
| --- | --- | --- |
| Classical CNN | ~1,854,000 | 87.61% |
| Hybrid Quantum | **24** | **97.01%** |

---

## 🛡️ Zero-Leakage Data Methodology

Many public Kaggle datasets suffer from systemic flaws such as identical train/test duplicate folders or frame-level patient leakage. This pipeline was rigorously sanitized to guarantee absolute mathematical purity:
- **Cryptographic Deduplication**: MD5 hashing is utilized to detect and destroy identical duplicate images across the dataset.
- **Strict Set Isolation**: Min/Max scaling for quantum angle encoding is calculated *exclusively* on the training set. The validation and test sets are scaled using these fixed bounds (with hard `[0, π]` clipping) to prevent statistical peeking.

---

## 🧠 Interpretability (Score-CAM & Grad-CAM)

To prove the models are making decisions based on true morphological features (ovarian follicles) rather than arbitrary noise or textual artifacts, this framework includes advanced interpretability maps. 
- **Score-CAM**: A purely gradient-free approach that upsamples CNN feature maps to mask the original image, providing incredibly clean, highly localized heatmaps of cystic formations.

---

## ⚙️ Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/Vansh-oberoi-github/Hybrid-Quantum-Deep-Learning-Framework-for-PCOS-Detection-Using-Ultrasound-Images.git
cd Hybrid-Quantum-Deep-Learning-Framework-for-PCOS-Detection-Using-Ultrasound-Images
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Full Experiment
The main orchestration script handles everything from data curation to model training and evaluation:
```bash
python run_experiment.py --mode all
```

*Individual modes available:*
- `--mode curate`: Rebuilds the dataset splits.
- `--mode classical`: Trains the baseline CNN.
- `--mode hybrid`: Trains the Quantum VQC.
- `--mode evaluate`: Generates comparative plots.

---

### Acknowledgments
This repository serves as a foundational proof-of-concept for the application of Quantum Machine Learning (QML) in the NISQ era for medical imaging tasks. 
