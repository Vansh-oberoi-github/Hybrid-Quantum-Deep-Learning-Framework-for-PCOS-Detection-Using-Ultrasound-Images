"""
quantum_circuit.py — Variational Quantum Circuit (VQC) Design
===============================================================
Uses PennyLane to create a parameterized quantum circuit for classification.

Circuit Design:
    4 qubits, 2 variational layers

    |0⟩ ─ RX(x₀) ─ Rot(θ₀,θ₁,θ₂) ─ ●─── ─ Rot(θ₁₂,θ₁₃,θ₁₄) ─ ●─── ─ ⟨Z⟩
    |0⟩ ─ RX(x₁) ─ Rot(θ₃,θ₄,θ₅) ─ X─●─ ─ Rot(θ₁₅,θ₁₆,θ₁₇) ─ X─●─
    |0⟩ ─ RX(x₂) ─ Rot(θ₆,θ₇,θ₈) ─ ──X● ─ Rot(θ₁₈,θ₁₉,θ₂₀) ─ ──X●
    |0⟩ ─ RX(x₃) ─ Rot(θ₉,θ₁₀,θ₁₁)─ ───X ─ Rot(θ₂₁,θ₂₂,θ₂₃) ─ ───X

Why this design:
    - Angle encoding maps classical features to quantum rotations
    - Rot gates (3 params each) provide full single-qubit rotation
    - CNOT ring creates entanglement between all qubits
    - 2 layers give enough expressivity without overfitting
    - 24 total quantum parameters (very lightweight)

Note: PennyLane 0.45 deprecated KerasLayer. We use a custom TF layer.
"""

import pennylane as qml
import numpy as np
import tensorflow as tf

from src.config import N_QUBITS, N_VQC_LAYERS


def create_quantum_device():
    """
    Create a PennyLane quantum device (simulator).
    Uses 'default.qubit' which is a CPU-based exact simulator.
    
    Returns:
        dev: PennyLane device
    """
    dev = qml.device("default.qubit", wires=N_QUBITS)
    print(f"[QUANTUM] Created device with {N_QUBITS} qubits")
    return dev


def angle_encoding(features, wires):
    """
    Encode classical features as rotation angles on qubits.
    Each feature x_i is encoded as RX(x_i) on qubit i.
    
    This is a simple but effective encoding for small feature vectors.
    
    Args:
        features: Array of shape (n_qubits,) — PCA-reduced features
        wires: List of qubit indices
    """
    for i, wire in enumerate(wires):
        qml.RX(features[i], wires=wire)


def variational_layer(weights, wires):
    """
    One layer of the variational circuit:
    1. Apply Rot(φ, θ, ω) to each qubit (full single-qubit rotation)
    2. Apply CNOT ring for entanglement: 0→1→2→3→0
    
    Args:
        weights: Array of shape (n_qubits, 3) — rotation angles
        wires: List of qubit indices
    """
    # Single-qubit rotations
    for i, wire in enumerate(wires):
        qml.Rot(weights[i, 0], weights[i, 1], weights[i, 2], wires=wire)
    
    # CNOT entanglement ring
    for i in range(len(wires)):
        qml.CNOT(wires=[wires[i], wires[(i + 1) % len(wires)]])


def build_vqc_qnode():
    """
    Build the VQC as a PennyLane QNode with TensorFlow interface.
    
    Returns:
        circuit: QNode callable
        dev: The quantum device
    """
    dev = create_quantum_device()
    wires = list(range(N_QUBITS))
    
    @qml.qnode(dev, interface="tf", diff_method="backprop")
    def circuit(inputs, weights):
        """
        Quantum circuit that classifies PCA features.
        
        Args:
            inputs: Classical features, shape (n_qubits,)
            weights: Trainable parameters, shape (n_layers, n_qubits, 3)
            
        Returns:
            Expectation value of PauliZ on qubit 0 (range [-1, 1])
        """
        # Step 1: Encode classical data
        angle_encoding(inputs, wires)
        
        # Step 2: Apply variational layers
        for layer_idx in range(N_VQC_LAYERS):
            variational_layer(weights[layer_idx], wires)
        
        # Step 3: Measure qubit 0
        return qml.expval(qml.PauliZ(0))
    
    total_params = N_VQC_LAYERS * N_QUBITS * 3
    print(f"[VQC] Circuit: {N_QUBITS} qubits, {N_VQC_LAYERS} layers, {total_params} parameters")
    
    return circuit, dev


class QuantumLayer(tf.keras.layers.Layer):
    """
    Custom Keras Layer that wraps the PennyLane VQC.
    
    Replaces the deprecated qml.qnn.KerasLayer.
    Processes each sample through the quantum circuit and returns
    the expectation value as a prediction.
    """
    
    def __init__(self, n_qubits=N_QUBITS, n_layers=N_VQC_LAYERS, **kwargs):
        super().__init__(**kwargs)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Build quantum circuit
        self.circuit, self.dev = build_vqc_qnode()
    
    def build(self, input_shape):
        """Create trainable quantum weights."""
        self.quantum_weights = self.add_weight(
            name="quantum_weights",
            shape=(self.n_layers, self.n_qubits, 3),
            initializer=tf.keras.initializers.RandomUniform(
                minval=0, maxval=2 * np.pi
            ),
            trainable=True,
            dtype=tf.float64
        )
        super().build(input_shape)
    
    def call(self, inputs):
        """
        Forward pass: run each input sample through the quantum circuit.
        
        Args:
            inputs: Tensor of shape (batch_size, n_qubits)
            
        Returns:
            Tensor of shape (batch_size, 1) — quantum predictions
        """
        # Cast inputs to float64 for PennyLane compatibility
        inputs = tf.cast(inputs, tf.float64)
        
        # Process each sample through the quantum circuit
        # Use tf.vectorized_map for efficiency
        results = tf.vectorized_map(
            lambda x: self.circuit(x, self.quantum_weights),
            inputs
        )
        
        # Reshape to (batch_size, 1)
        return tf.cast(tf.reshape(results, (-1, 1)), tf.float32)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
        })
        return config


def visualize_circuit():
    """
    Print a text visualization of the quantum circuit.
    Useful for understanding and debugging the circuit design.
    """
    dev = qml.device("default.qubit", wires=N_QUBITS)
    
    @qml.qnode(dev)
    def sample_circuit(inputs, weights):
        angle_encoding(inputs, list(range(N_QUBITS)))
        for layer_idx in range(N_VQC_LAYERS):
            variational_layer(weights[layer_idx], list(range(N_QUBITS)))
        return qml.expval(qml.PauliZ(0))
    
    # Run with dummy data to generate circuit drawing
    dummy_inputs = np.random.randn(N_QUBITS)
    dummy_weights = np.random.randn(N_VQC_LAYERS, N_QUBITS, 3)
    
    sample_circuit(dummy_inputs, dummy_weights)
    
    print("\n" + "=" * 60)
    print("  QUANTUM CIRCUIT DIAGRAM")
    print("=" * 60)
    print(qml.draw(sample_circuit)(dummy_inputs, dummy_weights))
    print("=" * 60)


if __name__ == "__main__":
    visualize_circuit()
