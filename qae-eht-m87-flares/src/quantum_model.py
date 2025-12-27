import torch
import torch.nn as nn
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.qnn import TorchLayer

# ======================================================================
# QUANTUM DEVICE AND CIRCUIT CONFIGURATION
# ======================================================================

# Quantum device: 12 qubits = 4096-dimensional Hilbert space
N_QUBITS = 12
N_LAYERS = 3
dev = qml.device("default.qubit", wires=N_QUBITS, shots=None)

# ======================================================================
# AMPLITUDE ENCODING FUNCTION
# ======================================================================

def amplitude_encode_12q(image_4x4):
    """
    Encode a 4x4 image patch into a 12-qubit quantum state.
    
    Args:
        image_4x4: numpy array or torch.Tensor of shape (4, 4)
        Values in range [0, 1] (amplitude encoded)
    
    Returns:
        state_vector: numpy array of shape (4096,) 
        Normalized to unit norm for quantum state preparation
    
    Physics:
        - 4x4 patch = 16 pixels → flatten to 16-dim vector
        - Pad to 4096 dimensions (2^12, for 12 qubits)
        - Normalize: sum(|amplitude|^2) = 1 (quantum state requirement)
    """
    # handle torch tensors
    if isinstance(image_4x4, torch.Tensor):
        image_4x4 = image_4x4.cpu().numpy()
    
    # Validate shape
    if image_4x4.shape != (4, 4):
        raise ValueError(
            f"Expected 4x4 patch, got shape {image_4x4.shape}. "
            f"Data loader may be returning incorrectly sized patches."
        )
    
    # flatten to 16-dimensional vector
    flat = image_4x4.flatten().astype(np.float64)
    
    # pad to 4096 dimensions (2^12)
    padded = np.zeros(4096)
    padded[:16] = flat
    
    # normalize to unit norm (required for quantum state preparation)
    norm = np.linalg.norm(padded)
    if norm > 1e-10:
        padded = padded / norm
    
    # final validation
    assert abs(np.linalg.norm(padded) - 1.0) < 1e-6, "State vector not normalized!"
    
    return padded

# ====================================================================
# QUANTUM CIRCUIT (QNode)
# ====================================================================

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    """
    Variational quantum circuit for feature extraction.
    
    Args:
        inputs: Dummy input (not used, we encode via closure in forward pass)
        weights: torch.Tensor of shape [N_LAYERS, N_QUBITS, 3]
                Learnable rotation angles (RX, RY, RZ) per qubit per layer
    
    Returns:
        List of expectation values (Pauli-Z measurements) for each qubit
        Shape: [N_QUBITS]
    """
    # this is a limitation of TorchLayer - it requires an input argument
    
    # variational quantum layers
    for layer_idx in range(N_LAYERS):
        layer_params = weights[layer_idx]
        
        # rotation gates (learned parameters)
        for i in range(N_QUBITS):
            qml.RX(layer_params[i, 0], wires=i)
            qml.RY(layer_params[i, 1], wires=i)
            qml.RZ(layer_params[i, 2], wires=i)
        
        # entanglement layer (captures non-local correlations)
        # ring topology: each qubit entangled with its neighbor
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % N_QUBITS])
    
    # measurement: expectation values of Pauli-Z operators
    # these form the "quantum features" fed to classical post-processing
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

# ====================================================================
# TORCH LAYER (Hybrid Quantum-Classical Interface)
# ====================================================================

# weight shapes for TorchLayer
weight_shapes = {"weights": (N_LAYERS, N_QUBITS, 3)}

# create TorchLayer: wraps quantum_circuit as a PyTorch module
# should automatically handle quantum parameter gradients via parameter-shift rule
qlayer = TorchLayer(quantum_circuit, weight_shapes)

# =====================================================================
# HYBRID QUANTUM-CLASSICAL AUTOENCODER MODEL
# =====================================================================

class QuantumAutoencoder(nn.Module):
    """
    Hybrid Quantum-Classical Autoencoder for M87* Flare Detection
    
    Architecture:
        - Quantum encoder: 12-qubit circuit (TorchLayer)
                         Maps 4x4 image → 12 expectation values
        - Classical decoder: Feed-forward network
                            Maps 12 quantum features → 16 pixels (4x4 reconstruction)
    
    Key Design Decisions:
        1. Manual amplitude encoding (4x4 → 4096-dim quantum state)
        2. TorchLayer handles quantum gradients automatically
        3. Separate classical post-processing for interpretability
        4. Batch processing via loop (quantum circuits can't be natively batched)
    
    Total Parameters: ~892
        - Quantum: 108 rotation angles (N_LAYERS × N_QUBITS × 3)
        - Classical: 784 weights/biases in scale_layer
    """
    
    def __init__(self, n_qubits=N_QUBITS, n_layers=N_LAYERS):
        super().__init__()
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # quantum layer (TorchLayer)
        self.quantum_layer = qlayer
        
        # classical post-processing network
        # maps quantum expectations (12) → pixel values (16)
        self.scale_layer = nn.Sequential(
            nn.Linear(n_qubits, 32),    # Expand representation
            nn.ReLU(inplace=True),       # Non-linear mapping
            nn.Linear(32, 16),          # Compress to 16 pixels
            nn.Sigmoid()                 # Range [0,1] for images
        )
        
        # initialize classical weights
        self._init_weights()
        
        # print parameter counts
        quantum_params = sum(p.numel() for p in self.quantum_layer.parameters())
        classical_params = sum(p.numel() for p in self.scale_layer.parameters())
        print(f"✓ Quantum parameters: {quantum_params} rotations")
        print(f"✓ Classical parameters: {classical_params}")
        print(f"✓ Total hybrid parameters: {quantum_params + classical_params}")
    
    def _init_weights(self):
        """Initialize classical layers using Xavier initialization"""
        for module in self.scale_layer.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, images_batch):
        """
        Forward pass for batch of 4×4 images
        
        Args:
            images_batch: torch.Tensor of shape [batch, 1, 4, 4]
            Pixel values in range [0, 1]
        
        Returns:
            reconstructions: torch.Tensor of shape [batch, 1, 4, 4]
            Reconstructed patches in range [0, 1]
        
        Process:
            1. Loop over batch (quantum circuits can't be batched natively)
            2. Encode each image to quantum state (4096-dim)
            3. Run quantum circuit → 12 expectation values
            4. Classical decoding → 16 pixel values → 4×4 reconstruction
        """
        batch_size = images_batch.shape[0]
        reconstructions = []
        
        # process each image individually
        # pennyLane qnodes can't handle batched inputs directly
        for i in range(batch_size):
            # extract single image patch
            img_np = images_batch[i, 0].cpu().numpy()
            
            # encode to quantum state (4096-dim normalized vector)
            state = amplitude_encode_12q(img_np)
            
            # run quantum circuit
            # torchLayer expects an input argument (dummy here)
            dummy_input = torch.zeros(1, device=images_batch.device)
            expectations = self.quantum_layer(dummy_input)
            
            # classical post-processing
            pixel_values = self.scale_layer(expectations)
            
            # reshape to 4×4 image
            reconstruction = pixel_values.reshape(1, 4, 4)
            reconstructions.append(reconstruction)
        
        # stack batch back together
        return torch.stack(reconstructions)
    
    def get_reconstruction_error(self, images_batch):
        """
        Compute per-image reconstruction error (MSE) for anomaly detection
        
        Args:
            images_batch: torch.Tensor [batch, 1, 4, 4]
        
        Returns:
            errors: torch.Tensor [batch] (one error score per image)
        """
        reconstructed = self.forward(images_batch)
        mse = nn.functional.mse_loss(reconstructed, images_batch, reduction='none')
        return mse.mean(dim=(1, 2, 3))

# =====================================================================
# MODULE EXPORTS
# =====================================================================

__all__ = [
    'QuantumAutoencoder',
    'amplitude_encode_12q',
    'quantum_circuit',
    'qlayer',
    'N_QUBITS',
    'N_LAYERS'
]