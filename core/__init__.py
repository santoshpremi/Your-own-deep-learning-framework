"""Deep Learning Framework Core Module.

This package contains the fundamental building blocks for neural networks:
- Tensor: Core data structure for numerical computations
- Layers: Various neural network layer implementations
- Activations: Activation functions and their derivatives
- Losses: Loss functions for model training
- Network: Neural network composition and training logic
- BatchNorm: Batch normalization layer implementation
"""

# Core components
from .tensor import Tensor, Shape
from .layers import InputLayer, FullyConnectedLayer
from .activations import ReLULayer, SigmoidLayer, SoftmaxLayer
from .losses import MSELossLayer, CrossEntropyLossLayer
from .network import Network
from .batch_norm import BatchNormLayer

__all__ = [
    # Core types
    'Tensor',
    'Shape',
    
    # Layers
    'InputLayer',
    'FullyConnectedLayer',
    'BatchNormLayer',
    
    # Activations
    'ReLULayer',
    'SigmoidLayer',
    'SoftmaxLayer',
    
    # Losses
    'MSELossLayer',
    'CrossEntropyLossLayer',
    
    # Network
    'Network',
]