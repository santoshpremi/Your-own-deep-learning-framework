"""Deep Learning Framework Core Module.

This package contains the fundamental building blocks for neural networks:
- Tensor: Core data structure for numerical computations
- Layers: Various neural network layer implementations
- Activations: Activation functions and their derivatives
- Losses: Loss functions for model training
- Network: Neural network composition and training logic
- BatchNorm: Batch normalization layer implementation
- Conv2D: 2D Convolutional layer
- Pooling2D: 2D Pooling layer (Max/Avg)
"""

# Core components
from .tensor import Tensor, Shape
from .layers import InputLayer, FullyConnectedLayer, Pooling2DLayer
from .activations import ReLULayer, SigmoidLayer, SoftmaxLayer
from .losses import MSELossLayer, CrossEntropyLossLayer
from .network import Network
from .batch_norm import BatchNormLayer
from .flatten import FlattenLayer
from .conv_layers import Conv2DLayer
from .dropout import DropoutLayer

__all__ = [
    # Core types
    'Tensor',
    'Shape',
    
    # Layers
    'InputLayer',
    'FullyConnectedLayer',
    'BatchNormLayer',
    'Conv2DLayer',
    'Pooling2DLayer',
    'DropoutLayer',
    
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