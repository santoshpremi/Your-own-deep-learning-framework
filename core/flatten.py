"""
Flatten layer implementation for the deep learning framework.
"""
import numpy as np
from .tensor import Tensor, Shape
from .layers import Layer

class FlattenLayer(Layer):
    """
    Flattens the input tensor while preserving the batch dimension.
    
    Input shape: (batch_size, channels, height, width)
    Output shape: (batch_size, channels * height * width)
    """
    def __init__(self):
        super().__init__()
        self.layer_type = "FlattenLayer"
        self.input_shape = None
    
    def forward(self, in_tensors: list, out_tensors: list) -> None:
        """
        Forward pass for flatten layer.
        
        Args:
            in_tensors: List containing input tensor with shape (batch_size, channels, height, width)
            out_tensors: List containing output tensor to store results
        """
        if not in_tensors or not out_tensors:
            return
            
        x = in_tensors[0].elements
        self.input_shape = x.shape  # Store input shape for backward pass
        
        # Flatten all dimensions except the batch dimension
        batch_size = x.shape[0]
        flattened = x.reshape(batch_size, -1)
        
        # Store output in the output tensor
        out_tensors[0].elements = flattened
        out_tensors[0].shape = Shape(*flattened.shape)
    
    def backward(self, out_tensors: list, in_tensors: list) -> None:
        """
        Backward pass for flatten layer.
        
        Args:
            out_tensors: List containing gradient tensor from next layer with shape (batch_size, flattened_size)
            in_tensors: List containing input tensor to store gradients with original shape (batch_size, channels, height, width)
        """
        if not out_tensors or not in_tensors or out_tensors[0].deltas is None:
            return
        
        # Reshape the gradient to match the input shape
        in_tensors[0].deltas = out_tensors[0].deltas.reshape(self.input_shape)
    
    def calc_delta_weights(self):
        """
        Calculate weight updates: No weights to update in flatten layer.
        """
        pass
