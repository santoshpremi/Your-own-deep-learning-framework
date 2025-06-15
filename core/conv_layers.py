"""
Convolutional neural network layers implementation.
"""

import numpy as np
from typing import Tuple, Optional, Union, Literal
from .tensor import Tensor, Shape
from .layers import Layer

class Conv2DLayer(Layer):
    """2D Convolutional Layer.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (number of filters)
        kernel_size: Size of the convolving kernel (int or tuple)
        stride: Stride of the convolution (int or tuple)
        padding: Padding added to all four sides of the input (int or tuple)
        dilation: Spacing between kernel elements (int or tuple)
        use_bias: Whether to add a bias term
    """
    
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 use_bias: bool = True):
        
        # Convert to tuple if single int is provided
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        super().__init__()
        self.layer_type = "Conv2DLayer"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = use_bias
        
        # Initialize weights using He initialization
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        scale = np.sqrt(2.0 / fan_in)
        
        # Initialize filters (out_channels, in_channels, kH, kW)
        self.weights = Tensor(
            elements=np.random.randn(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]) * scale,
            shape=Shape(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1])
        )
        
        # Initialize bias
        if self.use_bias:
            self.bias = Tensor(
                elements=np.zeros(out_channels),
                shape=Shape(out_channels)
            )
        else:
            self.bias = None
            
        # Gradients and cache
        self.d_weights = None
        self.d_bias = None
        self.input_shape = None
        self.output_shape = None
        self.cache = {}
    
    def _get_output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Calculate output shape given input shape (C, H, W)."""
        C, H, W = input_shape
        pad_h, pad_w = self.padding
        kH, kW = self.kernel_size
        stride_h, stride_w = self.stride
        dilation_h, dilation_w = self.dilation
        
        out_h = (H + 2 * pad_h - dilation_h * (kH - 1) - 1) // stride_h + 1
        out_w = (W + 2 * pad_w - dilation_w * (kW - 1) - 1) // stride_w + 1
        
        return (self.out_channels, out_h, out_w)
    
    def _pad_input(self, x: np.ndarray) -> np.ndarray:
        """Apply padding to input."""
        if np.sum(self.padding) == 0:
            return x
        
        pad_width = ((0, 0), 
                     (0, 0), 
                     (self.padding[0], self.padding[0]), 
                     (self.padding[1], self.padding[1]))
        return np.pad(x, pad_width, mode='constant')
    
    def forward(self, in_tensors: list, out_tensors: list) -> None:
        """Optimized forward pass for Conv2D layer using numpy stride tricks (im2col-like)."""
        x = in_tensors[0]
        x_data = x.elements
        batch_size, in_channels, H, W = x_data.shape
        self.input_shape = x_data.shape
        self.output_shape = self._get_output_shape((in_channels, H, W))
        out_channels, out_h, out_w = self.output_shape
        x_padded = self._pad_input(x_data)
        kH, kW = self.kernel_size
        sH, sW = self.stride
        # im2col: create a matrix where each row is a patch to be convolved
        shape = (
            batch_size,
            in_channels,
            out_h,
            out_w,
            kH,
            kW
        )
        strides = (
            x_padded.strides[0],
            x_padded.strides[1],
            x_padded.strides[2] * sH,
            x_padded.strides[3] * sW,
            x_padded.strides[2],
            x_padded.strides[3]
        )
        patches = np.lib.stride_tricks.as_strided(
            x_padded,
            shape=shape,
            strides=strides,
            writeable=False
        )
        # Reshape patches to (batch_size*out_h*out_w, in_channels*kH*kW)
        patches_reshaped = patches.transpose(0,2,3,1,4,5).reshape(batch_size*out_h*out_w, -1)
        # Reshape filters to (out_channels, in_channels*kH*kW)
        filters_reshaped = self.weights.elements.reshape(out_channels, -1)
        # Matrix multiply and reshape to (batch_size, out_channels, out_h, out_w)
        output = patches_reshaped @ filters_reshaped.T
        output = output.reshape(batch_size, out_h, out_w, out_channels).transpose(0,3,1,2)
        if self.use_bias:
            output += self.bias.elements.reshape(1, -1, 1, 1)
        self.cache['x_padded'] = x_padded
        out_tensors[0].elements = output
        out_tensors[0].shape = Shape(*output.shape)
    
    def backward(self, out_tensors: list, in_tensors: list) -> None:
        """Optimized backward pass for Conv2D layer using numpy vectorization (im2col-like)."""
        d_out_data = out_tensors[0].deltas
        batch_size, out_channels, out_h, out_w = d_out_data.shape
        kH, kW = self.kernel_size
        sH, sW = self.stride
        x_padded = self.cache['x_padded']
        in_channels = self.in_channels
        # im2col for input patches
        shape = (
            batch_size,
            in_channels,
            out_h,
            out_w,
            kH,
            kW
        )
        strides = (
            x_padded.strides[0],
            x_padded.strides[1],
            x_padded.strides[2] * sH,
            x_padded.strides[3] * sW,
            x_padded.strides[2],
            x_padded.strides[3]
        )
        patches = np.lib.stride_tricks.as_strided(
            x_padded,
            shape=shape,
            strides=strides,
            writeable=False
        )
        # Reshape for easier computation
        patches_reshaped = patches.transpose(0,2,3,1,4,5).reshape(batch_size*out_h*out_w, -1)
        d_out_flat = d_out_data.transpose(0,2,3,1).reshape(-1, out_channels)
        # Compute gradients for weights
        d_weights = d_out_flat.T @ patches_reshaped
        d_weights = d_weights.reshape(self.out_channels, self.in_channels, kH, kW)
        self.d_weights = d_weights
        # Compute gradients for bias
        if self.use_bias:
            self.d_bias = np.sum(d_out_data, axis=(0,2,3))
        # Compute gradient w.r.t input (dx)
        filters_reshaped = self.weights.elements.reshape(self.out_channels, -1)
        d_patches = d_out_flat @ filters_reshaped
        d_patches = d_patches.reshape(batch_size, out_h, out_w, kH, kW, in_channels)
        dx_padded = np.zeros_like(x_padded)
        for i in range(out_h):
            for j in range(out_w):
                # d_patches[:, i, j] shape: (batch_size, kH, kW, in_channels)
                # We need (batch_size, in_channels, kH, kW) to match dx_padded[:, :, ...]
                patch = d_patches[:, i, j].transpose(0, 3, 1, 2)
                dx_padded[:, :, i*sH:i*sH+kH, j*sW:j*sW+kW] += patch
        # Remove padding
        if np.sum(self.padding) > 0:
            pad_h, pad_w = self.padding
            dx = dx_padded[:, :, pad_h:-pad_h, pad_w:-pad_w]
        else:
            dx = dx_padded
        in_tensors[0].deltas = dx
        self.weights.deltas = self.d_weights
        if self.use_bias:
            self.bias.deltas = self.d_bias
        return Tensor(elements=dx, shape=Shape(*dx.shape))
    
    def calculate_delta_weights(self, out_tensors: list, in_tensors: list) -> None:
        """Calculate weight gradients (already done in backward pass)."""
        # Weight gradients are calculated in backward() method
        # This method exists to satisfy the layer interface
        pass
    
    def update_weights(self, learning_rate: float) -> None:
        """Update weights using gradient descent."""
        if self.d_weights is not None:
            self.weights.elements -= learning_rate * self.d_weights
            if self.use_bias and self.d_bias is not None:
                self.bias.elements -= learning_rate * self.d_bias


class Pooling2D(Layer):
    """2D Pooling Layer.
    
    Args:
        pool_size: Size of the pooling window (int or tuple)
        stride: Stride of the pooling operation (int or tuple)
        padding: Padding added to all four sides (int or tuple)
        mode: Pooling mode ('max' or 'avg')
    """
    
    def __init__(self, 
                 pool_size: Union[int, Tuple[int, int]] = 2,
                 stride: Optional[Union[int, Tuple[int, int]]] = None,
                 padding: Union[int, Tuple[int, int]] = 0,
                 mode: Literal['max', 'avg'] = 'max'):
        
        self.pool_size = (pool_size, pool_size) if isinstance(pool_size, int) else pool_size
        self.stride = stride if stride is not None else self.pool_size
        self.stride = (self.stride, self.stride) if isinstance(self.stride, int) else self.stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.mode = mode.lower()
        
        if self.mode not in ['max', 'avg']:
            raise ValueError("Mode must be either 'max' or 'avg'")
            
        super().__init__()
        self.layer_type = "Pooling2D"
        self.cache = {}
    
    def _get_output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Calculate output shape given input shape (C, H, W)."""
        C, H, W = input_shape
        pad_h, pad_w = self.padding
        pool_h, pool_w = self.pool_size
        stride_h, stride_w = self.stride
        
        out_h = (H + 2 * pad_h - pool_h) // stride_h + 1
        out_w = (W + 2 * pad_w - pool_w) // stride_w + 1
        
        return (C, out_h, out_w)
    
    def _pad_input(self, x: np.ndarray) -> np.ndarray:
        """Apply padding to input."""
        if np.sum(self.padding) == 0:
            return x
        
        pad_width = ((0, 0), 
                     (0, 0), 
                     (self.padding[0], self.padding[0]), 
                     (self.padding[1], self.padding[1]))
        return np.pad(x, pad_width, mode='constant')
    
    def forward(self, in_tensors: list, out_tensors: list) -> None:
        """Forward pass for Pooling layer using vectorized operations.
        
        Args:
            in_tensors: List containing input tensor
            out_tensors: List to store output tensor
        """
        x = in_tensors[0]
        # Input shape: (batch_size, channels, height, width)
        x_data = x.elements
        batch_size, channels, H, W = x_data.shape
        
        # Pad input if needed
        x_padded = self._pad_input(x_data)
        
        # Calculate output shape
        out_h = (H + 2 * self.padding[0] - self.pool_size[0]) // self.stride[0] + 1
        out_w = (W + 2 * self.padding[1] - self.pool_size[1]) // self.stride[1] + 1
        
        # Initialize output
        output = np.zeros((batch_size, channels, out_h, out_w))
        
        # Store for backward pass
        self.cache['x_padded'] = x_padded
        
        # Create windows for vectorized operations
        # Shape: (batch_size, channels, out_h, out_w, pool_h, pool_w)
        windows = np.lib.stride_tricks.sliding_window_view(
            x_padded, 
            (1, 1, self.pool_size[0], self.pool_size[1])
        )
        # Select windows according to stride
        windows = windows[:, :, ::self.stride[0], ::self.stride[1]]
        # Reshape to (batch_size, channels, out_h, out_w, pool_size[0]*pool_size[1])
        windows = windows.reshape(batch_size, channels, out_h, out_w, -1)
        
        if self.mode == 'max':
            # Max pooling
            output = np.max(windows, axis=4)
            # Store indices of max values for backward pass
            max_indices_flat = np.argmax(windows, axis=4)
            # Convert flat indices to 2D indices
            rows = max_indices_flat // self.pool_size[1]
            cols = max_indices_flat % self.pool_size[1]
            # Calculate global positions
            h_pos = np.arange(out_h)[:, np.newaxis] * self.stride[0] + rows
            w_pos = np.arange(out_w)[np.newaxis, :] * self.stride[1] + cols
            # Store for backward pass
            self.cache['max_indices'] = np.stack([h_pos, w_pos], axis=-1)
        else:
            # Average pooling
            output = np.mean(windows, axis=4)
        
        # Store output in the provided output tensor
        out_tensors[0].elements = output
        out_tensors[0].shape = Shape(*output.shape)
    
    def backward(self, out_tensors: list, in_tensors: list) -> None:
        """Backward pass for Pooling layer using vectorized operations.
        
        Args:
            out_tensors: List containing gradient tensor from next layer
            in_tensors: List to store gradient tensor for previous layer
        """
        d_out_data = out_tensors[0].deltas
        batch_size, channels, out_h, out_w = d_out_data.shape
        
        # Initialize gradient with padding
        dx = np.zeros_like(self.cache['x_padded'])
        
        if self.mode == 'max':
            # Max pooling backward - vectorized
            # Create a flat view of the output gradient
            d_out_flat = d_out_data.reshape(-1)
            
            # Get the max indices from forward pass
            h_indices = self.cache['max_indices'][..., 0].flatten()
            w_indices = self.cache['max_indices'][..., 1].flatten()
            
            # Create batch and channel indices
            b_indices = np.repeat(np.arange(batch_size), channels * out_h * out_w)
            c_indices = np.tile(np.repeat(np.arange(channels), out_h * out_w), batch_size)
            
            # Use np.add.at to accumulate gradients at max positions
            np.add.at(dx, (b_indices, c_indices, h_indices, w_indices), d_out_flat)
            
        else:
            # Average pooling backward - vectorized
            pool_area = self.pool_size[0] * self.pool_size[1]
            
            # Create a view of the output gradient with the same shape as windows
            d_out_expanded = np.repeat(
                np.repeat(d_out_data, self.pool_size[0], axis=2),
                self.pool_size[1],
                axis=3
            )
            
            # Slice to match the input size
            d_out_expanded = d_out_expanded[:, :, :dx.shape[2], :dx.shape[3]]
            
            # Distribute gradients equally
            dx = d_out_expanded / pool_area
        
        # Remove padding if any
        if np.sum(self.padding) > 0:
            pad_h, pad_w = self.padding
            dx = dx[:, :, pad_h:-pad_h, pad_w:-pad_w]
        
        # Store gradient in input tensor
        in_tensors[0].deltas = dx
    
    def calculate_delta_weights(self, out_tensors: list, in_tensors: list) -> None:
        """Calculate weight updates for the pooling layer.
        
        Args:
            out_tensors: List containing gradient tensor from next layer
            in_tensors: List containing input tensor
            
        Note: Pooling layers typically don't have learnable parameters,
        so this is a no-op method to satisfy the interface.
        """
        pass
