from .tensor import Tensor, Shape
import numpy as np
from enum import Enum

class PoolingType(Enum):
    MAX = "max"
    AVERAGE = "average"

class Layer:
    def __init__(self):
        self.layer_type = self.__class__.__name__  # Automatically get the class name as layer type
        self.num = None  # Will be set when added to network

    def forward(self, in_tensors: list, out_tensors: list) -> None:
        """Forward pass of the layer"""
        raise NotImplementedError

    def backward(self, out_tensors: list, in_tensors: list) -> None:
        """Backward pass of the layer"""
        raise NotImplementedError

    def calculate_delta_weights(self, out_tensors: list, in_tensors: list) -> None:
        """Calculate weight updates for the layer
        
        Args:
            out_tensors: List containing gradient tensor from next layer
            in_tensors: List containing input tensor
        """
        pass  # Default implementation does nothing

class InputLayer(Layer):
    def __init__(self):
        super().__init__()
        self.layer_type = "InputLayer"  # Explicitly set layer type

    def forward(self, in_tensors: list, out_tensors: list) -> None:
        """
        Forward pass for input layer
        Args:
            in_tensors: List containing input data (numpy array or Tensor)
            out_tensors: List to store output tensor
        """
        if not in_tensors:
            raise ValueError("Input tensor list is empty")
            
        # Get input data (first element in the list)
        inp = in_tensors[0]
        
        # Convert to numpy array if it's a Tensor
        if isinstance(inp, Tensor):
            elements = inp.elements
        else:
            elements = np.array(inp)
        
        # Ensure output tensor exists
        if not out_tensors:
            out_tensors.append(Tensor())
            
        # Store elements in output tensor
        out_tensors[0].elements = elements
        out_tensors[0].shape = Shape(*elements.shape) if elements.ndim > 1 else Shape(len(elements))

    def backward(self, out_tensors: list, in_tensors: list) -> None:
        """
        Backward pass for input layer
        Args:
            out_tensors: List containing gradient tensor from next layer
            in_tensors: List to store gradient tensor for previous layer
        """
        if not out_tensors or not in_tensors:
            return
            
        # Simply pass the gradient through
        in_tensors[0].deltas = out_tensors[0].deltas if hasattr(out_tensors[0], 'deltas') else None

    def calc_delta_weights(self):
        """Input layer has no weights to update"""
        pass

class FullyConnectedLayer(Layer):
    """A fully connected layer that applies a linear transformation to the input
    attributes:
        in_shape: Shape of the input
        out_shape: Shape of the output
        weights: the weights of the layer 
        bias: the biases of the layer
    """
    def __init__(self, in_shape: Shape, out_shape: Shape):
        super().__init__()
        self.layer_type = "FullyConnectedLayer"
        self.in_shape = in_shape
        self.out_shape = out_shape
        
        # Extract dimensions from shapes
        self.input_size = in_shape[0]
        self.output_size = out_shape[0]
        
        # Initialize weights and bias to match test values
        self.weights = Tensor(
            elements=np.random.uniform(-0.5, 0.5, (self.input_size, self.output_size)),
            deltas=np.zeros((self.input_size, self.output_size))
        )
        self.bias = Tensor(  # Note: Using bias instead of biases to match test
            elements=np.zeros(self.output_size),
            deltas=np.zeros(self.output_size)
        )

    def forward(self, in_tensors: list, out_tensors: list) -> None:
        """Forward pass using matrix operations"""
        # Compute output: input × weights + bias
        out_tensors[0].elements = np.dot(in_tensors[0].elements, self.weights.elements) + self.bias.elements

    def backward(self, out_tensors: list, in_tensors: list) -> None:
        """Backward pass using matrix operations"""
        # Compute input gradients: output_grad × weights.T
        in_tensors[0].deltas = np.dot(out_tensors[0].deltas, self.weights.elements.T)

    def calculate_delta_weights(self, out_tensors: list, in_tensors: list) -> None:
        """Calculate weight and bias updates"""
        # For weights: input × output_grad (transposed appropriately)
        if in_tensors[0].elements.ndim == 1:
            self.weights.deltas = np.outer(in_tensors[0].elements, out_tensors[0].deltas)
        else:
            self.weights.deltas = np.dot(in_tensors[0].elements.T, out_tensors[0].deltas)
        # For bias: sum over batch if 2-D else direct
        if out_tensors[0].deltas.ndim == 2:
            self.bias.deltas = np.sum(out_tensors[0].deltas, axis=0)
        else:
            self.bias.deltas = out_tensors[0].deltas

class Conv2DLayer(Layer):
    def __init__(self, in_shape: Shape, out_shape: Shape, kernel_size: Shape, num_filters: int, stride: Shape = None, dilation: int = 1):
        super().__init__()
        self.layer_type = "Conv2DLayer"
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.stride = stride if stride is not None else Shape(1, 1)
        self.dilation = dilation
        
        # weights have shape (2, 2, 2, 2) in tests
        # filters have the shape (rows, columns, num_channels, num_filters)
        self.weights = Tensor(
            elements=np.random.uniform(-0.5, 0.5, (kernel_size.dimensions[0], kernel_size.dimensions[1], in_shape.dimensions[2], num_filters)),
            deltas=np.zeros((kernel_size.dimensions[0], kernel_size.dimensions[1], in_shape.dimensions[2], num_filters))
        )
        
        # biases have shape (num_filters,)
        self.bias = Tensor(
            elements=np.zeros(num_filters),
            deltas=np.zeros(num_filters)
        )

    def forward(self, in_tensors: list, out_tensors: list) -> None:
        """
        Forward pass: Y = X * F + bias
        Args:
            in_tensors: List containing input tensor of shape (height, width, channels)
            out_tensors: List containing output tensor to store results
        """
        inp = in_tensors[0]
        in_height, in_width, in_channels = inp.elements.shape
        out_height = (in_height - self.kernel_size.dimensions[0]) // self.stride.dimensions[0] + 1
        out_width = (in_width - self.kernel_size.dimensions[1]) // self.stride.dimensions[1] + 1
        
        # Initialize output elements (height, width, num_filters)
        output = np.zeros((out_height, out_width, self.num_filters))
        
        # Perform convolution
        for h in range(out_height):
            for w in range(out_width):
                h_start = h * self.stride.dimensions[0]
                h_end = h_start + self.kernel_size.dimensions[0]
                w_start = w * self.stride.dimensions[1]
                w_end = w_start + self.kernel_size.dimensions[1]
                
                # Extract input patch: (kernel_h, kernel_w, in_channels)
                patch = inp.elements[h_start:h_end, w_start:w_end, :]
                
                # Calculate output for each filter
                for f in range(self.num_filters):
                    # weights[:, :, :, f] has shape (kernel_h, kernel_w, in_channels)
                    output[h, w, f] = np.sum(patch * self.weights.elements[:, :, :, f]) + self.bias.elements[f]
        
        # Store output
        out_tensors[0].elements = output

    def backward(self, out_tensors: list, in_tensors: list) -> None:
        """
        Backward pass: δX = δY * F_rot180
        Args:
            out_tensors: List containing gradient tensor from next layer
            in_tensors: List containing input tensor to store gradients
        """
        grad_out = out_tensors[0]
        in_height, in_width, in_channels = in_tensors[0].elements.shape
        out_height, out_width = grad_out.elements.shape[:2]
        
        # Initialize input gradient
        input_grads = np.zeros((in_height, in_width, in_channels))
        
        # For each position in the output gradient
        for h in range(out_height):
            for w in range(out_width):
                h_start = h * self.stride.dimensions[0]
                h_end = h_start + self.kernel_size.dimensions[0]
                w_start = w * self.stride.dimensions[1]
                w_end = w_start + self.kernel_size.dimensions[1]
                
                # Process deltas for each filter and channel
                for f in range(self.num_filters):
                    delta = grad_out.deltas[h, w, f]
                    for c in range(in_channels):
                        # Add contributions from each filter
                        input_grads[h_start:h_end, w_start:w_end, c] += \
                            delta * self.weights.elements[:, :, c, f]
        
        # Store gradients
        in_tensors[0].deltas = input_grads

    def calculate_delta_weights(self, out_tensors: list, in_tensors: list) -> None:
        """
        Calculate weight and bias updates for the convolutional layer
        Args:
            out_tensors: List containing output tensor with gradients
            in_tensors: List containing input tensor
        """
        in_tensor = in_tensors[0]
        out_tensor = out_tensors[0]
        out_height, out_width = out_tensor.elements.shape[:2]
        
        # Initialize gradients
        weight_grads = np.zeros_like(self.weights.elements)  # (kernel_h, kernel_w, in_channels, num_filters)
        bias_grads = np.zeros_like(self.bias.elements)  # (num_filters,)
        
        # Calculate gradients
        for h in range(out_height):
            for w in range(out_width):
                h_start = h * self.stride.dimensions[0]
                h_end = h_start + self.kernel_size.dimensions[0]
                w_start = w * self.stride.dimensions[1]
                w_end = w_start + self.kernel_size.dimensions[1]
                
                # Extract input patch (kernel_h, kernel_w, in_channels)
                patch = in_tensor.elements[h_start:h_end, w_start:w_end, :]
                
                for f in range(self.num_filters):
                    # Update weights
                    weight_grads[:, :, :, f] += patch * out_tensor.deltas[h, w, f]
                    # Update bias
                    bias_grads[f] += out_tensor.deltas[h, w, f]
        
        # Store gradients
        self.weights.deltas = weight_grads
        self.bias.deltas = bias_grads

class Pooling2DLayer(Layer):
    def __init__(self, kernel_size: Shape, stride: Shape, pooling_type: str = "max", in_shape: Shape = None, out_shape: Shape = None):
        super().__init__()
        self.layer_type = "Pooling2DLayer"
        self.kernel_size = kernel_size
        self.stride = stride
        if isinstance(pooling_type, str):
            pooling_type = PoolingType(pooling_type.lower())
        self.pooling_type = pooling_type
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.max_indices = []  # Store indices for max pooling backward pass

    def forward(self, in_tensors: list, out_tensors: list) -> None:
        """
        Forward pass for pooling layer
        Args:
            in_tensors: List containing input tensor with shape (batch_size, channels, height, width)
            out_tensors: List containing output tensor to store results
        """
        inp = in_tensors[0]
        batch_size, channels, in_height, in_width = inp.elements.shape
        out_height = (in_height - self.kernel_size.dimensions[0]) // self.stride.dimensions[0] + 1
        out_width = (in_width - self.kernel_size.dimensions[1]) // self.stride.dimensions[1] + 1
        
        # Initialize output elements with shape (batch_size, channels, out_height, out_width)
        output = np.zeros((batch_size, channels, out_height, out_width))
        self.max_indices = []  # Reset indices for new forward pass
        
        # Perform pooling for each sample in the batch
        for b in range(batch_size):
            batch_max_indices = []
            
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride.dimensions[0]
                        h_end = h_start + self.kernel_size.dimensions[0]
                        w_start = w * self.stride.dimensions[1]
                        w_end = w_start + self.kernel_size.dimensions[1]
                        
                        # Extract window for this channel
                        window = inp.elements[b, c, h_start:h_end, w_start:w_end]
                        
                        if self.pooling_type == PoolingType.MAX:
                            # Max pooling
                            max_val = np.max(window)
                            output[b, c, h, w] = max_val
                            
                            # Find position of max value
                            max_pos = np.unravel_index(np.argmax(window), window.shape)
                            batch_max_indices.append((c, h, w, h_start + max_pos[0], w_start + max_pos[1]))
                        else:
                            # Average pooling
                            output[b, c, h, w] = np.mean(window)
            
            if self.pooling_type == PoolingType.MAX:
                self.max_indices.append(batch_max_indices)
        
        # Store output in the output tensor
        out_tensors[0].elements = output
        out_tensors[0].shape = Shape(*output.shape)

    def backward(self, out_tensors: list, in_tensors: list) -> None:
        """
        Backward pass for pooling layer
        Args:
            out_tensors: List containing gradient tensor from next layer with shape (batch_size, channels, out_h, out_w)
            in_tensors: List containing input tensor to store gradients with shape (batch_size, channels, in_h, in_w)
        """
        if not out_tensors or not in_tensors or not out_tensors[0].deltas.any():
            return
            
        out_grad = out_tensors[0].deltas  # Shape: (batch_size, channels, out_h, out_w)
        batch_size, channels, out_h, out_w = out_grad.shape
        in_shape = in_tensors[0].elements.shape
        in_grad = np.zeros_like(in_tensors[0].elements)  # Shape: (batch_size, channels, in_h, in_w)
        
        if self.pooling_type == PoolingType.MAX:
            # For max pooling, only the max index gets the gradient
            for b in range(batch_size):  # For each sample in batch
                for c, h, w, max_h, max_w in self.max_indices[b]:
                    in_grad[b, c, max_h, max_w] += out_grad[b, c, h, w]
        else:
            # For average pooling, distribute gradient equally
            pool_size = self.kernel_size.dimensions[0] * self.kernel_size.dimensions[1]
            
            for b in range(batch_size):
                for c in range(channels):
                    for h in range(out_h):
                        for w in range(out_w):
                            h_start = h * self.stride.dimensions[0]
                            h_end = h_start + self.kernel_size.dimensions[0]
                            w_start = w * self.stride.dimensions[1]
                            w_end = w_start + self.kernel_size.dimensions[1]
                            
                            grad = out_grad[b, c, h, w] / pool_size
                            in_grad[b, c, h_start:h_end, w_start:w_end] += grad
        
        in_tensors[0].deltas = in_grad

    def calc_delta_weights(self):
        """
        Calculate weight updates: ∂L/∂f = X * δY, ∂L/∂bias = sum(δY)
        """
        batch_size = self.input.elements.shape[0]
        out_height, out_width = self.output.elements.shape[1:3]
        
        # Initialize weight gradients
        weight_grads = np.zeros_like(self.weights.elements)
        bias_grads = np.zeros_like(self.biases.elements)
        
        # Compute weight gradients
        for b in range(batch_size):
            for f in range(self.num_filters):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride.dimensions[0]
                        w_start = w * self.stride.dimensions[1]
                        h_end = h_start + self.kernel_size.dimensions[0]
                        w_end = w_start + self.kernel_size.dimensions[1]
                        
                        # Extract input patch
                        patch = self.input.elements[b, h_start:h_end, w_start:w_end, :]
                        
                        # Update weight gradients
                        weight_grads[f] += patch * self.output.deltas[b, h, w, f]
                        
                        # Update bias gradients
                        bias_grads[f] += self.output.deltas[b, h, w, f]
        
        self.weights.deltas = weight_grads
        self.biases.deltas = bias_grads