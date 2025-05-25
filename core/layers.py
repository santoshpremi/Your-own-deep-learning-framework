from .tensor import Tensor, Shape
import numpy as np

class Layer:
    def __init__(self):
        self.layer_type = self.__class__.__name__  # Automatically get the class name as layer type
        self.num = None  # Will be set when added to network

    def forward(self, inp: Tensor) -> Tensor:
        """Forward pass of the layer"""
        raise NotImplementedError

    def backward(self, grad_out: Tensor) -> Tensor:
        """Backward pass of the layer"""
        raise NotImplementedError

    def calc_delta_weights(self):
        """Calculate weight updates for the layer"""
        pass  # Default implementation does nothing

class InputLayer(Layer):
    def __init__(self):
        super().__init__()
        self.layer_type = "InputLayer"  # Explicitly set layer type

    def forward(self, inp) -> Tensor:
        """Transform input into a Tensor"""
        if isinstance(inp, Tensor):
            return inp
        # Convert input to Tensor if not already
        elements = np.array(inp)
        deltas = None
        shape = Shape(*elements.shape) if elements.ndim > 1 else Shape(len(elements))
        return Tensor(elements, deltas, shape)

    def backward(self, grad_out: Tensor) -> Tensor:
        """No transformation needed in backward pass"""
        return grad_out

    def calc_delta_weights(self):
        """Input layer has no weights to update"""
        pass

class FullyConnectedLayer(Layer):
    """A fully connected layer that applies a linear transformation to the input
    attributes:
        input_size: the number of input neurons
        output_size: the number of output neurons
        weights: the weights of the layer initialized with random values in [-0.5, 0.5]
        biases: the biases of the layer initialized with random values in [-0.5, 0.5]
        input: the input to the layer
        output: the output of the layer
    """
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.layer_type = "FullyConnectedLayer"  # Explicitly set layer type
        self.input_size = input_size
        self.output_size = output_size
        # Initialize weight matrix with random values in [-0.5, 0.5]
        self.weights = Tensor(
            elements=np.random.uniform(-0.5, 0.5, (output_size, input_size)),
            deltas=np.zeros((output_size, input_size)),
            shape=Shape(output_size, input_size)
        )
        # Initialize bias vector with random values in [-0.5, 0.5]
        self.biases = Tensor(
            elements=np.random.uniform(-0.5, 0.5, output_size),
            deltas=np.zeros(output_size),
            shape=Shape(output_size)
        )
        self.input = None
        self.output = None

    def forward(self, inp: Tensor) -> Tensor:
        """Forward pass computes output = weights * input + biases"""
        self.input = inp
        if inp.is_batch:
            # Batch matrix multiplication
            output_elements = np.dot(inp.elements, self.weights.elements.T) + self.biases.elements
        else:
            # Single sample
            output_elements = np.dot(self.weights.elements, inp.elements) + self.biases.elements
        self.output = Tensor(
            elements=output_elements,
            deltas=np.zeros_like(output_elements),
            shape=Shape(self.output_size) if not inp.is_batch else Shape(inp.get_batch_size(), self.output_size)
        )
        return self.output

    def backward(self, grad_out: Tensor) -> Tensor:
        """Backward pass computes gradients for weights, biases and input"""
        if grad_out.is_batch:
            # Batch processing
            if self.input.is_batch:
                # Accumulate weight deltas
                self.weights.deltas += np.dot(grad_out.elements.T, self.input.elements)
                # Accumulate bias deltas
                self.biases.deltas += np.sum(grad_out.elements, axis=0)
                # Compute input gradients
                input_grads = np.dot(grad_out.elements, self.weights.elements)
            else:
                raise ValueError("Input and gradient batch sizes must match")
        else:
            # Single sample
            # Accumulate weight deltas
            self.weights.deltas += np.outer(grad_out.elements, self.input.elements)
            # Accumulate bias deltas
            self.biases.deltas += grad_out.elements
            # Compute input gradients
            input_grads = np.dot(grad_out.elements, self.weights.elements)
        return Tensor(
            elements=input_grads,
            deltas=None,
            shape=self.input.shape
        )

    def calc_delta_weights(self):
        """Reset deltas after applying updates"""
        self.weights.deltas = np.zeros_like(self.weights.elements)
        self.biases.deltas = np.zeros_like(self.biases.elements)