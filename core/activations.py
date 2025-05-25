from .layers import Layer
from .tensor import Tensor, Shape
import numpy as np

class ActivationLayer(Layer):
    def __init__(self, activation_fn):
        super().__init__()
        self.layer_type = "ActivationLayer"  # Explicitly set layer type
        self.activation_fn = activation_fn

    def forward(self, inp: Tensor) -> Tensor:
        """Forward pass applies activation function"""
        self.input = inp
        output_elements = self.activation_fn.forward(inp.elements)
        self.output = Tensor(
            elements=output_elements,
            deltas=np.zeros_like(output_elements),
            shape=inp.shape
        )
        return self.output

    def backward(self, grad_out: Tensor) -> Tensor:
        """Backward pass computes gradients for activation function"""
        input_grads = grad_out.elements * self.activation_fn.backward(self.input.elements)
        return Tensor(elements=input_grads, deltas=None, shape=self.input.shape)

    def calc_delta_weights(self):
        """Activation layer has no weights to update"""
        pass

class SigmoidActivation:
    @staticmethod
    def forward(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def backward(x):
        s = 1 / (1 + np.exp(-x))
        return s * (1 - s)

class SigmoidLayer(ActivationLayer):
    def __init__(self):
        super().__init__(SigmoidActivation())
        self.layer_type = "SigmoidLayer"  # Override parent's layer type

class SoftmaxLayer(Layer):
    def __init__(self):
        super().__init__()
        self.layer_type = "SoftmaxLayer"  # Explicitly set layer type
        self.input = None
        self.output = None

    def forward(self, inp: Tensor) -> Tensor:
        """Forward pass applies softmax function"""
        self.input = inp
        # Subtract max for numerical stability
        if inp.is_batch:
            exp_elements = np.exp(inp.elements - np.max(inp.elements, axis=1, keepdims=True))
            sum_exp = np.sum(exp_elements, axis=1, keepdims=True)
            output_elements = exp_elements / sum_exp
        else:
            exp_elements = np.exp(inp.elements - np.max(inp.elements))
            sum_exp = np.sum(exp_elements)
            output_elements = exp_elements / sum_exp
        self.output = Tensor(
            elements=output_elements,
            deltas=np.zeros_like(output_elements),
            shape=inp.shape
        )
        return self.output

    def backward(self, grad_out) -> Tensor:
        """Backward pass computes gradients for softmax"""
        if isinstance(grad_out, np.ndarray):
            grad_out = Tensor(elements=grad_out, deltas=None, shape=Shape(len(grad_out)))
        if grad_out.is_batch:
            # Vectorized 
            input_grads = self.output.elements * (grad_out.elements - (grad_out.elements * self.output.elements).sum(axis=1, keepdims=True))
        else:
            input_grads = self.output.elements * (grad_out.elements - np.dot(grad_out.elements, self.output.elements))          
        return Tensor(
            elements=input_grads,
            deltas=None,
            shape=self.input.shape
        )

    def calc_delta_weights(self):
        """Softmax layer has no weights to update"""
        pass