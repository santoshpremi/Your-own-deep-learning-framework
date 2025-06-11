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

class SigmoidLayer(Layer):
    def __init__(self):
        super().__init__()
        self.layer_type = "SigmoidLayer"
        
    def forward(self, in_tensors: list, out_tensors: list) -> None:
        """Forward pass applies sigmoid activation"""
        self.input = in_tensors[0]
        out_tensors[0].elements = 1 / (1 + np.exp(-in_tensors[0].elements))
        
    def backward(self, out_tensors: list, in_tensors: list) -> None:
        """Backward pass computes sigmoid gradient"""
        sigmoid_out = 1 / (1 + np.exp(-in_tensors[0].elements))
        in_tensors[0].deltas = out_tensors[0].deltas * sigmoid_out * (1 - sigmoid_out)

class TanhActivation:
    @staticmethod
    def forward(x):
        return np.tanh(x)

    @staticmethod
    def backward(x):
        return 1 - np.square(np.tanh(x))

class TanhLayer(Layer):
    def __init__(self):
        super().__init__()
        self.layer_type = "TanhLayer"

    def forward(self, in_tensors: list, out_tensors: list) -> None:
        """Forward pass applies tanh activation"""
        self.input = in_tensors[0]
        out_tensors[0].elements = np.tanh(in_tensors[0].elements)

    def backward(self, out_tensors: list, in_tensors: list) -> None:
        """Backward pass computes tanh gradient"""
        tanh_out = np.tanh(self.input.elements)
        in_tensors[0].deltas = out_tensors[0].deltas * (1 - np.square(tanh_out))

class SoftmaxLayer(Layer):
    def __init__(self):
        super().__init__()
        self.layer_type = "SoftmaxLayer"

    def forward(self, in_tensors: list, out_tensors: list) -> None:
        """Forward pass applies softmax activation"""
        self.input = in_tensors[0]
        x_in = self.input.elements
        if x_in.ndim == 1:  # single sample
            x_shift = x_in - np.max(x_in)
            exp_x = np.exp(x_shift)
            self.output = exp_x / np.sum(exp_x)
        else:
            x_shift = x_in - np.max(x_in, axis=1, keepdims=True)
            exp_x = np.exp(x_shift)
            self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        out_tensors[0].elements = self.output
        
    def backward(self, out_tensors: list, in_tensors: list) -> None:
        """
        Backward pass for softmax layer.

        When using Softmax with Cross-Entropy loss, the gradient of the loss
        with respect to the input of the softmax layer simplifies to
        (prediction - target). This combined gradient is calculated in the
        network's `calculate_loss_gradient` method.

        Therefore, this backward pass is a simple pass-through of the gradient
        from the loss function to the previous layer.
        """
        grad_out = out_tensors[0].deltas
        sm = out_tensors[0].elements  # already forward output

        if sm.ndim == 1:
            jacobian = np.diag(sm) - np.outer(sm, sm)
            in_tensors[0].deltas = np.dot(grad_out, jacobian)
        else:
            # batch formula: dL/dz_i = s_i * (dL/dy_i - sum_j dL/dy_j * s_j)
            sum_term = np.sum(grad_out * sm, axis=1, keepdims=True)
            in_tensors[0].deltas = sm * (grad_out - sum_term)

class ReLULayer(Layer):
    def __init__(self):
        super().__init__()
        self.layer_type = "ReLULayer"
        
    def forward(self, in_tensors: list, out_tensors: list) -> None:
        """Forward pass applies ReLU activation"""
        out_tensors[0].elements = np.maximum(0, in_tensors[0].elements)
        self.input = in_tensors[0]  # Store for backward pass
        
    def backward(self, out_tensors: list, in_tensors: list) -> None:
        """Backward pass computes ReLU gradient"""
        in_tensors[0].deltas = out_tensors[0].deltas * (in_tensors[0].elements > 0)