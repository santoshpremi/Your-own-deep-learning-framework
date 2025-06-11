import numpy as np
from .tensor import Tensor, Shape
from .layers import Layer

class BatchNormLayer(Layer):
    """Batch Normalization layer for 1-D feature vectors (after FC layers)."""
    def __init__(self, feature_dim: int, momentum: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.layer_type = "BatchNormLayer"
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.eps = eps

        # Learnable parameters: gamma (scale) & beta (shift)
        self.weights = Tensor(elements=np.ones(feature_dim), deltas=np.zeros(feature_dim))  # gamma
        self.bias = Tensor(elements=np.zeros(feature_dim), deltas=np.zeros(feature_dim))    # beta

        # Running statistics for inference (not used here but kept for completeness)
        self.running_mean = np.zeros(feature_dim)
        self.running_var = np.ones(feature_dim)

    def forward(self, in_tensors: list, out_tensors: list) -> None:
        x = in_tensors[0].elements  # shape (N, D)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        self.batch_mean = np.mean(x, axis=0)
        self.batch_var = np.var(x, axis=0)
        self.x_hat = (x - self.batch_mean) / np.sqrt(self.batch_var + self.eps)
        out_tensors[0].elements = self.weights.elements * self.x_hat + self.bias.elements
        # Store for backward
        self.input = in_tensors[0]

    def backward(self, out_tensors: list, in_tensors: list) -> None:
        dout = out_tensors[0].deltas  # (N, D)
        if dout.ndim == 1:
            dout = dout.reshape(1, -1)
        N = dout.shape[0]
        gamma = self.weights.elements
        x_hat = self.x_hat

        # Gradients w.r.t parameters
        self.weights.deltas = np.sum(dout * x_hat, axis=0)
        self.bias.deltas = np.sum(dout, axis=0)

        # Gradient w.r.t input
        dx_hat = dout * gamma
        var_eps = self.batch_var + self.eps
        dvar = np.sum(dx_hat * (self.input.elements - self.batch_mean) * -0.5 * np.power(var_eps, -1.5), axis=0)
        dmean = np.sum(dx_hat * -1.0 / np.sqrt(var_eps), axis=0) + dvar * np.mean(-2.0 * (self.input.elements - self.batch_mean), axis=0)
        dx = dx_hat / np.sqrt(var_eps) + dvar * 2.0 * (self.input.elements - self.batch_mean) / N + dmean / N
        in_tensors[0].deltas = dx

    def calculate_delta_weights(self, out_tensors: list, in_tensors: list) -> None:
        # Already computed in backward
        pass