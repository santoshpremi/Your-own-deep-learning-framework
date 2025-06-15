import numpy as np
from .tensor import Tensor, Shape
from .layers import Layer

class DropoutLayer(Layer):
    def __init__(self, rate=0.5):
        super().__init__()
        self.layer_type = "DropoutLayer"
        self.rate = rate
        self.mask = None
        self.training = True
    
    def forward(self, in_tensors: list, out_tensors: list) -> None:
        x = in_tensors[0].elements
        if self.training:
            self.mask = np.random.binomial(1, 1-self.rate, size=x.shape) / (1-self.rate)
            out_tensors[0].elements = x * self.mask
        else:
            out_tensors[0].elements = x
    
    def backward(self, out_tensors: list, in_tensors: list) -> None:
        if self.training and self.mask is not None:
            in_tensors[0].deltas = out_tensors[0].deltas * self.mask
        else:
            in_tensors[0].deltas = out_tensors[0].deltas
    
    def calculate_delta_weights(self, out_tensors: list, in_tensors: list) -> None:
        pass