from typing import List
import numpy as np
import os
import pickle
from core.tensor import Tensor, Shape
from core.layers import Layer

class Network:
    def __init__(self, layers: List[Layer] = None, learning_rate=0.01):
        """Initialize network with layers and learning rate"""
        self.layers = layers if layers is not None else []
        self.learning_rate = learning_rate
        self.tensors = []  # To store intermediate tensors for backpropagation

    def add_layer(self, layer: Layer):
        """Add a layer to the network and assign it a unique number"""
        layer.num = len(self.layers)
        self.layers.append(layer)

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform forward pass through all layers.
        Args:
            x: Input data tensor
        Returns:
            Tensor: Network output
        """
        self.tensors = []
        current_tensor = x

        for layer in self.layers:
            self.tensors.append(current_tensor)

            if layer.layer_type == "InputLayer":
                current_tensor = layer.forward(current_tensor)
            else:
                batch_size = current_tensor.elements.shape[0]
                if hasattr(layer, 'out_shape'):
                    out_dim = layer.out_shape.dimensions[0]
                    out_shape = Shape(batch_size, out_dim)
                else:
                    out_shape = current_tensor.shape
                
                out_tensor = Tensor(elements=np.zeros(out_shape.dimensions), shape=out_shape)
                layer.forward([current_tensor], [out_tensor])
                current_tensor = out_tensor
        
        self.tensors.append(current_tensor)
        return current_tensor

    def backward(self, grad: Tensor):
        """
        Perform backward pass (backpropagation) through all layers.
        Args:
            grad: Gradient from the loss function (as a Tensor object)
        """
        self.tensors[-1].deltas = grad.elements

        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            out_tensor = self.tensors[i + 1]
            in_tensor = self.tensors[i]
            
            if layer.layer_type != "InputLayer":
                layer.backward([out_tensor], [in_tensor])

    def update_weights(self):
        """Update weights using gradient descent"""
        for i, layer in enumerate(self.layers):
            if not hasattr(layer, 'weights'):
                continue

            in_tensor = self.tensors[i]
            out_tensor = self.tensors[i + 1]

            if hasattr(layer, 'calculate_delta_weights'):
                layer.calculate_delta_weights([out_tensor], [in_tensor])

            layer.weights.elements -= self.learning_rate * layer.weights.deltas
            layer.weights.deltas.fill(0)
            
            if hasattr(layer, 'bias'):
                layer.bias.elements -= self.learning_rate * layer.bias.deltas
                layer.bias.deltas.fill(0)

    def calculate_loss(self, output: Tensor, target: Tensor) -> float:
        """Calculate cross entropy loss"""
        epsilon = 1e-15
        predictions = np.clip(output.elements, epsilon, 1 - epsilon)
        N = target.elements.shape[0]
        ce_loss = -np.sum(target.elements * np.log(predictions)) / N
        return ce_loss

    def calculate_loss_gradient(self, output: Tensor, target: Tensor) -> Tensor:
        """Calculate gradient of cross entropy loss with respect to output"""
        grad = output.elements - target.elements
        return Tensor(elements=grad)

    def train_step(self, x: Tensor, y: Tensor) -> float:
        """Perform one training step (forward pass, backward pass, parameter update)"""
        output = self.forward(x)
        loss = self.calculate_loss(output, y)
        grad = self.calculate_loss_gradient(output, y)
        self.backward(grad)
        self.update_weights()
        return loss

    def save_params(self, folder_path):
        """Save network parameters to files"""
        os.makedirs(folder_path, exist_ok=True)
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                layer_id = f"{layer.layer_type}_{layer.num}"
                weights_file = os.path.join(folder_path, f"{layer_id}_weights.pkl")
                with open(weights_file, 'wb') as f:
                    pickle.dump(layer.weights.elements, f)
                if hasattr(layer, 'bias'):
                    bias_file = os.path.join(folder_path, f"{layer_id}_bias.pkl")
                    with open(bias_file, 'wb') as f:
                        pickle.dump(layer.bias.elements, f)

    def load_params(self, folder_path):
        """Load network parameters from files"""
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                layer_id = f"{layer.layer_type}_{layer.num}"
                weights_file = os.path.join(folder_path, f"{layer_id}_weights.pkl")
                with open(weights_file, 'rb') as f:
                    layer.weights.elements = pickle.load(f)
                if hasattr(layer, 'bias'):
                    bias_file = os.path.join(folder_path, f"{layer_id}_bias.pkl")
                    with open(bias_file, 'rb') as f:
                        layer.bias.elements = pickle.load(f)