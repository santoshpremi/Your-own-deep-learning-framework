from typing import List
import numpy as np
import os
import pickle
from core.tensor import Tensor
from core.layers import Layer

class Network:
    def __init__(self, layers: List[Layer]=None, loss_fn=None, activation_fn=None, learning_rate=0.01):
        """Initialize network with layers, loss function, activation and learning rate"""
        self.layers = layers if layers is not None else []
        self.loss_fn = loss_fn
        self.activation_fn = activation_fn
        self.learning_rate = learning_rate
        # Add unique identifiers to each layer for saving/loading
        for i, layer in enumerate(self.layers):
            layer.layer_type = layer.__class__.__name__
            layer.num = i

    def add_layer(self, layer):
        """Add a layer to the network and assign it a unique number"""
        layer.num = len(self.layers)  # Assign the next available number
        self.layers.append(layer)

    def forward(self, x):
        """
        Perform forward pass through all layers.
        Args:
            x: Input data
        Returns:
            Tensor: Network output
        """
        current = x
        for layer in self.layers:
            current = layer.forward(current)
        return current

    def backward(self, grad):
        """
        Perform backward pass (backpropagation) through all layers.
        Args:
            grad: Gradient from the loss function
        """
        current_grad = grad
        for layer in reversed(self.layers):
            current_grad = layer.backward(current_grad)

    def update_weights(self):
        """
        Update network weights using Stochastic Gradient Descent (SGD).
        Updates both weights and biases for all layers that have them.
        """
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                layer.weights.elements -= self.learning_rate * layer.weights.deltas
                # Update biases using SGD
                layer.biases.elements -= self.learning_rate * layer.biases.deltas 
            layer.calc_delta_weights()

    def compute_loss(self, output, target):
        """
        Compute the loss between network output and target.
        Supports both cross-entropy (for classification) and MSE (for regression).
        Args:
            output: Network output tensor
            target: Target values tensor
        Returns:
            float: Computed loss value
        """
        if isinstance(target.elements, np.ndarray) and target.elements.ndim > 1:
            # Cross entropy for classification tasks
            epsilon = 1e-15  # Small value to avoid log(0)
            output_clipped = np.clip(output.elements, epsilon, 1 - epsilon)
            return -np.sum(target.elements * np.log(output_clipped))
        else:
            # Mean Squared Error for regression tasks
            return np.mean((np.array(output.elements) - np.array(target.elements)) ** 2)

    def compute_loss_gradient(self, output, target):
        """
        Compute the gradient of the loss function.
        Args:
            output: Network output tensor
            target: Target values tensor
        Returns:
            numpy.ndarray: Gradient of the loss
        """
        if isinstance(target.elements, np.ndarray) and target.elements.ndim > 1:
            # Cross entropy gradient
            return -target.elements/output.elements
        else:
            # MSE gradient
            return 2 * (np.array(output.elements) - np.array(target.elements)) / len(output.elements)

    def save_params(self, folder_path):
        """Save network parameters to files"""
        os.makedirs(folder_path, exist_ok=True)
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                # Create unique identifier for the layer
                layer_id = f"{layer.layer_type}_{layer.num}"
                # Save weights
                weights_file = os.path.join(folder_path, f"{layer_id}_weights.pkl")
                with open(weights_file, 'wb') as f:
                    pickle.dump(layer.weights.elements, f)
                # Save biases
                biases_file = os.path.join(folder_path, f"{layer_id}_biases.pkl")
                with open(biases_file, 'wb') as f:
                    pickle.dump(layer.biases.elements, f)

    def load_params(self, folder_path):
        """Load network parameters from files"""
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                # Create unique identifier for the layer
                layer_id = f"{layer.layer_type}_{layer.num}"
                # Load weights
                weights_file = os.path.join(folder_path, f"{layer_id}_weights.pkl")
                with open(weights_file, 'rb') as f:
                    layer.weights.elements = pickle.load(f)
                # Load biases
                biases_file = os.path.join(folder_path, f"{layer_id}_biases.pkl")
                with open(biases_file, 'rb') as f:
                    layer.biases.elements = pickle.load(f)

    def train_step(self, x, y, load_existing=False, save_path=None):
        """
        Perform a single training step.
        Args:
            x: Input data
            y: Target values
            load_existing: Whether to load existing weights
            save_path: Path to save/load weights
        Returns:
            float: Loss value for this step
        """
        # Load existing weights if requested
        if load_existing and save_path:
            self.load_params(save_path)
            return None
        # Forward pass
        output = self.forward(x)
        # Compute loss and gradient
        loss = self.compute_loss(output, y)
        grad = self.compute_loss_gradient(output, y)
        # Backward pass
        self.backward(grad)
        # Update weights
        self.update_weights()
        # Save if path provided
        if save_path:
            self.save_params(save_path)
        return loss