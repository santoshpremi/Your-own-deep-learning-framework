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
            
            # Prepare input tensor list
            in_tensors = [current_tensor]
            
            # Create output tensor with appropriate shape
            if hasattr(layer, 'out_shape'):
                batch_size = current_tensor.elements.shape[0] if hasattr(current_tensor, 'elements') else len(current_tensor)
                if isinstance(layer.out_shape, Shape):
                    out_shape = Shape(batch_size, *layer.out_shape.dimensions)
                else:
                    out_shape = Shape(batch_size, layer.out_shape)
            else:
                out_shape = current_tensor.shape if hasattr(current_tensor, 'shape') else Shape(len(current_tensor))
            
            # Create output tensor with zeros
            out_tensor = Tensor(elements=np.zeros(out_shape.dimensions), shape=out_shape)
            
            # Call layer's forward method with both input and output tensors
            layer.forward(in_tensors, [out_tensor])
            current_tensor = out_tensor
        
        self.tensors.append(current_tensor)
        return current_tensor

    def backward(self, grad: Tensor):
        """
        Perform backward pass (backpropagation) through all layers.
        Args:
            grad: Gradient from the loss function (as a Tensor object)
        """
        try:
            # Initialize gradients for the output layer
            if not hasattr(self.tensors[-1], 'deltas') or self.tensors[-1].deltas is None:
                self.tensors[-1].deltas = np.zeros_like(self.tensors[-1].elements)
            self.tensors[-1].deltas = grad.elements

            # Backward pass through layers in reverse order
            for i in range(len(self.layers) - 1, -1, -1):
                layer = self.layers[i]
                if i + 1 >= len(self.tensors):
                    continue
                    
                out_tensor = self.tensors[i + 1]
                in_tensor = self.tensors[i] if i < len(self.tensors) else None
                
                # Initialize deltas if they don't exist
                if not hasattr(out_tensor, 'deltas') or out_tensor.deltas is None:
                    out_tensor.deltas = np.zeros_like(out_tensor.elements)
                    
                if in_tensor is not None:
                    if not hasattr(in_tensor, 'deltas') or in_tensor.deltas is None:
                        in_tensor.deltas = np.zeros_like(in_tensor.elements)
                
                # Skip input layer
                if hasattr(layer, 'layer_type') and layer.layer_type == "InputLayer":
                    continue
                    
                # Perform backward pass for the layer
                if hasattr(layer, 'backward'):
                    layer.backward([out_tensor], [in_tensor] if in_tensor is not None else [])
                    
                # Calculate weight gradients immediately after backward pass
                if hasattr(layer, 'calculate_delta_weights') and in_tensor is not None:
                    layer.calculate_delta_weights([out_tensor], [in_tensor])
                    
        except Exception as e:
            print(f"Error in backward pass: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def update_weights(self):
        """Update weights using gradient descent"""
        for i, layer in enumerate(self.layers):
            # Skip layers without weights or with None weights
            if not hasattr(layer, 'weights') or layer.weights is None:
                continue
                
            # Skip if weights.deltas is None
            if not hasattr(layer.weights, 'deltas') or layer.weights.deltas is None:
                continue
                
            # Only update if deltas is not None
            if layer.weights.deltas is not None:
                layer.weights.elements -= self.learning_rate * layer.weights.deltas
                layer.weights.deltas.fill(0)
            
            # Update bias if it exists and has deltas
            if hasattr(layer, 'bias') and layer.bias is not None and hasattr(layer.bias, 'deltas') and layer.bias.deltas is not None:
                layer.bias.elements -= self.learning_rate * layer.bias.deltas
                layer.bias.deltas.fill(0)

    def calculate_loss(self, output: Tensor, target: Tensor) -> float:
        """Calculate cross entropy loss"""
        epsilon = 1e-15
        predictions = np.clip(output.elements, epsilon, 1 - epsilon)
        
        # Ensure predictions and targets have compatible shapes
        if predictions.shape != target.elements.shape:
            if predictions.ndim > target.elements.ndim and predictions.shape[2] == 1:
                # Handle case where predictions have extra dimensions
                predictions = predictions.squeeze(2)
            
            if predictions.shape != target.elements.shape:
                raise ValueError(f"Shape mismatch in loss calculation. Predictions: {predictions.shape}, Targets: {target.elements.shape}")
        
        N = target.elements.shape[0]
        ce_loss = -np.sum(target.elements * np.log(predictions)) / N
        return ce_loss

    def calculate_loss_gradient(self, output: Tensor, target: Tensor) -> Tensor:
        """Calculate gradient of cross entropy loss with respect to output"""
        grad = output.elements - target.elements
        return Tensor(elements=grad)

    def train_step(self, x: Tensor, y: Tensor):
        """Perform one training step (forward pass, backward pass, parameter update)"""
        try:
            # Forward pass
            output = self.forward(x)
            
            # Calculate loss
            loss = self.calculate_loss(output, y)
            
            # Calculate gradient of loss
            grad = self.calculate_loss_gradient(output, y)
            
            # Initialize gradients for all tensors
            for tensor in self.tensors:
                if not hasattr(tensor, 'deltas'):
                    tensor.deltas = np.zeros_like(tensor.elements)
            
            # Backward pass
            self.backward(grad)
            
            # Update weights
            self.update_weights()
            
            return loss
            
        except Exception as e:
            print(f"Error in train_step: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

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