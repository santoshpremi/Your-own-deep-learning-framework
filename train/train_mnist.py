"""MNIST Digit Classification Training Script.

This script trains a configurable neural network on the MNIST dataset.
It supports various hyperparameters and includes features like early stopping,
learning rate decay, and model checkpointing.
"""

import os
import time
from typing import Dict, List, Tuple, Optional
import numpy as np

# Core framework imports
from core import (
    Network, Tensor, Shape,
    InputLayer, FullyConnectedLayer,
    ReLULayer, SigmoidLayer, SoftmaxLayer,
    BatchNormLayer
)
from utils.data_loader import load_mnist

def create_network(
    learning_rate: float = 0.01,
    hidden_size: int = 512,
    activation: str = "relu",
    use_batch_norm: bool = True,
) -> Network:
    """Build a configurable MLP for MNIST classification.

    Architecture: Input(784) -> FC(hidden_size) -> [BatchNorm] -> Activation -> FC(10) -> Softmax

    Args:
        learning_rate: Initial learning rate for SGD optimizer.
        hidden_size: Number of neurons in the hidden layer.
        activation: Activation function to use ("relu" or "sigmoid").
        use_batch_norm: Whether to use batch normalization after the first FC layer.

    Returns:
        Configured neural network ready for training.
    """
    network = Network(learning_rate=learning_rate)

    def _get_activation():
        """Helper to get the specified activation layer."""
        if activation.lower() == "relu":
            return ReLULayer()
        elif activation.lower() == "sigmoid":
            return SigmoidLayer()
        raise ValueError(f"Unsupported activation: {activation}. Use 'relu' or 'sigmoid'.")

    # Input layer
    network.add_layer(InputLayer())

    # First fully connected layer with optional batch norm
    fc1 = FullyConnectedLayer(Shape(784), Shape(hidden_size))
    
    # Initialize weights using He/Xavier initialization
    init_scale = np.sqrt(2.0 / 784) if activation == "relu" else np.sqrt(1.0 / 784)
    fc1.weights.elements = np.random.randn(784, hidden_size) * init_scale
    network.add_layer(fc1)

    # Optional batch normalization
    if use_batch_norm:
        network.add_layer(BatchNormLayer(hidden_size))
    
    # Activation
    network.add_layer(_get_activation())


    # Output layer
    output_scale = np.sqrt(2.0 / hidden_size) if activation == "relu" else np.sqrt(1.0 / hidden_size)
    output_layer = FullyConnectedLayer(Shape(hidden_size), Shape(10))
    output_layer.weights.elements = np.random.randn(hidden_size, 10) * output_scale
    network.add_layer(output_layer)
    
    # Softmax for classification
    network.add_layer(SoftmaxLayer())
    
    return network

def calculate_accuracy(network: Network, x: np.ndarray, y: np.ndarray) -> float:
    """Calculate classification accuracy for the given input-output pairs.
    
    Args:
        network: Trained neural network.
        x: Input data of shape (num_samples, 784).
        y: One-hot encoded true labels of shape (num_samples, 10).
        
    Returns:
        Accuracy as a float between 0 and 1.
    """
    # Forward pass to get predictions
    predictions = network.forward(Tensor(elements=x))
    
    # Convert predictions and true labels to class indices
    predicted_labels = np.argmax(predictions.elements, axis=1)
    true_labels = np.argmax(y, axis=1)
    
    # Calculate and return accuracy
    return np.mean(predicted_labels == true_labels)

def train_mnist(
    epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 0.01,
    activation: str = "relu",
    use_batch_norm: bool = True,
    results_dir: str = "mnist_results",
    checkpoint_freq: int = 5,
) -> Tuple[Network, List[Dict]]:
    """Train a neural network on the MNIST dataset.
    
    Args:
        epochs: Maximum number of training epochs.
        batch_size: Number of samples per batch.
        learning_rate: Initial learning rate for SGD.
        activation: Activation function to use ("relu" or "sigmoid").
        use_batch_norm: Whether to use batch normalization.
        results_dir: Directory to save training results and checkpoints.
        checkpoint_freq: Save model checkpoint every N epochs.
        
    Returns:
        Tuple of (trained_network, training_history)
    """
    # Setup results directory
    weights_dir = os.path.join(results_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    # Load dataset
    print("Loading MNIST dataset...")
    X_train, X_test, y_train, y_test = load_mnist()
    num_train = X_train.shape[0]
    
    # Initialize network
    network = create_network(
        learning_rate=learning_rate,
        activation=activation,
        use_batch_norm=use_batch_norm,
    )
    
    # Training state
    best_accuracy = 0.0
    patience = 5  # Early stopping patience
    patience_counter = 0
    min_improvement = 0.001
    results = []
    
    print("\nStarting training...")
    print(f"Training samples: {num_train}, Batch size: {batch_size}, Batches/epoch: {num_train // batch_size}")
    
    # Main training loop
    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0.0
        
        # Shuffle training data
        perm = np.random.permutation(num_train)
        
        # Learning rate decay
        if epoch in (10, 15, 20):
            network.learning_rate *= 0.5
            print(f"\nDecayed learning rate to {network.learning_rate:.5f}")
        
        # Process batches
        num_batches = num_train // batch_size
        for batch_idx in range(num_batches):
            # Get batch
            start = batch_idx * batch_size
            end = start + batch_size
            batch_indices = perm[start:end]
            
            batch_X = X_train[batch_indices]
            batch_y = y_train[batch_indices]
            
            # Create tensors
            x_tensor = Tensor(elements=batch_X, shape=Shape(len(batch_indices), 784))
            y_tensor = Tensor(elements=batch_y, shape=Shape(len(batch_indices), 10))
            
            # Training step
            loss = network.train_step(x_tensor, y_tensor)
            total_loss += loss
            
            # Log progress
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == num_batches:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{num_batches}, Loss: {loss:.4f}")
        
        # Epoch evaluation
        train_accuracy = calculate_accuracy(network, X_train, y_train)
        test_accuracy = calculate_accuracy(network, X_test, y_test)
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / num_batches
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{epochs} completed in {epoch_time:.1f}s")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Train Accuracy: {train_accuracy:.4f}")
        print(f"  Test Accuracy:  {test_accuracy:.4f}")
        
        # Checkpoint model
        if (epoch + 1) % checkpoint_freq == 0:
            checkpoint_path = os.path.join(weights_dir, f"epoch_{epoch+1:03d}.pkl")
            network.save_params(checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")
        
        # Early stopping check
        if test_accuracy > best_accuracy + min_improvement:
            best_accuracy = test_accuracy
            patience_counter = 0
            best_model_path = os.path.join(weights_dir, "best_model.pkl")
            network.save_params(best_model_path)
            print(f"  New best model saved with accuracy: {best_accuracy:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping - No improvement for {patience} epochs")
                break
        
        # Record results
        results.append({
            'epoch': epoch + 1,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'loss': avg_loss,
            'time': epoch_time,
            'learning_rate': network.learning_rate,
        })
    
    # Save final results
    _save_training_results(results, results_dir)
    print(f"\nTraining completed! Best test accuracy: {best_accuracy:.4f}")
    
    return network, results


def _save_training_results(results: List[Dict], results_dir: str) -> None:
    """Save training results to a text file."""
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "training_results.txt")
    
    with open(results_file, 'w') as f:
        # Write header
        f.write("Epoch\tTrain Acc\tTest Acc\tLoss\tTime (s)\tLR\n")
        
        # Write results
        for result in results:
            f.write(
                f"{result['epoch']}\t"
                f"{result['train_accuracy']:.4f}\t"
                f"{result['test_accuracy']:.4f}\t"
                f"{result['loss']:.4f}\t"
                f"{result['time']:.1f}\t"
                f"{result['learning_rate']:.6f}\n"
            )
    
    print(f"Training results saved to {results_file}")
