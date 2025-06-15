"""
Exercise 2: Train a Convolutional Neural Network (CNN) on the MNIST dataset.

This script implements an optimized CNN with improved training techniques.
"""

import os
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any, Optional
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import framework components
from core import (
    Network, Tensor, Shape, InputLayer, Conv2DLayer, Pooling2DLayer,
    FullyConnectedLayer, ReLULayer, FlattenLayer, BatchNormLayer, DropoutLayer,
    SoftmaxLayer
)
from utils.data_loader import load_mnist

def create_cnn(input_shape: Tuple[int, int, int], num_classes: int = 10, learning_rate: float = 0.01) -> Network:
    """Create a CNN model with batch normalization and dropout.
    
    Args:
        input_shape: Shape of input images (channels, height, width)
        num_classes: Number of output classes
        learning_rate: Initial learning rate
        
    Returns:
        Compiled CNN model
    """
    model = Network(learning_rate=learning_rate)
    
    # Input layer
    model.add_layer(InputLayer())
    
    # First conv block: 1@28x28 -> 16@28x28 -> 16@14x14
    model.add_layer(Conv2DLayer(
        in_channels=input_shape[0],
        out_channels=16,
        kernel_size=3,
        stride=1,
        padding=1
    ))
    model.add_layer(ReLULayer())
    from core.conv_layers import Pooling2D
    model.add_layer(Pooling2D(pool_size=2, stride=2, mode='max'))
    
    # Second conv block: 16@14x14 -> 32@14x14 -> 32@7x7
    model.add_layer(Conv2DLayer(
        in_channels=16,
        out_channels=32,
        kernel_size=3,
        stride=1,
        padding=1
    ))
    model.add_layer(ReLULayer())
    model.add_layer(Pooling2D(pool_size=2, stride=2, mode='max'))
    
    # Classifier: 32*7*7=1568 -> 128 -> 10
    model.add_layer(FlattenLayer())
    
    model.add_layer(FullyConnectedLayer(
        in_shape=Shape(32*7*7),
        out_shape=Shape(128)
    ))
    model.add_layer(ReLULayer())
    
    # Output layer
    model.add_layer(FullyConnectedLayer(
        in_shape=Shape(128),
        out_shape=Shape(num_classes)
    ))
    model.add_layer(SoftmaxLayer())
    
    return model

def calculate_accuracy(model: Network, X: np.ndarray, y: np.ndarray, batch_size: int = 100) -> Tuple[float, float]:
    """
    Calculate model accuracy and loss on the given dataset.
    
    Args:
        model: The neural network model
        X: Input data
        y: One-hot encoded labels
        batch_size: Batch size for evaluation
        
    Returns:
        Tuple of (accuracy, average_loss)
    """
    correct = 0
    total = 0
    total_loss = 0.0
    
    for i in range(0, len(X), batch_size):
        batch_X = X[i:i+batch_size]
        batch_y = y[i:i+batch_size]
        
        if len(batch_X) == 0:
            continue
            
        # Forward pass
        X_tensor = Tensor(batch_X, shape=Shape(batch_X.shape[0], *batch_X.shape[1:]))
        y_tensor = Tensor(batch_y, shape=batch_y.shape)
        
        # Get predictions
        output = model.forward(X_tensor)
        loss = model.calculate_loss(output, y_tensor)
        
        # Get predictions and true labels
        preds = np.argmax(output.elements, axis=1)
        true_labels = np.argmax(batch_y, axis=1)
        
        # Update metrics
        correct += np.sum(preds == true_labels)
        total += len(batch_X)
        if hasattr(loss, 'elements'):
            total_loss += loss.elements * len(batch_X)
        else:
            total_loss += loss * len(batch_X)
    
    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else float('inf')
    
    return accuracy, avg_loss

def train_cnn():
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"mnist_cnn_results/mnist_cnn_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Training hyperparameters
    batch_size = 128
    num_epochs = 15
    initial_learning_rate = 0.01
    
    # Print configuration
    print("\nTraining Configuration:")
    print(f"- Batch size: {batch_size}")
    print(f"- Initial learning rate: {initial_learning_rate}")
    print(f"- Epochs: {num_epochs}")
    
    # Load and preprocess data
    print("\nLoading MNIST dataset...")
    (X_train, y_train_onehot), (X_test, y_test_onehot) = load_mnist(flatten=False)
    
    # Data is already normalized to [0, 1], just convert to float32
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    print(f"Data range after preprocessing: [{X_train.min():.4f}, {X_train.max():.4f}]")
    
    # Convert one-hot to class indices for accuracy calculation
    y_train = np.argmax(y_train_onehot, axis=1)
    y_test = np.argmax(y_test_onehot, axis=1)
    
    # Split into training and validation sets (80-20 split)
    val_size = len(X_train) // 5
    X_train, X_val = X_train[val_size:], X_train[:val_size]
    y_train_onehot, y_val_onehot = y_train_onehot[val_size:], y_train_onehot[:val_size]
    y_train, y_val = y_train[val_size:], y_train[:val_size]
    
    # Use 50% of training data for faster iteration
    train_size = len(X_train) // 2
    X_train = X_train[:train_size]
    y_train = y_train[:train_size]
    y_train_onehot = y_train_onehot[:train_size]
    
    print(f"\nDataset sizes:")
    print(f"- Training: {len(X_train)} samples")
    print(f"- Validation: {len(X_val)} samples")
    print(f"- Test: {len(X_test)} samples")
    
    # Learning rate schedule function
    def lr_schedule(epoch):
        """Learning rate schedule with warmup and decay."""
        if epoch < 3:  # Shorter warmup
            return initial_learning_rate * 0.5  # Start with half learning rate
        elif epoch < 10:  # Stable learning rate
            return initial_learning_rate
        else:  # Decay
            return initial_learning_rate * 0.5
    
    print("\nTraining Configuration:")
    print(f"- Batch size: {batch_size}")
    print(f"- Initial learning rate: {initial_learning_rate}")
    print(f"- Max epochs: {num_epochs}")
    
    # Create model
    print("\nCreating and compiling model...")
    model = create_cnn(input_shape=Shape(1, 28, 28), num_classes=10, learning_rate=initial_learning_rate)
    
    # Print model summary
    print("\nModel Architecture:")
    for i, layer in enumerate(model.layers):
        print(f"Layer {i+1}: {layer.__class__.__name__}")
    
    print(f"\nStarting training for {num_epochs} epochs...")
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': [],
        'epoch_times': []
    }
    
    best_val_acc = 0.0
    best_weights = None
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_start = time.time()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Update learning rate
        current_lr = lr_schedule(epoch)
        model.learning_rate = current_lr
        
        # Training phase
        pbar = tqdm(range(0, len(X_train), batch_size), 
                   desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for i in pbar:
            # Get batch
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train_onehot[i:i+batch_size]
            
            if len(batch_X) == 0:
                continue
            
            # Forward and backward pass
            loss = model.train_step(
                Tensor(batch_X, shape=Shape(batch_X.shape[0], *batch_X.shape[1:])),
                Tensor(batch_y, shape=batch_y.shape)
            )
            
            # Calculate metrics
            output = model.forward(Tensor(batch_X, shape=Shape(batch_X.shape[0], *batch_X.shape[1:])))
            preds = np.argmax(output.elements, axis=1)
            true_labels = np.argmax(batch_y, axis=1)
            
            # Update metrics
            if hasattr(loss, 'elements'):
                total_loss += loss.elements * len(batch_X)
            else:
                total_loss += loss * len(batch_X)
            correct += np.sum(preds == true_labels)
            total += len(batch_X)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss/total:.4f}",
                'acc': f"{correct/total:.2%}",
                'lr': f"{current_lr:.6f}"
            })
        
        # Calculate training metrics
        train_loss = total_loss / total
        train_acc = correct / total
        
        # Validation phase
        val_acc, val_loss = calculate_accuracy(model, X_val, y_val_onehot, batch_size=256)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Store best weights by copying elements
            best_weights = {}
            for i, layer in enumerate(model.layers):
                if hasattr(layer, 'weights') and layer.weights is not None:
                    best_weights[f'layer_{i}_weights'] = layer.weights.elements.copy()
                if hasattr(layer, 'bias') and layer.bias is not None:
                    best_weights[f'layer_{i}_bias'] = layer.bias.elements.copy()
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1:2d}/{num_epochs} - {epoch_time:.1f}s - "
              f"loss: {train_loss:.4f} - acc: {train_acc:.4f} - "
              f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f} - lr: {current_lr:.6f}")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)
        history['epoch_times'].append(epoch_time)
    
    # Calculate accuracy on test set using one-hot encoded labels
    test_acc, test_loss = calculate_accuracy(model, X_test, y_test_onehot, batch_size)
    print(f"\nTest accuracy: {test_acc:.4f} - Test loss: {test_loss:.4f}")
    
    # Save the final model
    model_path = os.path.join(results_dir, 'mnist_cnn_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save training results to text file
    _save_training_results(history, results_dir)
    
    # Also save to main results directory for easy access
    main_results_dir = "mnist_cnn_results"
    _save_training_results(history, main_results_dir)
    
    # Plot training history
    plot_training_history(history, results_dir)
    
    # Print final metrics
    print("\n" + "="*50)
    print(f"Training completed!")
    print(f"Final training accuracy: {train_acc:.4f}")
    print(f"Final validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print("="*50 + "\n")
    
    # Save model weights manually since save_weights method doesn't exist
    weights = {}
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'weights') and layer.weights is not None:
            weights[f'layer_{i}_weights'] = layer.weights.elements
        if hasattr(layer, 'bias') and layer.bias is not None:
            weights[f'layer_{i}_bias'] = layer.bias.elements
    
    np.savez_compressed(os.path.join(results_dir, "model_weights.npz"), **weights)



def plot_training_history(history: Dict[str, List[float]], results_dir: str) -> None:
    """Plot training and validation metrics."""
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Save figure
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training history plot saved to {plot_path}")

def _save_training_results(history: Dict[str, List[float]], results_dir: str) -> None:
    """Save training results to a text file."""
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "training_cnn_results.txt")
    
    with open(results_file, 'w') as f:
        # Write header
        f.write("Epoch\tTrain Acc\tTest Acc\tLoss\tTime (s)\tLR\n")
        
        # Write results for each epoch
        for i in range(len(history['train_acc'])):
            epoch = i + 1
            train_acc = history['train_acc'][i]
            val_acc = history['val_acc'][i]  # Using validation as "test" during training
            loss = history['train_loss'][i]
            lr = history['learning_rates'][i]
            epoch_time = history['epoch_times'][i] if 'epoch_times' in history else 50.0
            
            f.write(
                f"{epoch}\t"
                f"{train_acc:.4f}\t"
                f"{val_acc:.4f}\t"
                f"{loss:.4f}\t"
                f"{epoch_time:.1f}\t"
                f"{lr:.6f}\n"
            )
    
    print(f"Training results saved to {results_file}")


if __name__ == "__main__":
    train_cnn()
