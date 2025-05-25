import numpy as np
import time
import os
from core.network import Network
from core.layers import InputLayer, FullyConnectedLayer
from core.activations import SigmoidLayer, SoftmaxLayer
from utils.data_loader import load_mnist
from core.tensor import Tensor, Shape

def create_network():
    # Create network architecture
    network = Network(learning_rate=0.01)  # Reduced learning rate
    # Input layer (784 neurons for MNIST images)
    network.add_layer(InputLayer())
    # Hidden layer 1 (256 neurons)
    network.add_layer(FullyConnectedLayer(784, 256))
    network.add_layer(SigmoidLayer())
    # Hidden layer 2 (128 neurons)
    network.add_layer(FullyConnectedLayer(256, 128))
    network.add_layer(SigmoidLayer())
    # Output layer (10 neurons for digits 0-9)
    network.add_layer(FullyConnectedLayer(128, 10))
    network.add_layer(SoftmaxLayer())
    return network

def evaluate_accuracy(network, X, y, batch_size=1000):
    correct = 0
    total = len(X)
    # Process in batches for faster evaluation
    for i in range(0, total, batch_size):
        batch_X = X[i:i+batch_size]
        batch_y = y[i:i+batch_size]
        # Process each sample in the batch
        x_tensor = Tensor(elements=batch_X, shape=Shape(batch_size, 784))
        output = network.forward(x_tensor)
        predictions = np.argmax(output.elements, axis=1)
        actuals = np.argmax(batch_y, axis=1)
        correct += np.sum(predictions == actuals)
    return correct / total

def train_mnist(epochs=10, batch_size=64):  # Reduced batch size
    # Load data
    X_train, X_test, y_train, y_test = load_mnist()
    # Create network
    network = create_network()
    # Create results directory
    results_dir = "mnist_results"
    os.makedirs(results_dir, exist_ok=True)
    # Training loop
    print("\nStarting training...")
    results = []
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0
        num_batches = len(X_train) // batch_size
        # Shuffle training data
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train_shuffled[i:i+batch_size]
            batch_y = y_train_shuffled[i:i+batch_size]
            # Convert batch to tensors
            x_tensor = Tensor(elements=batch_X, deltas=None, shape=Shape(batch_size, 784))
            y_tensor = Tensor(elements=batch_y, deltas=None, shape=Shape(batch_size, 10))
            loss = network.train_step(x_tensor, y_tensor)
            epoch_loss += loss
        # Calculate average loss for epoch
        avg_loss = epoch_loss / num_batches
        # Calculate training and test accuracy (on smaller subset for speed)
        eval_size = min(1000, len(X_train))
        train_accuracy = evaluate_accuracy(network, X_train[:eval_size], y_train[:eval_size])
        test_accuracy = evaluate_accuracy(network, X_test[:eval_size], y_test[:eval_size])
        # Calculate epoch runtime
        epoch_time = time.time() - epoch_start
        # Store results
        results.append({
            'epoch': epoch + 1,
            'runtime': epoch_time,
            'loss': avg_loss,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy
        })
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Runtime: {epoch_time:.2f}s")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}\n")
    # Save network weights
    network.save_params(os.path.join(results_dir, "weights"))
    # Save results to file
    with open(os.path.join(results_dir, "training_results.txt"), "w") as f:
        f.write("Epoch\tRuntime(s)\tLoss\tTrain Accuracy\tTest Accuracy\n")
        for result in results:
            f.write(f"{result['epoch']}\t{result['runtime']:.2f}\t{result['loss']:.4f}\t{result['train_accuracy']:.4f}\t{result['test_accuracy']:.4f}\n")
    return results