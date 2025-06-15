# Exercise 1: Regular Neural Network (Multi-Layer Perceptron)

- **Last Commit**: `final complete optimized ex-01`

## Requirements

- Python 3.8+
- NumPy
- scikit-learn (for data loading and preprocessing)
- tqdm (for progress bars)

##  Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/santoshpremi/Your-own-deep-learning-framework.git
   cd Your-own-deep-learning-framework
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**:
   ```bash
   python3 train/train_mnist.py
   ```

4. **Run unit tests**:
   ```bash
   python3 unittest_main.py
   ```



##  Network Architecture

The default network architecture for MNIST classification:

1. Input Layer (784 neurons)
2. Fully Connected Layer (512 neurons)
3. Batch Normalization (optional)
4. ReLU/Sigmoid Activation
5. Fully Connected Layer (10 neurons)
6. Softmax Activation

##  Configuration

Customize training in `train_mnist.py` with these parameters:

```python
network, history = train_mnist(
    epochs=30,               # Number of training epochs
    batch_size=256,          # Batch size
    learning_rate=0.01,      # Initial learning rate
    activation="relu",       # Activation function: 'relu' or 'sigmoid'
    use_batch_norm=True,    # Enable/disable batch normalization
    results_dir="results",   # Directory to save results
    checkpoint_freq=5        # Save checkpoint every N epochs
)
```

## Training Results

Training progress is saved in `mnist_results/`:
- `training_results.txt`: Detailed metrics for each epoch
- `weights/`: Saved model checkpoints
  - `best_model.pkl`: Best performing model
  - `epoch_XXX.pkl`: Checkpoints at specified intervals

## Testing Results
   test_results.txt

## Notes

- The framework achieves ~97% test accuracy on MNIST with default settings
- Training progress is logged to both console and file
- Early stopping is implemented to prevent overfitting
- Learning rate decay is applied at specific epochs


### Saving and Loading Weights
- Weights and biases are saved in the `mnist_results/weights/` directory.
- The best performing model is saved as `best_model.pkl` in the weights directory.
- Model checkpoints are saved every N epochs (configurable via `checkpoint_freq` parameter).

### Results
- Training results, including epoch number, runtime, loss, training accuracy, and test accuracy, are saved in `mnist_results/training_results.txt`.
- The results file is formatted as a tab-separated file for easy analysis.

## Example of Training Output:

| Epoch | Runtime(s) | Loss     | Train Accuracy | Test Accuracy |
|-------|------------|----------|----------------|---------------|
| 1     | 1.99       | 26.0067  | 0.9570         | 0.9390        |
| 2     | 1.87       | 11.5466  | 0.9750         | 0.9600        |
| 3     | 1.94       | 8.4959   | 0.9810         | 0.9620        |
| 4     | 1.92       | 6.5702   | 0.9850         | 0.9630        |
| 5     | 1.92       | 5.3774   | 0.9910         | 0.9690        |
| 6     | 2.01       | 4.4345   | 0.9860         | 0.9650        |
| 7     | 1.89       | 3.7090   | 0.9930         | 0.9680        |
| 8     | 2.22       | 3.1280   | 0.9950         | 0.9700        |
| 9     | 2.21       | 2.6291   | 0.9960         | 0.9660        |
| 10    | 1.90       | 2.2387   | 0.9980         | 0.9710        |

---






# Exercise 2: Convolutional Neural Network (CNN)

- **Last Commit**: `final complete optimized ex-02`

##  Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/santoshpremi/Your-own-deep-learning-framework.git
   cd Your-own-deep-learning-framework
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Training

Train the CNN using:

```bash
python3 cnn_main.py
```

Or directly:

```bash
python3 train/train_mnist_cnn.py
```

4. **Run unit tests**:
   ```bash
   python3 unittest_cnn_main.py
   ```


## Network Architecture

The CNN architecture for MNIST classification:

1. **Input Layer** (1×28×28 images)
2. **Conv2D Layer 1** (32 filters, 3×3 kernel, ReLU)
3. **MaxPooling2D** (2×2 kernel, stride=2) → 32×13×13
4. **Conv2D Layer 2** (64 filters, 3×3 kernel, ReLU)
5. **MaxPooling2D** (2×2 kernel, stride=2) → 64×5×5
6. **Flatten Layer** → 1600 features
7. **Fully Connected Layer** (128 neurons, ReLU)
8. **Fully Connected Output** (10 neurons, Softmax)

## Configuration

Customize CNN training in `train_mnist_cnn.py` with these parameters:

```python
model, history = train_cnn(
    epochs=15,               # Number of training epochs
    batch_size=64,           # Batch size
    learning_rate=0.001,     # Initial learning rate
    results_dir="mnist_cnn_results",  # Directory to save results
    checkpoint_freq=5        # Save checkpoint every N epochs
)
```

## Training Results

Training progress is saved in `mnist_cnn_results/`:
- `training_cnn_results.txt`: Detailed metrics for each epoch
- `weights/`: Saved model checkpoints
  - `best_model.pkl`: Best performing model
  - `epoch_XXX.pkl`: Checkpoints at specified intervals

### Saving and Loading Weights
- Weights and biases are saved in the `mnist_cnn_results/weights/` directory.
- The best performing model is saved as `best_model.pkl` in the weights directory.
- Model checkpoints are saved every N epochs (configurable via `checkpoint_freq` parameter).

### Results
- Training results, including epoch number, runtime, loss, training accuracy, and test accuracy, are saved in `mnist_cnn_results/training_cnn_results.txt`.
- The results file is formatted as a tab-separated file for easy analysis.

## Testing

```bash
python3 unittest_cnn_main.py
```

### Testing Results
- `test_cnn_results.txt`: CNN model validation and performance analysis
  - 11 tests run, 11 passed, 0 failed
  - Tests for Conv2D, MaxPool2D, Flatten layers, and CNN integration

## Example of Training Output:

| Epoch | Runtime(s) | Loss     | Train Accuracy | Test Accuracy |
|-------|------------|----------|----------------|---------------|
| 1     | 45.2       | 4.3471   | 0.6680         | 0.6520        |
| 2     | 43.1       | 2.1250   | 0.8640         | 0.8590        |
| 3     | 42.8       | 1.5430   | 0.9120         | 0.9080        |
| 4     | 42.5       | 1.2140   | 0.9350         | 0.9280        |
| 5     | 42.3       | 0.9850   | 0.9480         | 0.9420        |
| 6     | 42.1       | 0.8200   | 0.9570         | 0.9510        |
| 7     | 41.9       | 0.6950   | 0.9640         | 0.9580        |
| 8     | 41.8       | 0.5980   | 0.9710         | 0.9640        |
| 9     | 41.6       | 0.5240   | 0.9760         | 0.9690        |
| 10    | 41.4       | 0.4650   | 0.9800         | 0.9720        |

---

