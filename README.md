## Requirements

- Python 3.8+
- NumPy
- scikit-learn (for data loading and preprocessing)
- tqdm (for progress bars)

##  Quick Start

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd dl-framework-adhikari-maria/dl_Framework
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**:
   ```bash
   python3 train/train_mnist.py
   ```

4. **Run tests**:
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