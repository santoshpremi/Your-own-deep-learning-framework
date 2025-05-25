
## Requirements
- Python 3.8+
- NumPy
- scikit-learn

## Usage
1. Clone the repository:
    ```bash
    git clone <repository_url>
    cd deep_numpy_nn
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the training script:
    ```bash
    python3 main.py
    ```

## Observations
- The framework is designed to handle mini-batch training for efficiency.
- The network architecture consists of an input layer, two hidden layers with sigmoid activation, and an output layer with softmax activation.
- The training process runs for 10 epochs with a batch size of 64.
- The accuracy of the trained network on the test data is noted down for each epoch.

## Notes
- Ensure that the `saved_models` and `assets` directories are created as needed.
- The training results are saved in the `mnist_results` directory.

## Implementation Details

### Core Components
- **Tensor**: Represents the data elements, gradients, and shape.
- **Layers**: Includes input, fully connected, activation, and softmax layers.
- **Network**: Manages the forward and backward passes, weight updates, and training process.

### Training Process
- **Data Loading**: MNIST dataset is loaded and preprocessed (normalized and one-hot encoded).
- **Network Architecture**: Consists of:
  - Input Layer
  - Two Hidden Layers with Sigmoid Activation
  - Output Layer with Softmax Activation
- **Training Loop**: Runs for 10 epochs with a batch size of 64.
- **Evaluation**: Computes and prints training and test accuracy for each epoch.

### Saving and Loading Weights
- Weights and biases are saved in the `saved_models/weights` directory.
- Weights and biases can be loaded from the same directory to continue training or for inference.

### Results
- Training results, including epoch number, runtime, loss, training accuracy, and test accuracy, are saved in `mnist_results/training_results.txt`.

## Example Output
Epoch Runtime(s)	Loss	Train Accuracy	Test Accuracy
1	1.99	26.0067	            0.9570	            0.9390
2	1.87	11.5466	            0.9750	            0.9600
3	1.94	8.4959	            0.9810	            0.9620
4	1.92	6.5702	            0.9850	            0.9630
5	1.92	5.3774	            0.9910	            0.9690
6	2.01	4.4345	            0.9860	            0.9650
7	1.89	3.7090	            0.9930	            0.9680
8	2.22	3.1280	            0.9950	            0.9700
9	2.21	2.6291	            0.9960	            0.9660
10	1.90	2.2387	            0.9980	            0.9710