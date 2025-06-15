"""
Comprehensive tests for CNN layers to validate Exercise 2 implementation.

This test suite validates the Conv2D and Pooling2D layers used in the CNN training
and ensures they work correctly for MNIST digit classification.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.conv_layers import Conv2DLayer, Pooling2D
from core.tensor import Tensor, Shape
from core.activations import ReLULayer, SoftmaxLayer
from core.layers import FullyConnectedLayer
from core.flatten import FlattenLayer
from core import Network, InputLayer
from utils.data_loader import load_mnist


class TestConv2DLayer(unittest.TestCase):
    """Test Conv2D layer implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.conv_layer = Conv2DLayer(
            in_channels=1, 
            out_channels=2, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        
    def test_initialization(self):
        """Test Conv2D layer initialization."""
        # Check layer type
        self.assertEqual(self.conv_layer.layer_type, "Conv2DLayer")
        
        # Check dimensions
        self.assertEqual(self.conv_layer.in_channels, 1)
        self.assertEqual(self.conv_layer.out_channels, 2)
        self.assertEqual(self.conv_layer.kernel_size, (3, 3))
        self.assertEqual(self.conv_layer.stride, (1, 1))
        self.assertEqual(self.conv_layer.padding, (1, 1))
        
        # Check weight initialization
        self.assertEqual(self.conv_layer.weights.elements.shape, (2, 1, 3, 3))
        self.assertTrue(self.conv_layer.use_bias)
        self.assertEqual(self.conv_layer.bias.elements.shape, (2,))
        
        # Check He initialization (weights should have reasonable variance)
        weight_std = np.std(self.conv_layer.weights.elements)
        expected_std = np.sqrt(2.0 / (1 * 3 * 3))  # He initialization
        self.assertAlmostEqual(weight_std, expected_std, delta=0.5)
        
    def test_forward_pass(self):
        """Test Conv2D forward pass with known input."""
        # Create test input (batch_size=2, channels=1, height=4, width=4)
        input_data = np.random.randn(2, 1, 4, 4)
        input_tensor = Tensor(input_data, shape=Shape(*input_data.shape))
        
        # Create output tensor
        output_tensor = Tensor(elements=np.zeros((2, 2, 4, 4)))
        
        # Forward pass
        self.conv_layer.forward([input_tensor], [output_tensor])
        
        # Check output shape
        self.assertEqual(output_tensor.elements.shape, (2, 2, 4, 4))
        
        # Check that output is not all zeros (layer should compute something)
        self.assertGreater(np.abs(output_tensor.elements).sum(), 0.01)
        
    def test_backward_pass(self):
        """Test Conv2D backward pass and gradient computation."""
        # Create test input
        input_data = np.random.randn(2, 1, 4, 4)
        input_tensor = Tensor(input_data, shape=Shape(*input_data.shape))
        
        # Forward pass
        output_tensor = Tensor(elements=np.zeros((2, 2, 4, 4)))
        self.conv_layer.forward([input_tensor], [output_tensor])
        
        # Create output gradients
        output_grad = np.random.randn(2, 2, 4, 4)
        output_tensor.deltas = output_grad
        
        # Backward pass
        self.conv_layer.backward([output_tensor], [input_tensor])
        
        # Check that gradients were computed
        self.assertIsNotNone(self.conv_layer.weights.deltas)
        self.assertIsNotNone(self.conv_layer.bias.deltas)
        self.assertIsNotNone(input_tensor.deltas)
        
        # Check gradient shapes
        self.assertEqual(self.conv_layer.weights.deltas.shape, (2, 1, 3, 3))
        self.assertEqual(self.conv_layer.bias.deltas.shape, (2,))
        self.assertEqual(input_tensor.deltas.shape, (2, 1, 4, 4))
        
        # Check that gradients are not all zeros
        self.assertGreater(np.abs(self.conv_layer.weights.deltas).sum(), 0.01)
        self.assertGreater(np.abs(self.conv_layer.bias.deltas).sum(), 0.01)
        
    def test_mnist_compatible_shapes(self):
        """Test Conv2D with MNIST-like input shapes."""
        # MNIST-like input (1 channel, 28x28)
        conv_mnist = Conv2DLayer(
            in_channels=1, 
            out_channels=16, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        
        # Test input batch
        batch_size = 4
        input_data = np.random.randn(batch_size, 1, 28, 28)
        input_tensor = Tensor(input_data, shape=Shape(*input_data.shape))
        
        # Forward pass
        output_tensor = Tensor(elements=np.zeros((batch_size, 16, 28, 28)))
        conv_mnist.forward([input_tensor], [output_tensor])
        
        # Check output shape (should preserve spatial dimensions with padding=1)
        self.assertEqual(output_tensor.elements.shape, (batch_size, 16, 28, 28))


class TestPooling2D(unittest.TestCase):
    """Test Pooling2D layer implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pool_layer = Pooling2D(pool_size=2, stride=2, mode='max')
        
    def test_initialization(self):
        """Test Pooling2D layer initialization."""
        # Check layer type
        self.assertEqual(self.pool_layer.layer_type, "Pooling2D")
        
        # Check parameters
        self.assertEqual(self.pool_layer.pool_size, (2, 2))
        self.assertEqual(self.pool_layer.stride, (2, 2))
        self.assertEqual(self.pool_layer.mode, 'max')
        
    def test_max_pooling_forward(self):
        """Test max pooling forward pass."""
        # Create test input (batch_size=2, channels=2, height=4, width=4)
        input_data = np.random.randn(2, 2, 4, 4)
        input_tensor = Tensor(input_data, shape=Shape(*input_data.shape))
        
        # Create output tensor
        output_tensor = Tensor(elements=np.zeros((2, 2, 2, 2)))
        
        # Forward pass
        self.pool_layer.forward([input_tensor], [output_tensor])
        
        # Check output shape (2x2 pooling should halve spatial dimensions)
        self.assertEqual(output_tensor.elements.shape, (2, 2, 2, 2))
        
        # For max pooling, output values should be <= max of input
        self.assertLessEqual(output_tensor.elements.max(), input_data.max() + 1e-6)
        
    def test_avg_pooling_forward(self):
        """Test average pooling forward pass."""
        pool_avg = Pooling2D(pool_size=2, stride=2, mode='avg')
        
        # Create test input with known values
        input_data = np.ones((1, 1, 4, 4)) * 2.0
        input_tensor = Tensor(input_data, shape=Shape(*input_data.shape))
        
        # Forward pass
        output_tensor = Tensor(elements=np.zeros((1, 1, 2, 2)))
        pool_avg.forward([input_tensor], [output_tensor])
        
        # For average pooling of all 2.0 values, output should be 2.0
        expected_output = np.ones((1, 1, 2, 2)) * 2.0
        np.testing.assert_array_almost_equal(output_tensor.elements, expected_output)
        
    def test_mnist_compatible_pooling(self):
        """Test pooling with MNIST-like shapes."""
        # After first conv: 16 channels, 28x28
        input_data = np.random.randn(4, 16, 28, 28)
        input_tensor = Tensor(input_data, shape=Shape(*input_data.shape))
        
        # Forward pass
        output_tensor = Tensor(elements=np.zeros((4, 16, 14, 14)))
        self.pool_layer.forward([input_tensor], [output_tensor])
        
        # Check output shape (should halve spatial dimensions)
        self.assertEqual(output_tensor.elements.shape, (4, 16, 14, 14))


class TestCNNIntegration(unittest.TestCase):
    """Test integration of CNN layers in a complete network."""
    
    def test_simple_cnn_forward(self):
        """Test forward pass through a simple CNN."""
        # Create a simple CNN: Conv -> ReLU -> Pool -> Flatten -> FC -> Softmax
        model = Network(learning_rate=0.01)
        
        model.add_layer(InputLayer())
        model.add_layer(Conv2DLayer(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1))
        model.add_layer(ReLULayer())
        model.add_layer(Pooling2D(pool_size=2, stride=2, mode='max'))
        model.add_layer(FlattenLayer())
        model.add_layer(FullyConnectedLayer(in_shape=Shape(4*14*14), out_shape=Shape(10)))
        model.add_layer(SoftmaxLayer())
        
        # Test input (MNIST-like)
        batch_size = 2
        input_data = np.random.randn(batch_size, 1, 28, 28) * 0.1
        input_tensor = Tensor(input_data, shape=Shape(*input_data.shape))
        
        # Forward pass
        output = model.forward(input_tensor)
        
        # Check output shape and properties
        self.assertEqual(output.elements.shape, (batch_size, 10))
        
        # Check softmax properties (probabilities should sum to 1)
        output_sums = np.sum(output.elements, axis=1)
        np.testing.assert_array_almost_equal(output_sums, np.ones(batch_size), decimal=5)
        
        # Check that all outputs are positive (softmax property)
        self.assertTrue(np.all(output.elements >= 0))
        
    def test_cnn_training_step(self):
        """Test that CNN can perform a training step without errors."""
        # Create simple CNN
        model = Network(learning_rate=0.1)
        
        model.add_layer(InputLayer())
        model.add_layer(Conv2DLayer(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1))
        model.add_layer(ReLULayer())
        model.add_layer(Pooling2D(pool_size=2, stride=2, mode='max'))
        model.add_layer(FlattenLayer())
        model.add_layer(FullyConnectedLayer(in_shape=Shape(2*14*14), out_shape=Shape(3)))
        model.add_layer(SoftmaxLayer())
        
        # Test data
        batch_size = 2
        input_data = np.random.randn(batch_size, 1, 28, 28) * 0.1
        target_data = np.eye(3)[np.random.randint(0, 3, batch_size)]  # One-hot
        
        input_tensor = Tensor(input_data, shape=Shape(*input_data.shape))
        target_tensor = Tensor(target_data, shape=Shape(*target_data.shape))
        
        # Save initial weights
        initial_conv_weights = model.layers[1].weights.elements.copy()
        initial_fc_weights = model.layers[5].weights.elements.copy()
        
        # Training step
        loss = model.train_step(input_tensor, target_tensor)
        
        # Check that loss is reasonable
        self.assertIsInstance(loss, (int, float))
        self.assertGreater(loss, 0)  # Cross-entropy loss should be positive
        self.assertLess(loss, 10)    # Should not be too large for random data
        
        # Check that weights were updated
        conv_weight_change = np.abs(model.layers[1].weights.elements - initial_conv_weights).mean()
        fc_weight_change = np.abs(model.layers[5].weights.elements - initial_fc_weights).mean()
        
        self.assertGreater(conv_weight_change, 1e-8, "Conv weights should change during training")
        self.assertGreater(fc_weight_change, 1e-8, "FC weights should change during training")


class TestMNISTIntegration(unittest.TestCase):
    """Test CNN with actual MNIST data."""
    
    def test_mnist_data_compatibility(self):
        """Test that CNN can process actual MNIST data."""
        # Load a small sample of MNIST data
        (X_train, y_train), (X_test, y_test) = load_mnist(flatten=False)
        
        # Take small sample for testing
        X_sample = X_train[:2].astype(np.float32)
        y_sample = y_train[:2]
        
        # Create simple CNN
        model = Network(learning_rate=0.01)
        model.add_layer(InputLayer())
        model.add_layer(Conv2DLayer(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1))
        model.add_layer(ReLULayer())
        model.add_layer(Pooling2D(pool_size=2, stride=2, mode='max'))
        model.add_layer(FlattenLayer())
        model.add_layer(FullyConnectedLayer(in_shape=Shape(4*14*14), out_shape=Shape(10)))
        model.add_layer(SoftmaxLayer())
        
        # Test forward pass
        input_tensor = Tensor(X_sample, shape=Shape(*X_sample.shape))
        output = model.forward(input_tensor)
        
        # Check output
        self.assertEqual(output.elements.shape, (2, 10))
        self.assertTrue(np.all(output.elements >= 0))
        self.assertTrue(np.all(output.elements <= 1))
        
        # Check that predictions sum to 1 (softmax property)
        output_sums = np.sum(output.elements, axis=1)
        np.testing.assert_array_almost_equal(output_sums, np.ones(2), decimal=5)


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)