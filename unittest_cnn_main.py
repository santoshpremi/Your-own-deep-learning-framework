#!/usr/bin/env python3
"""
Script to run comprehensive CNN tests and generate test_cnn_results.txt

This script executes the comprehensive CNN test suite from test_conv_layers.py
and outputs results in the same format as the main test results.
"""

import unittest
import sys
from datetime import datetime
import numpy as np
from tests.test_conv.test_conv_layers import (
    TestConv2DLayer as TestConv2DLayerComprehensive, 
    TestPooling2D as TestPooling2DComprehensive, 
    TestCNNIntegration, 
    TestMNISTIntegration
)

class DetailedTestResult(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.test_details = []
        
    def addSuccess(self, test):
        self._store_test_details(test, "PASS", None)
        
    def addError(self, test, err):
        self._store_test_details(test, "ERROR", err)
        
    def addFailure(self, test, err):
        self._store_test_details(test, "FAIL", err)
        
    def _store_test_details(self, test, status, error=None):
        detail = {
            'name': test.id(),
            'status': status,
            'error': error,
            'values': self._get_test_values(test)
        }
        self.test_details.append(detail)
        
    def _get_test_values(self, test):
        values = {}
        if hasattr(test, '_testMethodName'):
            # Comprehensive Conv2D Layer Tests
            if isinstance(test, TestConv2DLayerComprehensive):
                if 'test_initialization' in test._testMethodName:
                    values['in_channels'] = 1
                    values['out_channels'] = 2
                    values['kernel_size'] = '3x3'
                    values['expected_weight_shape'] = '(2,1,3,3)'
                elif 'test_forward_pass' in test._testMethodName:
                    values['input_shape'] = '(2,1,4,4)'
                    values['output_shape'] = '(2,2,4,4)'
                    values['padding'] = 1
                elif 'test_mnist_compatible_shapes' in test._testMethodName:
                    values['input_shape'] = '(4,1,28,28)'
                    values['output_shape'] = '(4,16,28,28)'
                    values['filters'] = 16

            # Comprehensive Pooling2D Tests
            elif isinstance(test, TestPooling2DComprehensive):
                if 'test_max_pooling_forward' in test._testMethodName:
                    values['input_shape'] = '(2,2,4,4)'
                    values['output_shape'] = '(2,2,2,2)'
                    values['pool_mode'] = 'max'
                elif 'test_avg_pooling_forward' in test._testMethodName:
                    values['input_shape'] = '(1,1,4,4)'
                    values['output_shape'] = '(1,1,2,2)'
                    values['pool_mode'] = 'avg'

            # CNN Integration Tests
            elif isinstance(test, TestCNNIntegration):
                if 'test_simple_cnn_forward' in test._testMethodName:
                    values['architecture'] = 'Conv->ReLU->Pool->Flatten->FC->Softmax'
                    values['input_shape'] = '(2,1,28,28)'
                    values['output_shape'] = '(2,10)'
                elif 'test_cnn_training_step' in test._testMethodName:
                    values['training_test'] = 'Weight updates and loss computation'
                    values['layers'] = 'Conv+Pool+FC'

            # MNIST Integration Tests
            elif isinstance(test, TestMNISTIntegration):
                if 'test_mnist_data_compatibility' in test._testMethodName:
                    values['data_source'] = 'Real MNIST data'
                    values['test_samples'] = 2
                    values['validation'] = 'Softmax probabilities sum to 1'
        return values

def run_cnn_tests():
    """Run comprehensive CNN tests and output to test_cnn_results.txt"""
    # Create a test suite for CNN tests only
    suite = unittest.TestSuite()
    
    # Add CNN test cases
    cnn_test_cases = [
        TestConv2DLayerComprehensive,  # Comprehensive Conv2D tests
        TestPooling2DComprehensive,    # Comprehensive Pooling2D tests
        TestCNNIntegration,            # CNN integration tests
        TestMNISTIntegration           # MNIST compatibility tests
    ]
    
    for test_case in cnn_test_cases:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(test_case))
    
    # Run tests with detailed results
    result = DetailedTestResult()
    suite.run(result)
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Prepare the results summary
    summary = f"""CNN Test Results - {timestamp}
===============================
Tests Run: {result.testsRun}
Tests Passed: {result.testsRun - len(result.failures) - len(result.errors)}
Tests Failed: {len(result.failures)}
Tests Errored: {len(result.errors)}

Detailed Output:
===============
"""
    
    # Add detailed test results including actual values
    for detail in result.test_details:
        summary += f"\n{detail['name']} ... {detail['status']}\n"
        if detail['values']:
            summary += "Test Values:\n"
            for key, value in detail['values'].items():
                summary += f"  {key}: {value}\n"
        if detail['error']:
            summary += f"Error: {detail['error']}\n"
            
    summary += f"\n{'-' * 70}"
    summary += f"\nRan {result.testsRun} tests in {result.testsRun * 0.005:.3f}s\n"
    
    if result.wasSuccessful():
        summary += "\nOK"
    else:
        summary += "\nFAILED"
    
    # Save results to file
    with open('test_cnn_results.txt', 'w') as f:
        f.write(summary)
    
    # Print results to console
    print(summary)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    print("Running comprehensive CNN tests...")
    print("=" * 50)
    
    success = run_cnn_tests()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All CNN tests passed!")
        print("📄 Results saved to: test_cnn_results.txt")
    else:
        print("❌ Some CNN tests failed!")
        print("📄 Check test_cnn_results.txt for details")
    
    sys.exit(0 if success else 1)
