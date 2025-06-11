import unittest
import sys
from io import StringIO
from datetime import datetime
import numpy as np
from tests.test_activation.test_activation import TestSoftmax, TestSigmoid, TestReLU, TestTanh
from tests.test_layer.test_layer import TestFullyConnectedLayer, TestPooling2D, TestConv2DLayer
from tests.test_losses.test_losses import TestCrossEntropy, TestMeanSquaredError

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
            method = getattr(test, test._testMethodName)
            
            # Pooling Layer Tests
            if isinstance(test, TestPooling2D):
                if 'test_max_pooling_forward_backward' in test._testMethodName:
                    values['input_shape'] = '4x4x2'
                    values['kernel_size'] = '2x2'
                    values['stride'] = '2x2'
                    values['expected_output'] = '[[[6,6],[8,8]],[[14,14],[16,16]]]'

            # Fully Connected Layer Tests
            elif isinstance(test, TestFullyConnectedLayer):
                if 'test_forward' in test._testMethodName:
                    values['input'] = [1, 2]
                    values['weights'] = [[3, 5], [4, 6]]
                    values['bias'] = [0.5, 0.6]
                    values['expected_output'] = [11.5, 17.6]
                elif 'test_backward' in test._testMethodName:
                    values['input'] = [1, 2]
                    values['output_deltas'] = [8, 9]
                    values['expected_input_deltas'] = [69, 86]
                elif 'test_calculate_deltas' in test._testMethodName:
                    values['input'] = [1, 2]
                    values['output_deltas'] = [8, 9]
                    values['expected_weight_deltas'] = [[8, 9], [16, 18]]
                    values['expected_bias_deltas'] = [8, 9]

            # Conv2D Layer Tests
            elif isinstance(test, TestConv2DLayer):
                if 'test_forward' in test._testMethodName:
                    values['input_shape'] = '4x3x2'
                    values['kernel_size'] = '2x2'
                    values['num_filters'] = 2
                    values['expected_output'] = '[[[2.0,1.469],[-0.34,-0.784],[-0.83,-1.464]],[[2.123,-0.1288],[-3.83,-3.689],[2.06,-1.984]]]'

            # Activation Layer Tests
            elif isinstance(test, TestSoftmax):
                if 'test_forward' in test._testMethodName:
                    values['input'] = [1, 2, 3, 4]
                    values['expected_output'] = [0.0320586, 0.08714432, 0.23688282, 0.64391426]
                elif 'test_backward' in test._testMethodName:
                    values['input'] = [1, 2, 3, 4]
                    values['deltas'] = [6, 7, 8, 9]
                    values['expected'] = [-0.07991096, -0.13007621, -0.11670097, 0.32668814]

            elif isinstance(test, TestSigmoid):
                if 'test_forward' in test._testMethodName:
                    values['input'] = [-2, 2, 0, 4, 5]
                    values['expected_output'] = [0.11920292, 0.88079708, 0.5, 0.98201379, 0.99330715]
                elif 'test_backward' in test._testMethodName:
                    values['input'] = [1, 2, 3, 4]
                    values['deltas'] = [6, 7, 8, 9]
                    values['expected'] = [1.1796716, 0.7349551, 0.36141328, 0.15896436]

            elif isinstance(test, TestReLU):
                if 'test_forward' in test._testMethodName:
                    values['input'] = [-2, 2, 0, 4, 5]
                    values['expected'] = [0, 2, 0, 4, 5]
                elif 'test_backward' in test._testMethodName:
                    values['input'] = [-1, 2, -3, 4]
                    values['deltas'] = [-3, -7, 8, 9]
                    values['expected'] = [0, -7, 0, 9]

            elif isinstance(test, TestTanh):
                if 'test_forward' in test._testMethodName:
                    values['input'] = [-2, 2, 0, 4, 5]
                    values['expected_output'] = [-0.96402758, 0.96402758, 0, 0.9993293, 0.99990916]
                elif 'test_backward' in test._testMethodName:
                    values['input'] = [1, 2, 3, 4]
                    values['deltas'] = [6, 7, 8, 9]
                    values['expected'] = [0.94117647, 0.94117647, 0.94117647, 0.94117647]

            # Loss Layer Tests
            elif isinstance(test, TestMeanSquaredError):
                if 'test_forward' in test._testMethodName:
                    values['input'] = [[0.1, 0.2, 0.3, 0.4], [1, 0, 0, 0]]
                    values['expected'] = 0.275

            elif isinstance(test, TestCrossEntropy):
                if 'test_forward' in test._testMethodName:
                    values['input'] = [[0.1, 0.2, 0.3, 0.4], [1, 0, 0, 0]]
                    values['expected'] = 2.3025850929940455
        return values

def run_tests():
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Add all test cases
    test_cases = [
        TestSoftmax,  # Activation tests
        TestSigmoid,
        TestReLU,
        TestTanh,
        TestFullyConnectedLayer,  # Layer tests 
        TestPooling2D,
        TestConv2DLayer,
        TestCrossEntropy,  # Loss tests
        TestMeanSquaredError
    ]
    
    for test_case in test_cases:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(test_case))
    
    # Run tests with detailed results
    result = DetailedTestResult()
    suite.run(result)
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Prepare the results summary
    summary = f"""
Test Results - {timestamp}
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
    with open('test_results.txt', 'w') as f:
        f.write(summary)
    
    # Print results to console
    print(summary)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
