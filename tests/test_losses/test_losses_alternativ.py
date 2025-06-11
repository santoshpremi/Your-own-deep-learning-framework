import numpy as np
import unittest

from src.layers.losses import CrossEntropy, MeanSquaredError
from src.network.tensor import Tensor


# Import CrossEntropy and Tensor from your own module

class TestCrossEntropy(unittest.TestCase):
    def setUp(self) -> None:
        self.cross_entropy = CrossEntropy()

    def test_forward(self) -> None:
        test_cases = [
            # Check some basic cases
            {
                'predictions': [Tensor(elements=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64))],
                'targets': [Tensor(elements=np.array([1, 0, 0, 0], dtype=np.float64))],
                'expected_output': 2.3025850929940455,
            },
            {
                'predictions': [Tensor(elements=np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64))],
                'targets': [Tensor(elements=np.array([0, 0, 0, 1], dtype=np.float64))],
                'expected_output': 1.3862943611198906,
            },
            # Check an edge case where the prediction for the target class is close to 1
            {
                'predictions': [Tensor(elements=np.array([0.01, 0.01, 0.01, 0.97], dtype=np.float64))],
                'targets': [Tensor(elements=np.array([0, 0, 0, 1], dtype=np.float64))],
                'expected_output': 0.030459207484708574,
            },
            # Check an edge case where the prediction for the target class is close to 0
            {
                'predictions': [Tensor(elements=np.array([0.99, 0.01, 0.01, 0.01], dtype=np.float64))],
                'targets': [Tensor(elements=np.array([0, 0, 0, 1], dtype=np.float64))],
                'expected_output': 4.605170185988091,
            },
        ]

        for test_case in test_cases:
            out_tensors = [Tensor(elements=np.array([0], dtype=np.float64))]

            self.cross_entropy.forward(test_case['predictions'], test_case['targets'], out_tensors)
            self.assertTrue(
                np.allclose(
                    out_tensors[0].elements,
                    test_case['expected_output'],
                    rtol=1e-05,
                    atol=1e-08,
                ),
                f"forward function does not calculate the correct outputs for predictions {test_case['predictions'][0].elements} and targets {test_case['targets'][0].elements}",
            )

    def test_backward(self) -> None:
        test_cases = [
            # Check some basic cases
            {
                'predictions': [Tensor(elements=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64))],
                'targets': np.array([1, 0, 0, 0], dtype=np.float64),
                'expected_output': np.array([-10., 0., 0., 0.]),
            },
            {
                'predictions': [Tensor(elements=np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64))],
                'targets': np.array([0, 0, 0, 1], dtype=np.float64),
                'expected_output': np.array([0., 0., 0., -4.]),
            },
            # Check an edge case where the prediction for the target class is close to 1
            {
                'predictions': [Tensor(elements=np.array([0.01, 0.01, 0.01, 0.97], dtype=np.float64))],
                'targets': np.array([0, 0, 0, 1], dtype=np.float64),
                'expected_output': np.array([0., 0., 0., -1.03092783505]),
            },
            # Check an edge case where the prediction for the target class is close to 0
            {
                'predictions': [Tensor(elements=np.array([0.99, 0.01, 0.01, 0.01], dtype=np.float64))],
                'targets': np.array([0, 0, 0, 1], dtype=np.float64),
                'expected_output': np.array([0., 0., 0., -100.]),
            },
        ]

        for test_case in test_cases:
            self.cross_entropy.backward(test_case['predictions'], test_case['targets'])
            self.assertTrue(
                np.allclose(
                    test_case['predictions'][0].deltas,
                    test_case['expected_output'],
                    rtol=1e-05,
                    atol=1e-08,
                ),
                f"backward function does not calculate the correct outputs for predictions {test_case['predictions'][0].elements} and targets {test_case['targets']}",
            )


class TestMeanSquaredError(unittest.TestCase):
    def setUp(self) -> None:
        self.mean_squared_error = MeanSquaredError()

    def test_forward(self) -> None:
        test_cases = [
            {
                'predictions': [Tensor(elements=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64))],
                'targets': [Tensor(elements=np.array([1, 0, 0, 0], dtype=np.float64))],
                'expected_output': 0.27500,
            },
            {
                'predictions': [Tensor(elements=np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64))],
                'targets': [Tensor(elements=np.array([0, 0, 0, 1], dtype=np.float64))],
                'expected_output':  0.18750,
            },
            {
                'predictions': [Tensor(elements=np.array([0.01, 0.01, 0.01, 0.97], dtype=np.float64))],
                'targets': [Tensor(elements=np.array([0, 0, 0, 1], dtype=np.float64))],
                'expected_output': 0.00030,
            },
            {
                'predictions': [Tensor(elements=np.array([0.99, 0.01, 0.01, 0.01], dtype=np.float64))],
                'targets': [Tensor(elements=np.array([0, 0, 0, 1], dtype=np.float64))],
                'expected_output': 0.49010,
            },
        ]

        for test_case in test_cases:
            out_tensors = [Tensor(elements=np.array([0], dtype=np.float64))]
            self.mean_squared_error.forward(test_case['predictions'], test_case['targets'], out_tensors)
            self.assertTrue(
                np.allclose(
                    out_tensors[0].elements,
                    test_case['expected_output'],
                    rtol=1e-05,
                    atol=1e-08,
                ),
                f"forward function does not calculate the correct outputs for predictions {test_case['predictions'][0].elements} and targets {test_case['targets'][0].elements}",
            )

    def test_backward(self) -> None:
        test_cases = [
            {
                'predictions': [Tensor(elements=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64))],
                'targets': [Tensor(elements=np.array([1, 0, 0, 0], dtype=np.float64))],
                'expected_output': np.array([-0.45,  0.1 ,  0.15,  0.2]),
            },
            {
                'predictions': [Tensor(elements=np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64))],
                'targets': [Tensor(elements=np.array([0, 0, 0, 1], dtype=np.float64))],
                'expected_output': np.array([0.125, 0.125, 0.125, -0.375]),
            },
            {
                'predictions': [Tensor(elements=np.array([0.01, 0.01, 0.01, 0.97], dtype=np.float64))],
                'targets': [Tensor(elements=np.array([0, 0, 0, 1], dtype=np.float64))],
                'expected_output': np.array([0.005, 0.005, 0.005, -0.015]),
            },
            {
                'predictions': [Tensor(elements=np.array([0.99, 0.01, 0.01, 0.01], dtype=np.float64))],
                'targets': [Tensor(elements=np.array([0, 0, 0, 1], dtype=np.float64))],
                'expected_output': np.array([0.495, 0.005, 0.005, -0.495])
            },
        ]

        for test_case in test_cases:
            self.mean_squared_error.backward(test_case['predictions'], test_case['targets'])
            self.assertTrue(
                np.allclose(
                    test_case['predictions'][0].deltas,
                    test_case['expected_output'],
                    rtol=1e-05,
                    atol=1e-08,
                ),
                f"backward function does not calculate the correct outputs for predictions {test_case['predictions'][0].elements} and targets {test_case['targets'][0].elements}",
            )


if __name__ == '__main__':
    unittest.main()
