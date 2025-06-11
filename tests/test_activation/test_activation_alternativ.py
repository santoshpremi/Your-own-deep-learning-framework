import unittest
from src.layers.activation import Softmax, Sigmoid, ReLU
from src.network.tensor import Tensor
import numpy as np


class TestSoftmax(unittest.TestCase):
    def setUp(self) -> None:
        self.softmax = Softmax()

    def test_forward(self) -> None:
        in_tensors = [Tensor(elements=np.array([1, 2, 3, 4], dtype=np.float64))]
        out_tensors = [Tensor(elements=np.array([0, 0, 0, 0], dtype=np.float64))]
        self.softmax.forward(in_tensors, out_tensors)
        print(out_tensors[0].elements)
        self.assertTrue(
            np.allclose(
                out_tensors[0].elements,
                np.array([0.0320586, 0.08714432, 0.23688282, 0.64391426]),
                rtol=1e-05,
                atol=1e-08,
            ),
            "forward function does not calculate the correct outputs",
        )

    def test_backward(self) -> None:
        in_tensors = [Tensor(elements=np.array([1, 2, 3, 4], dtype=np.float64))]
        out_tensors = [Tensor(elements=np.array([0, 0, 0, 0], dtype=np.float64))]
        out_tensors[0].deltas = np.array([6, 7, 8, 9])

        sm_diag = np.array([[0.0320586, 0, 0, 0], [0, 0.08714432, 0, 0], [0, 0, 0.23688282, 0], [0, 0, 0, 0.64391426]])
        sm_mat = np.array([[0.001027753833960000, 0.002793725, 0.007594132, 0.020642989695636000],
                           [0.002793724897152000, 0.007594133, 0.020642992, 0.056113470326003200],
                           [0.007594131573252000, 0.020642992, 0.05611347, 0.152532225747013000],
                           [0.020642989695636000, 0.05611347, 0.152532226, 0.414625574231348000]])

        target_values = np.dot(out_tensors[0].deltas, (sm_diag - sm_mat))
        print(f"targets: {target_values}")
        self.softmax.forward(in_tensors, out_tensors)
        self.softmax.backward(out_tensors, in_tensors)
        print(f"calculation: {in_tensors[0].deltas}")
        self.assertTrue(
            np.allclose(
                in_tensors[0].deltas,
                target_values,
                rtol=1e-05,
                atol=1e-08,
            ),
            "backward function does not calculate the correct outputs",
        )


class TestSigmoid(unittest.TestCase):
    def setUp(self) -> None:
        self.sigmoid = Sigmoid()

    def test_forward(self) -> None:
        in_tensors = [Tensor(elements=np.array([-2, 2, 0, 4, 5], dtype=np.float64))]
        out_tensors = [Tensor(elements=np.array([0, 0, 0, 0, 0], dtype=np.float64))]
        expected_output = np.array([0.11920292, 0.88079708, 0.5, 0.98201379, 0.99330715])

        self.sigmoid.forward(in_tensors, out_tensors)
        self.assertTrue(
            np.allclose(
                out_tensors[0].elements,
                expected_output,
                rtol=1e-05,
                atol=1e-08,
            ),
            "forward function does not calculate the correct outputs",
        )

    def test_backward(self) -> None:
        in_tensors = [Tensor(elements=np.array([1, 2, 3, 4], dtype=np.float64))]
        out_tensors = [Tensor(elements=np.array([0, 0, 0, 0], dtype=np.float64))]
        out_tensors[0].deltas = np.array([6, 7, 8, 9])
        expected_output = [1.1796716, 0.7349551, 0.36141328, 0.15896436]

        self.sigmoid.forward(in_tensors, out_tensors)
        self.sigmoid.backward(out_tensors, in_tensors)
        self.assertTrue(
            np.allclose(
                in_tensors[0].deltas,
                expected_output,
                rtol=1e-05,
                atol=1e-08,
            ),
            "backward function does not calculate the correct outputs",
        )


class TestReLU(unittest.TestCase):
    def setUp(self) -> None:
        self.relu = ReLU()

    def test_forward(self) -> None:
        in_tensors = [Tensor(elements=np.array([-2, 2, 0, 4, 5], dtype=np.float64))]
        out_tensors = [Tensor(elements=np.array([0, 0, 0, 0, 0], dtype=np.float64))]
        expected_output = np.array([0, 2, 0, 4, 5])

        self.relu.forward(in_tensors, out_tensors)
        self.assertTrue(
            np.allclose(
                out_tensors[0].elements,
                expected_output,
                rtol=1e-05,
                atol=1e-08,
            ),
            "forward function does not calculate the correct outputs",
        )

    def test_backward(self) -> None:
        in_tensors = [Tensor(elements=np.array([-1, 2, -3, 4], dtype=np.float64))]
        out_tensors = [Tensor(elements=np.array([0, 0, 0, 0], dtype=np.float64))]
        out_tensors[0].deltas = np.array([-3, -7, 8, 9])
        expected_output = np.array([0, -7, 0, 9])

        self.relu.forward(in_tensors, out_tensors)
        self.relu.backward(out_tensors, in_tensors)
        self.assertTrue(
            np.allclose(
                in_tensors[0].deltas,
                expected_output,
                rtol=1e-05,
                atol=1e-08,
            ),
            "backward function does not calculate the correct outputs",
        )


if __name__ == "__main__":
    unittest.main()
