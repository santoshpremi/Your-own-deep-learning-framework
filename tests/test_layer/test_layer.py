import unittest
import numpy as np
from core.layers import FullyConnectedLayer, Pooling2DLayer, Conv2DLayer, PoolingType
from core.tensor import Tensor, Shape


class TestPooling2D(unittest.TestCase):
    def test_max_pooling_forward_backward(self):
        in_tensor = Tensor(elements=np.array([
            [
                [1, 1],
                [2, 2],
                [3, 3],
                [4, 4]
            ],
            [
                [5, 5],
                [6, 6],
                [7, 7],
                [8, 8]
            ],
            [
                [9, 9],
                [10, 10],
                [11, 11],
                [12, 12]
            ],
            [
                [13, 13],
                [14, 14],
                [15, 15],
                [16, 16]
            ]
        ]))
        out_tensor = Tensor(elements=np.zeros((2, 2, 2)), shape=Shape((2, 2, 2)))
        pool_layer = Pooling2DLayer(kernel_size=Shape((2, 2)), stride=Shape((2, 2)), pooling_type=PoolingType.MAX, in_shape=Shape((4, 4, 2)), out_shape=Shape((2, 2, 2)))

        pool_layer.forward([in_tensor], [out_tensor])
        expected_forward_result = np.array([
            [
                [6, 6],
                [8, 8]
            ],
            [
                [14, 14],
                [16, 16]
            ]
        ])
        np.testing.assert_array_almost_equal(out_tensor.elements, expected_forward_result)

        out_tensor.deltas = np.ones((2, 2, 2))
        pool_layer.backward([out_tensor], [in_tensor])
        expected_backward_result = np.array([
                    [
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0]
                    ],
                    [
                        [0, 0],
                        [1, 1],
                        [0, 0],
                        [1, 1]
                    ],
                    [
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0]
                    ],
                    [
                        [0, 0],
                        [1, 1],
                        [0, 0],
                        [1, 1]
                    ]
                ])

        np.testing.assert_array_almost_equal(in_tensor.deltas, expected_backward_result)

class TestFullyConnectedLayer(unittest.TestCase):
    def setUp(self) -> None:
        self.weight_matrix = Tensor(
            elements=np.array([[3, 5], [4, 6]], dtype=np.float64)
        )
        self.bias = Tensor(elements=np.array([0.5, 0.6], dtype=np.float64))
        self.fc_layer = FullyConnectedLayer(in_shape=Shape((2,1)), out_shape=Shape((2, 1)))
        self.fc_layer.weights = self.weight_matrix
        self.fc_layer.bias = self.bias

    def test_forward(self) -> None:
        in_tensors = [Tensor(elements=np.array([1, 2], dtype=np.float64))]
        out_tensors = [Tensor(elements=np.array([0, 0], dtype=np.float64))]
        self.fc_layer.forward(in_tensors, out_tensors)
        self.assertTrue(
            np.array_equal(out_tensors[0].elements, np.array([11.5, 17.6])),
            "FC Layer forward function does not calculate the correct outputs",
        )

    def test_backward(self) -> None:
        in_tensors = [Tensor(elements=np.array([1, 2], dtype=np.float64))]
        out_tensors = [Tensor(elements=np.array([0, 0], dtype=np.float64))]
        out_tensors[0].deltas = np.array([8, 9])
        self.fc_layer.backward(out_tensors, in_tensors)
        self.assertTrue(
            np.array_equal(in_tensors[0].deltas, np.array([69, 86])),
            "FC Layer backward function does not calculate the correct outputs",
        )

    def test_calculate_deltas(self) -> None:
        in_tensors = [Tensor(elements=np.array([1, 2], dtype=np.float64))]
        out_tensors = [Tensor(elements=np.array([0, 0], dtype=np.float64))]
        out_tensors[0].deltas = np.array([8, 9])
        self.fc_layer.calculate_delta_weights(out_tensors, in_tensors)
        self.assertTrue(
            np.array_equal(self.weight_matrix.deltas, np.array([[8, 9], [16, 18]])),
            "FCLayer calculate delta weights function does not calculate the correct deltas for the weight matrix",
        )
        self.assertTrue(
            np.array_equal(self.bias.deltas, out_tensors[0].deltas),
            "calculate delta weights function does not calculate the correct deltas for the bias",
        )

class TestConv2DLayer(unittest.TestCase):
    def test_forward(self) -> None:
        conv2d = Conv2DLayer(in_shape=Shape((4, 3, 2)), out_shape=Shape((3, 2, 2)), kernel_size=Shape((2, 2)), num_filters=2)
        # this filter has the shape (rows, columns, num_channels, num_filters)
        conv2d.weights = Tensor(
            elements=np.array(
                [[[ [ 0.1, 0.37],
                    [ 0.7, 0.9 ]],

                    [[ 0.3, 0.32],
                    [ 0.9, 0.2 ]]],


                    [[[-0.2, -0.9 ],
                    [ 0.6, 0.3 ]],

                    [[ 0.4, 0.17],
                    [-1.1, -0.7 ]]]]
            )
        )

        conv2d.bias = Tensor(elements=np.array([0, 0]))

        # this tensor has the following shape (rows, columns, num_channels)
        in_tensor = Tensor(
            elements=np.array(
                [[  [ 0.1, 0.9 ],
                    [ 1.2, 1.1 ],
                    [ 0.01, 3.2 ]],

                    [[-0.2, 0.3 ],
                    [ 1.4, 0.7 ],
                    [ 0.2, 1.7 ]],

                    [[ 0.5, 0.5 ],
                    [ 1.6, 2.2 ],
                    [-0.3, 6.3 ]],

                    [[ 0.6, 0.65],
                    [ 2.2, 4.4 ],
                    [ 4., 8.2 ]]]
            )
        )

        out_tensors = [Tensor(elements=np.zeros((3, 2, 2), dtype=np.float64))]

        conv2d.forward([in_tensor], out_tensors)

        expected_output = Tensor(
            elements=np.array(
                [[  [ 2., 1.469 ],
                    [ 2.123, -0.1288]],

                    [[-0.34, -0.784 ],
                    [-3.83, -3.689 ]],

                    [[-0.83, -1.464 ],
                    [ 2.06, -1.984 ]]],
                dtype=np.float64,
            )
        )

        self.assertTrue(
            np.allclose(
                out_tensors[0].elements,
                expected_output.elements,
                rtol=1e-05,
                atol=1e-07,
            ),
            "Conv2D forward function does not calculate the correct outputs",
        )

    def test_backward(self) -> None:
        conv2d = Conv2DLayer(in_shape=Shape((4, 3, 2)), out_shape=Shape((3, 2, 2)), kernel_size=Shape((2, 2)), num_filters=2)
        # this filter has the shape (rows, columns, num_channels, num_filters)
        conv2d.weights = Tensor(
            elements=np.array(
                [[[ [ 0.1, 0.37],
                    [ 0.7, 0.9 ]],

                    [[ 0.3, 0.32],
                    [ 0.9, 0.2 ]]],


                    [[[-0.2, -0.9 ],
                    [ 0.6, 0.3 ]],

                    [[ 0.4, 0.17],
                    [-1.1, -0.7 ]]]]
            )
        )

        conv2d.bias = Tensor(elements=np.array([0, 0]))

        # this tensor has the following shape (rows, columns, channels)
        # init out_tensor (actually values do not matter since they are not needed for the backward pass
        out_tensor = Tensor(
            elements=np.array(
                [   [[ 2.,      1.469 ],
                    [ 2.123,  -0.1288]],

                    [[-0.34,   -0.784 ],
                    [-3.83,   -3.689 ]],

                    [[-0.83,   -1.464 ],
                    [ 2.06,   -1.984 ]]],
                dtype=np.float64,
            )
        )
        # create made up deltas which suit for our test
        out_tensor.deltas = np.array(
            [[  [ 0.1,  -0.5 ],
                [-0.25, -0.8 ]],

                [[ 0.33,  0.2 ],
                [ 1.3,   0.81]],

                [[-0.6,   0.1 ],
                [ 0.01,  1.1 ]]],
            dtype=np.float64,
        )

        in_tensor = Tensor(
            elements=np.array(
                [[  [ 0.1, 0.9 ],
                    [ 1.2, 1.1 ],
                    [ 0.01, 3.2 ]],

                    [[-0.2, 0.3 ],
                    [ 1.4, 0.7 ],
                    [ 0.2, 1.7 ]],

                    [[ 0.5, 0.5 ],
                    [ 1.6, 2.2 ],
                    [-0.3, 6.3 ]],

                    [[ 0.6, 0.65],
                    [ 2.2, 4.4 ],
                    [ 4., 8.2 ]]]
            )
        )

        conv2d.backward([out_tensor], [in_tensor])

        expected_output = Tensor(
            elements=np.array(
                    [   [[-0.175,      -0.38      ],
                        [-0.451,      -0.905     ],
                        [-0.33100003, -0.385     ]],

                        [[ 0.537,       0.32099998],
                        [ 1.3177,      1.8259999 ],
                        [ 0.41320002,  2.1669998 ]],

                        [[-0.269,      -0.072     ],
                        [-0.5629999,   0.997     ],
                        [ 1.0127001,  -1.7679999 ]],

                        [[ 0.03000001, -0.33      ],
                        [-1.215,       0.926     ],
                        [ 0.191,      -0.78099996]]]
            )
        )

        self.assertTrue(
            np.allclose(
                in_tensor.deltas,
                expected_output.elements,
                rtol=1e-05,
                atol=1e-08,
            ),
            f"Conv2D backward function does not calculate the correct outputs.\nOutputs:\n{in_tensor.deltas}\nbut expected:\n{expected_output.elements}",
        )

    def test_weight_update(self) -> None:
        conv2d = Conv2DLayer(in_shape=Shape((4, 3, 2)), out_shape=Shape((3, 2, 2)), kernel_size=Shape((2, 2)), num_filters=2)
        # this filter has the shape (rows, columns, num_channels, num_filters)
        conv2d.weights = Tensor(
            elements=np.array(
                [[[ [ 0.1, 0.37],
                    [ 0.7, 0.9 ]],

                    [[ 0.3, 0.32],
                    [ 0.9, 0.2 ]]],


                    [[[-0.2, -0.9 ],
                    [ 0.6, 0.3 ]],

                    [[ 0.4, 0.17],
                    [-1.1, -0.7 ]]]]
            )
        )

        conv2d.bias = Tensor(elements=np.array([0.0, 0.0]))

        # this tensor has the following shape (rows, columns, channels)
        # init out_tensor (actually values do not matter since they are not needed for the backward pass
        out_tensor = Tensor(
            elements=np.array(
                [   [[ 2.,      1.469 ],
                    [ 2.123,  -0.1288]],

                    [[-0.34,   -0.784 ],
                    [-3.83,   -3.689 ]],

                    [[-0.83,   -1.464 ],
                    [ 2.06,   -1.984 ]]],
                dtype=np.float64,
            )
        )

        # create made up deltas which suit for our test
        out_tensor.deltas = np.array(
            [[  [ 0.1,  -0.5 ],
                [-0.25, -0.8 ]],

                [[ 0.33,  0.2 ],
                [ 1.3,   0.81]],

                [[-0.6,   0.1 ],
                [ 0.01,  1.1 ]]],
            dtype=np.float64,
        )

        in_tensor = Tensor(
            elements=np.array(
                [[  [ 0.1, 0.9 ],
                    [ 1.2, 1.1 ],
                    [ 0.01, 3.2 ]],

                    [[-0.2, 0.3 ],
                    [ 1.4, 0.7 ],
                    [ 0.2, 1.7 ]],

                    [[ 0.5, 0.5 ],
                    [ 1.6, 2.2 ],
                    [-0.3, 6.3 ]],

                    [[ 0.6, 0.65],
                    [ 2.2, 4.4 ],
                    [ 4., 8.2 ]]]
            )
        )

        conv2d.calculate_delta_weights([out_tensor], [in_tensor])

        expected_output = Tensor(
            elements=np.array(
                [   [[[ 1.18,    1.894 ],
                    [ 0.546,   1.767 ]],

                    [[-0.1235, -0.336 ],
                    [ 0.494,   5.557 ]]],


                    [[[ 1.537,   2.856 ],
                    [ 2.534,   6.077 ]],

                    [[-1.052,   3.837 ],
                    [ 6.003,  13.293 ]]]
                ], dtype=np.float64
            )
        )

        self.assertTrue(
            np.allclose(
                conv2d.weights.deltas,
                expected_output.elements,
                rtol=1e-05,
                atol=1e-08,
            ),
            f"Conv2D calculate_weights_deltas function does not calculate the correct outputs."
            f"\nOutputs:\n{conv2d.weights.deltas}\nbut expected:\n{expected_output.elements}",
        )

    


if __name__ == "__main__":
    unittest.main()
