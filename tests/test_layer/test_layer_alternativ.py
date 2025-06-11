import unittest
from layer import FullyConnectedLayer, Pooling2D, Conv2D

import unittest
import numpy as np
from tensor import Tensor


class TestPooling2D(unittest.TestCase):
    def test_max_pooling_forward_backward(self):
        in_tensor = Tensor(elements=np.array([
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16]
            ],
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16]
            ]
        ]))
        out_tensor = Tensor(shape=(2, 2, 2), init_type="zeros")
        pool_layer = Pooling2D(filter_shape=(2, 2), stride=(2, 2), mode="max", in_shape=(2, 4, 4))

        pool_layer.forward([in_tensor], [out_tensor])
        expected_forward_result = np.array([
            [
                [6, 8],
                [14, 16]
            ],
            [
                [6, 8],
                [14, 16]
            ]
        ])
        np.testing.assert_array_almost_equal(out_tensor.elements, expected_forward_result)

        out_tensor.deltas = np.ones((2, 2, 2))
        pool_layer.backward([out_tensor], [in_tensor])
        expected_backward_result = np.array([[
            [0, 0, 0, 0],
            [0, 1, 0, 1],
            [0, 0, 0, 0],
            [0, 1, 0, 1]
        ],
            [
                [0, 0, 0, 0],
                [0, 1, 0, 1],
                [0, 0, 0, 0],
                [0, 1, 0, 1]
            ]
        ])
        np.testing.assert_array_almost_equal(in_tensor.deltas, expected_backward_result)

    def test_average_pooling_forward_backward(self):
        in_tensor = Tensor(elements=np.array([[
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ]]))
        out_tensor = Tensor(shape=(1, 2, 2), init_type="zeros")
        pool_layer = Pooling2D(filter_shape=(2, 2), stride=(2, 2), mode="average", in_shape=(1, 4, 4))

        pool_layer.forward([in_tensor], [out_tensor])
        expected_forward_result = np.array([[
            [3.5, 5.5],
            [11.5, 13.5]
        ]])
        np.testing.assert_array_almost_equal(out_tensor.elements, expected_forward_result)

        out_tensor.deltas = np.ones((1, 2, 2))
        pool_layer.backward([out_tensor], [in_tensor])
        expected_backward_result = np.array([[
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25]
        ]])
        np.testing.assert_array_almost_equal(in_tensor.deltas, expected_backward_result)


class TestFullyConnectedLayer(unittest.TestCase):
    def setUp(self) -> None:
        self.weight_matrix = Tensor(
            elements=np.array([[3, 5], [4, 6]], dtype=np.float64)
        )
        self.bias = Tensor(elements=np.array([0.5, 0.6], dtype=np.float64))
        self.fc_layer = FullyConnectedLayer(self.weight_matrix, self.bias)

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


    def test_forward(self) -> None:
        conv2d = Conv2D(2, (2, 2), (2, 4, 3))
        # this filter has the shape (num_filters, num_channels, rows, columns)
        conv2d.filters = Tensor(
            elements=np.array(
                [
                    [[[0.1, 0.3], [-0.2, 0.4]], [[0.7, 0.9], [0.6, -1.1]]],
                    [[[0.37, 0.32], [-0.9, 0.17]], [[0.9, 0.2], [0.3, -0.7]]],
                ]
            )
        )

        # this tensor has the following shape (channels, rows, columns)
        in_tensor = Tensor(
            elements=np.array(
                [
                    [
                        [0.1, 1.2, 0.01],
                        [-0.2, 1.4, 0.2],
                        [0.5, 1.6, -0.3],
                        [0.6, 2.2, 4.0],
                    ],
                    [
                        [0.9, 1.1, 3.2],
                        [0.3, 0.7, 1.7],
                        [0.5, 2.2, 6.3],
                        [0.65, 4.4, 8.2],
                    ],
                ]
            )
        )

        out_tensors = conv2d.create_out_tensors([in_tensor])

        conv2d.forward([in_tensor], out_tensors)

        expected_output = Tensor(
            elements=np.array(
                [
                    [[2.0, 2.123], [-0.34, -3.83], [-0.83, 2.06]],
                    [[1.469, -0.1288], [-0.784, -3.689], [-1.464, -1.984]],
                ],
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
        conv2d = Conv2D(2, (2, 2), (2, 4, 3))

        conv2d.filters = Tensor(
            elements=np.array(
                [
                    [
                        [
                            [0.1, 0.3],
                            [-0.2, 0.4]
                        ],

                        [
                            [0.7, 0.9],
                            [0.6, -1.1]
                        ]
                    ],

                    [
                        [
                            [0.37, 0.32],
                            [-0.9, 0.17]
                        ],

                        [
                            [0.9, 0.2],
                            [0.3, -0.7]
                        ]
                    ]
                ])
        )

        # this tensor has the following shape (channels, rows, columns)
        # init out_tensor (actually values do not matter since they are not needed for the backward pass
        out_tensor = Tensor(
            elements=np.array(
                [
                    [[2.0, 2.123], [-0.34, -3.83], [-0.83, 2.06]],
                    [[1.469, -0.1288], [-0.784, -3.689], [-1.464, -1.984]],
                ],
                dtype=np.float64,
            )
        )
        # create made up deltas which suit for our test
        out_tensor.deltas = np.array(
            [
                [[0.1, -0.25], [0.33, 1.3], [-0.6, 0.01]],
                [[-0.5, -0.8], [0.2, 0.81], [0.1, 1.1]],
            ],
            dtype=np.float64,
        )

        in_tensor = Tensor(
            elements=np.array(
                [
                    [
                        [0.1, 1.2, 0.01],
                        [-0.2, 1.4, 0.2],
                        [0.5, 1.6, -0.3],
                        [0.6, 2.2, 4.0],
                    ],
                    [
                        [0.9, 1.1, 3.2],
                        [0.3, 0.7, 1.7],
                        [0.5, 2.2, 6.3],
                        [0.65, 4.4, 8.2],
                    ],
                ]
            )
        )

        conv2d.backward([out_tensor], [in_tensor])

        expected_output = Tensor(
            elements=np.array(
                [
                    [
                        [-0.175, -0.451, -0.33100003],
                        [0.537, 1.3177, 0.41320002],
                        [-0.269, -0.5629999, 1.0127001],
                        [0.030000009, -1.215, 0.191],
                    ],
                    [
                        [-0.38, -0.905, -0.385],
                        [0.32099998, 1.8259999, 2.1669998],
                        [-0.072000004, 0.997, -1.7679999],
                        [-0.33, 0.926, -0.78099996],
                    ],
                ]
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
        conv2d = Conv2D(2, (2, 2), (2, 4, 3))

        conv2d.filters = Tensor(
            elements=np.array(
                [
                    [
                        [
                            [0.1, 0.3],
                            [-0.2, 0.4]
                        ],

                        [
                            [0.7, 0.9],
                            [0.6, -1.1]
                        ]
                    ],

                    [
                        [
                            [0.37, 0.32],
                            [-0.9, 0.17]
                        ],

                        [
                            [0.9, 0.2],
                            [0.3, -0.7]
                        ]
                    ]
                ])
        )

        # this tensor has the following shape (channels, rows, columns)
        # init out_tensor (actually values do not matter since they are not needed for the backward pass
        out_tensor = Tensor(
            elements=np.array(
                [
                    [[2.0, 2.123], [-0.34, -3.83], [-0.83, 2.06]],
                    [[1.469, -0.1288], [-0.784, -3.689], [-1.464, -1.984]],
                ],
                dtype=np.float64,
            )
        )
        # create made up deltas which suit for our test
        out_tensor.deltas = np.array(
            [
                [[0.1, -0.25], [0.33, 1.3], [-0.6, 0.01]],
                [[-0.5, -0.8], [0.2, 0.81], [0.1, 1.1]],
            ],
            dtype=np.float64,
        )

        in_tensor = Tensor(
            elements=np.array(
                [
                    [
                        [0.1, 1.2, 0.01],
                        [-0.2, 1.4, 0.2],
                        [0.5, 1.6, -0.3],
                        [0.6, 2.2, 4.0],
                    ],
                    [
                        [0.9, 1.1, 3.2],
                        [0.3, 0.7, 1.7],
                        [0.5, 2.2, 6.3],
                        [0.65, 4.4, 8.2],
                    ],
                ]
            )
        )

        conv2d.calculate_delta_weights([out_tensor], [in_tensor])

        expected_output = Tensor(
            elements=np.array(
                [
                    [
                        [
                            [1.18, -0.1235],
                            [1.537, -1.052]
                        ],

                        [
                            [1.894, -0.336 ],
                            [2.856, 3.837 ]
                        ]
                    ],

                    [
                        [
                            [0.546, 0.494 ],
                            [2.534, 6.003 ]
                        ],

                        [
                            [1.767, 5.557],
                            [6.077, 13.293]
                        ]
                    ]
                ], dtype=np.float64
            )
        )
        self.assertTrue(
            np.allclose(
                conv2d.filters.deltas,
                expected_output.elements,
                rtol=1e-05,
                atol=1e-08,
            ),
            f"Conv2D calculate_weights_deltas function does not calculate the correct outputs."
            f"\nOutputs:\n{conv2d.filters.deltas}\nbut expected:\n{expected_output.elements}",
        )


if __name__ == "__main__":
    unittest.main()
