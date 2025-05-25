import numpy as np

# A 'Tensor' class in which the elements, the deltas, and a shape are stored.
class Tensor:
    def __init__(self, elements, deltas=None, shape=None):
        """
        Initialize a Tensor with elements, deltas, and shape.
        Args:
            elements: Input data (can be single sample or batch)
            deltas: Gradients (can be single sample or batch)
            shape: Shape of the tensor
        """
        self.elements = np.array(elements)
        if deltas is not None:
            self.deltas = np.array(deltas)
        else:
            self.deltas = None
        # Handle shape
        if shape is not None:
            self.shape = shape
        else:
            if self.elements.ndim == 1:
                self.shape = Shape(len(self.elements))
            else:
                self.shape = Shape(*self.elements.shape)

    @property
    def is_batch(self):
        """Check if tensor contains a batch of samples"""
        return self.elements.ndim > 1

    def get_batch_size(self):
        """Get the batch size if tensor contains a batch"""
        return self.elements.shape[0] if self.is_batch else 1

    def get_sample(self, index):
        """Get a single sample from the batch"""
        if not self.is_batch:
            return self
        return Tensor(
            elements=self.elements[index],
            deltas=self.deltas[index] if self.deltas is not None else None,
            shape=Shape(*self.elements.shape[1:])
        )

# A shape class that specifies the dimension of the data
class Shape:
    def __init__(self, *args):
        self.dimensions = args