import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_mnist(flatten: bool = True):
    """Load MNIST dataset.
    
    Args:
        flatten: If True, flatten images to 1D vectors. If False, keep as 2D images.
        
    Returns:
        (X_train, y_train), (X_test, y_test): Training and test data
    """
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    
    # Reshape to (N, 1, 28, 28) if not flattening
    if not flatten:
        X = X.reshape(-1, 1, 28, 28)
    
    # Normalize pixel values to [0, 1]
    X = X.astype(np.float32) / 255.0
    
    # Convert labels to one-hot encoding
    num_classes = 10
    y_one_hot = np.zeros((y.size, num_classes), dtype=np.float32)
    y_one_hot[np.arange(y.size), y.astype(int)] = 1
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_one_hot, test_size=0.2, random_state=42
    )
    
    return (X_train, y_train), (X_test, y_test)