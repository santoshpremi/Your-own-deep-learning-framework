# MNIST CNN Results - Exercise 2

This directory contains the training results and model weights for the Convolutional Neural Network (CNN) trained on the MNIST dataset for Exercise 2.

## Structure

```
mnist_cnn_results/
├── training_results.txt    # Detailed training metrics for each epoch
└── weights/               # Saved model weights and checkpoints
    ├── Conv2DLayer_1_bias.pkl           # Conv layer 1 bias weights
    ├── Conv2DLayer_1_weights.pkl        # Conv layer 1 filter weights
    ├── Conv2DLayer_4_bias.pkl           # Conv layer 4 bias weights
    ├── Conv2DLayer_4_weights.pkl        # Conv layer 4 filter weights
    ├── FullyConnectedLayer_8_bias.pkl   # FC layer 8 bias weights
    ├── FullyConnectedLayer_8_weights.pkl # FC layer 8 weights
    ├── FullyConnectedLayer_10_bias.pkl  # FC layer 10 bias weights
    ├── FullyConnectedLayer_10_weights.pkl # FC layer 10 weights
    ├── best_model.pkl/      # Best performing model checkpoint
    ├── epoch_005.pkl/       # Model checkpoint at epoch 5
    ├── epoch_010.pkl/       # Model checkpoint at epoch 10
    └── epoch_012.pkl/       # Final model checkpoint at epoch 12
```

## Training Results Summary

- **Architecture**: CNN with Conv2D → Pool → Conv2D → Pool → Flatten → FC → FC → Softmax
- **Total Epochs**: 12
- **Training Status**: NEEDS DEBUGGING - Model showing poor performance
- **Best Test Accuracy**: 9.5% (Random chance level - indicates implementation issues)
- **Final Loss**: ~31.0 (Very high - suggests training problems)

## Issues Identified

⚠️ **Warning**: This CNN training shows signs of fundamental implementation problems:
- Accuracy stuck at random chance level (~9.6%)
- Loss not decreasing over epochs
- Consistent poor performance across all epochs

## Comparison with Exercise 1 (Regular Neural Network)

| Metric | Exercise 1 (MLP) | Exercise 2 (CNN) | Status |
|--------|------------------|------------------|---------|
| Best Test Accuracy | 97.75% | 9.50% | ❌ CNN Needs Fix |
| Final Loss | 0.0300 | 31.02 | ❌ CNN Not Learning |
| Training Convergence | ✅ Successful | ❌ Failed | ❌ Debug Required |

## Next Steps

1. **Debug CNN Implementation**: Check Conv2D layer forward/backward pass
2. **Verify Data Preprocessing**: Ensure MNIST data is properly formatted for CNN
3. **Check Architecture**: Verify layer connections and shapes
4. **Review Training Loop**: Ensure gradients are flowing correctly

## Files Format

- `training_results.txt`: Tab-separated values with columns: Epoch, Train Acc, Test Acc, Loss, Time (s), LR
- Weight files: Individual layer parameters saved as pickle files
- Checkpoint directories: Complete model state at specific epochs
