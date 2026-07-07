```markdown
# PyTorch Neural Network Classification

## Overview
This notebook explores neural network classification using PyTorch, starting with simple binary classification on 'circle' data and progressing to multi-class classification on 'blob' data. It covers model building, training, evaluation, and the importance of non-linear activation functions.

## Setup
To run this notebook, ensure you have the necessary libraries installed. You can install them using pip:
```bash
pip install torch torchvision torchaudio sklearn matplotlib pandas torchmetrics
```

## Data

### 1. Circle Data (Binary Classification)
- **Generation**: Synthetic 2D data generated using `sklearn.datasets.make_circles`.
- **Visualization**: Scatter plot showing two concentric circles, colored by class.
- **Splitting**: Divided into training and testing sets (80/20 split).
- **Conversion**: Converted from NumPy arrays to PyTorch tensors with `float` data type.

### 2. Blob Data (Multi-Class Classification)
- **Generation**: Synthetic 2D multi-class data generated using `sklearn.datasets.make_blobs`.
- **Parameters**: `NUM_CLASSES=4`, `NUM_FEATURES=2`, `cluster_std=1.5`.
- **Visualization**: Scatter plot showing distinct clusters, colored by class.
- **Splitting**: Divided into training and testing sets (80/20 split).
- **Conversion**: Converted from NumPy arrays to PyTorch tensors (`float` for features, `long` for labels).

## Models

### 1. `CircleModelV0` (Linear Model - Binary Classification)
- **Architecture**: Two `nn.Linear` layers (`in=2, out=5`, `in=5, out=1`).
- **Activation**: None (purely linear).
- **Purpose**: Demonstrates the limitations of linear models on non-linearly separable data.

### 2. `CircleModelV1` (Deeper Linear Model - Binary Classification)
- **Architecture**: Three `nn.Linear` layers (`in=2, out=10`, `in=10, out=10`, `in=10, out=1`).
- **Activation**: None.
- **Purpose**: Shows that increasing layers in a linear model doesn't solve non-linear problems.

### 3. `CircleModelV2` (Non-linear Model - Binary Classification)
- **Architecture**: Three `nn.Linear` layers with `nn.ReLU` activation functions between them (`in=2, out=10`, `ReLU`, `in=10, out=10`, `ReLU`, `in=10, out=1`).
- **Purpose**: Illustrates how non-linearity enables the model to learn complex patterns.

### 4. `BlobModel` (Non-linear Model - Multi-Class Classification)
- **Architecture**: Sequential stack of `nn.Linear` and `nn.ReLU` layers.
  - `nn.Linear(in_features=2, out_features=8)`
  - `nn.ReLU()`
  - `nn.Linear(in_features=8, out_features=8)`
  - `nn.ReLU()`
  - `nn.Linear(in_features=8, out_features=4)` (output features match `NUM_CLASSES`)
- **Purpose**: Applied to the multi-class 'blob' dataset.

## Training Process

### Common Elements:
- **Device Agnostic Code**: Models and data are moved to `cuda` (GPU) if available, otherwise `cpu`.
- **Loss Function**: 
  - Binary Classification: `nn.BCEWithLogitsLoss()` (combines Sigmoid and Binary Cross Entropy).
  - Multi-Class Classification: `nn.CrossEntropyLoss()`.
- **Optimizer**: `torch.optim.SGD` (Stochastic Gradient Descent) with a learning rate of `0.1` or `0.01`.
- **Accuracy Function**: Custom `accuracy_fn` to calculate percentage accuracy.
- **Epochs**: Models are trained for 100 to 1000 epochs.
- **Training Loop**: Standard PyTorch training loop including forward pass, loss calculation, backward pass, and optimizer step.
- **Evaluation Loop**: Model is set to `eval()` mode, predictions are made with `torch.inference_mode()`, and test loss/accuracy are calculated.

### Specifics:
- **Linear Models (`ModelV0`, `ModelV1`)**: Show consistently low accuracy (around 50%) due to inability to learn non-linear decision boundaries.
- **Non-linear Model (`ModelV2`)**: Achieves significantly higher accuracy (around 74% train, 79% test) on the 'circle' data, demonstrating the power of activation functions.
- **Blob Model (`ModelV4`)**: Achieves very high accuracy (up to 99.5%) on the 'blob' data, successfully classifying the distinct clusters.

## Evaluation

### Metrics Used:
- **Loss**: Binary Cross Entropy Loss or Cross Entropy Loss.
- **Accuracy**: Custom `accuracy_fn`.
- **Decision Boundary Plots**: Visualizations (`plot_decision_boundary`) are used to show how well each model separates the classes.

### `torchmetrics` Integration:
- The `torchmetrics` library is used for more robust evaluation of the `BlobModel`.
- **Metrics calculated**: Accuracy, Precision, Recall, and F1 Score.
- **Results for `BlobModel`**: All metrics are around `0.995`, indicating excellent performance on the multi-class dataset.

## Key Learnings
- Linear models are insufficient for non-linearly separable data.
- Non-linear activation functions (like ReLU) are crucial for neural networks to learn complex patterns.
- PyTorch provides flexible tools for building, training, and evaluating various neural network architectures.
- `torchmetrics` offers a convenient way to compute a wide range of classification metrics.

```
