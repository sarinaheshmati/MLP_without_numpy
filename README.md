
# MLP from Scratch (No NumPy)

This project implements a basic Multi-Layer Perceptron (MLP) from scratch without using any external libraries like NumPy. The implementation includes forward propagation, backpropagation using automatic differentiation, and optimization for weight updates.

## Project Overview

### Key Components
1. **MLP Structure**: 
   - The MLP consists of multiple layers, each containing several neurons. Each neuron computes a weighted sum of the inputs, applies a non-linear activation (e.g., Tanh), and outputs the result.
   
2. **Forward Propagation**:
   - The forward pass computes predictions by passing inputs through the network layers.


3. **Backpropagation (Automatic Differentiation)**:
   - The backpropagation step is crucial for training the network. It involves calculating the gradient of the loss function with respect to each of the model’s parameters (weights and biases). This gradient is used to update the parameters to minimize the loss.
   - In this implementation, backpropagation is handled using **reverse-mode automatic differentiation**, which constructs a computational graph during the forward pass. Each operation (e.g., addition, multiplication) creates a node in the graph, allowing gradients to be calculated efficiently during the backward pass.
   
   - **Reverse-Mode Autodiff**: Instead of manually deriving the gradients, each tensor keeps track of its parent tensors and the operation used to create it. By recursively traversing this graph in reverse (starting from the loss), the gradient with respect to each parameter is automatically computed.
    
4. **Optimizer**:
   - Weights and biases are updated using an optimizer that performs gradient descent based on the computed gradients.

### Main Methods
- `forward(x)`: Performs forward propagation through the network.
- `__call__(x)`: A callable wrapper around `forward`.
- `parameters()`: Returns a list of all network weights.

### Training Loop
- The model is trained using a synthetic dataset, where mean squared error (MSE) is minimized using gradient descent.
- The gradients are computed automatically during backpropagation, and weights are updated accordingly.

## Usage

To train the MLP:
1. Create an MLP model by specifying the input size and the number of neurons in each layer.
2. Use the provided optimizer for gradient updates.
3. Run the training loop, which includes forward pass, loss computation, and backward pass.

Example code for training is included in the file.

## Visualization
The computational graph of the network can be visualized using Graphviz to track the operations and gradients during training.
