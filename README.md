Neural Network XOR Implementation

This C++ project implements a flexible neural network framework and demonstrates its capabilities by solving the XOR problem. The implementation showcases object-oriented design principles and modern C++ features while providing a clear example of how neural networks work.
Project Overview

The neural network successfully learns the XOR operation, which is a classical problem in neural network training. The XOR operation returns true (1) when inputs are different and false (0) when inputs are the same.
Core Components
1. Matrix Operations (
)

The foundation of the neural network is the Matrix class, which provides essential operations:

    Matrix multiplication (dot product)
    Element-wise multiplication (Hadamard product)
    Matrix transposition
    Basic arithmetic operations (+, -, *)
    Random initialization

2. Neural Network Architecture (
)

The NeuralNetwork class orchestrates the entire learning process:

    Forward propagation through layers
    Backward propagation for updating weights
    Training loop implementation
    Prediction generation

3. Layer Implementation (
)

Each Layer class represents a fully connected neural network layer:

    Weights and bias initialization
    Forward propagation implementation
    Backward propagation with gradient calculations
    Weight and bias updates during training

4. Activation Functions (
)

Two activation functions are implemented:

    ReLU (Rectified Linear Unit)
        Forward: max(0, x)
        Backward: 1 if x > 0, else 0
    Sigmoid
        Forward: 1/(1 + e^(-x))
        Backward: sigmoid(x) * (1 - sigmoid(x))

5. Loss Functions (
)

The Mean Squared Error (MSE) loss function is implemented:

    Forward: average of squared differences
    Backward: derivative for gradient descent

6. Utilities (
)

Helper functions for:

    XOR data generation
    Accuracy metrics calculation

Implementation Details

The network architecture used for the XOR problem consists of:

    Input layer: 2 neurons (for two binary inputs)
    Hidden layer: 4 neurons with ReLU activation
    Output layer: 1 neuron with Sigmoid activation

The training process:

    Generates 1000 training samples of XOR data
    Trains for 1000 epochs
    Uses a learning rate of 0.1
    Prints progress every 100 epochs
    Tests the model on 100 separate test samples

The example achieves 100% accuracy on the XOR problem, demonstrating proper convergence and learning capabilities.
Technical Implementation

The implementation uses several modern C++ features:

    Template programming for flexible data types
    Smart pointers for memory management
    STL containers and algorithms
    RAII principles
    Namespace organization
    Multiple inheritance for activation and loss functions

The backpropagation algorithm is implemented efficiently using matrix operations, and the code structure allows for easy extension with new activation functions, loss functions, or layer types.

The project demonstrates proper separation of concerns, with each component handling its specific responsibilities while maintaining clean interfaces between different parts of the system.