
#pragma once
#include "matrix.hpp"
#include "activation.hpp"
#include <memory>

// Layer class: Represents a fully connected neural network layer
// Manages weights, biases, and activation functions for one layer
template<typename T>
class Layer {
private:
    Matrix<T> weights;    // Weight matrix: [input_size x output_size]
    Matrix<T> bias;       // Bias vector: [1 x output_size]
    std::shared_ptr<Activation::ActivationFunction<T>> activation;  // Activation function
    Matrix<T> input;      // Cached input for backward pass
    Matrix<T> output;     // Cached output for backward pass
    Matrix<T> delta;      // Cached error terms for backward pass

public:
    // Constructor: Initialize layer with specified dimensions and activation
    Layer(size_t input_size, size_t output_size, 
          std::shared_ptr<Activation::ActivationFunction<T>> act)
        : weights(input_size, output_size),
          bias(1, output_size),
          activation(act),
          input(1, input_size),      // Cache matrices initialized with proper dimensions
          output(1, output_size),    // for storing intermediate values during
          delta(1, output_size)      // forward and backward passes
    {
        // Initialize weights and biases with random values
        weights.randomize();  // Random initialization helps break symmetry
        bias.randomize();     // during training
    }

    // Forward propagation through layer
    // Computes: activation(input * weights + bias)
    Matrix<T> forward(const Matrix<T>& input) {
        this->input = input;  // Cache input for backward pass
        
        // Compute weighted sum: z = input * weights + bias
        Matrix<T> z = input.dot(weights);
        // Add bias to each output neuron
        for (size_t i = 0; i < z.getRows(); ++i) {
            for (size_t j = 0; j < z.getCols(); ++j) {
                z.at(i, j) += bias.at(0, j);
            }
        }
        
        // Apply activation function and cache result
        output = activation->forward(z);
        return output;
    }

    // Backward propagation through layer
    // Updates weights and biases, returns propagated error
    Matrix<T> backward(const Matrix<T>& error, T learning_rate) {
        // Compute local gradient: error * activation_derivative
        Matrix<T> delta = error.hadamard(activation->backward(output));
        this->delta = delta;  // Cache for potential later use

        // Compute weight gradients
        Matrix<T> weights_gradient = input.transpose().dot(delta);
        
        // Compute bias gradients (sum error terms for each output neuron)
        Matrix<T> bias_gradient(1, bias.getCols());
        for (size_t j = 0; j < delta.getCols(); ++j) {
            T sum = 0;
            for (size_t i = 0; i < delta.getRows(); ++i) {
                sum += delta.at(i, j);
            }
            bias_gradient.at(0, j) = sum;
        }

        // Update weights and bias using gradient descent
        weights = weights - weights_gradient * learning_rate;
        bias = bias - bias_gradient * learning_rate;

        // Propagate error to previous layer
        return delta.dot(weights.transpose());
    }

    // Accessor methods for layer components
    const Matrix<T>& getWeights() const { return weights; }
    const Matrix<T>& getBias() const { return bias; }
    const Matrix<T>& getOutput() const { return output; }
    const Matrix<T>& getDelta() const { return delta; }
};
