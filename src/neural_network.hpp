
#pragma once
#include "layer.hpp"
#include "loss.hpp"
#include <vector>
#include <memory>

// Neural Network class: Orchestrates the interaction between layers
// Manages forward and backward propagation through the network
template<typename T>
class NeuralNetwork {
private:
    // Collection of network layers in sequence
    std::vector<std::shared_ptr<Layer<T>>> layers;
    
    // Loss function used to compute error and gradients
    std::shared_ptr<Loss::LossFunction<T>> loss_function;

public:
    // Constructor: Initialize network with specified loss function
    NeuralNetwork(std::shared_ptr<Loss::LossFunction<T>> loss) 
        : loss_function(loss) {}

    // Add new layer to network
    // Layers are processed in sequence during forward/backward passes
    void addLayer(std::shared_ptr<Layer<T>> layer) {
        layers.push_back(layer);
    }

    // Forward propagation: Process input through all layers
    // Returns final layer output (network prediction)
    Matrix<T> forward(const Matrix<T>& input) {
        Matrix<T> current = input;
        // Pass data through each layer sequentially
        for (auto& layer : layers) {
            current = layer->forward(current);
        }
        return current;
    }

    // Backward propagation: Update network weights based on error
    // Returns computed loss value for monitoring training progress
    T backward(const Matrix<T>& expected, T learning_rate) {
        // Compute initial error from loss function derivative
        Matrix<T> error = loss_function->derivative(layers.back()->getOutput(), expected);
        
        // Propagate error backward through network
        // Each layer updates its weights and returns propagated error
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            error = (*it)->backward(error, learning_rate);
        }

        // Return final loss value
        return loss_function->calculate(layers.back()->getOutput(), expected);
    }

    // Training step: Combine forward and backward passes
    // Returns loss value for current training step
    T train(const Matrix<T>& input, const Matrix<T>& expected, T learning_rate) {
        forward(input);
        return backward(expected, learning_rate);
    }

    // Generate predictions for new input data
    // Used for inference after training
    Matrix<T> predict(const Matrix<T>& input) {
        return forward(input);
    }
};
