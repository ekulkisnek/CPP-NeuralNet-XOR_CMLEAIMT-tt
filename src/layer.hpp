#pragma once
#include "matrix.hpp"
#include "activation.hpp"
#include <memory>

template<typename T>
class Layer {
private:
    Matrix<T> weights;
    Matrix<T> bias;
    std::shared_ptr<Activation::ActivationFunction<T>> activation;
    Matrix<T> input;
    Matrix<T> output;
    Matrix<T> delta;

public:
    Layer(size_t input_size, size_t output_size, 
          std::shared_ptr<Activation::ActivationFunction<T>> act)
        : weights(input_size, output_size),
          bias(1, output_size),
          activation(act),
          input(1, input_size),      // Initialize with proper dimensions
          output(1, output_size),    // Initialize with proper dimensions
          delta(1, output_size)      // Initialize with proper dimensions
    {
        weights.randomize();
        bias.randomize();
    }

    Matrix<T> forward(const Matrix<T>& input) {
        this->input = input;
        Matrix<T> z = input.dot(weights);
        for (size_t i = 0; i < z.getRows(); ++i) {
            for (size_t j = 0; j < z.getCols(); ++j) {
                z.at(i, j) += bias.at(0, j);
            }
        }
        output = activation->forward(z);
        return output;
    }

    Matrix<T> backward(const Matrix<T>& error, T learning_rate) {
        Matrix<T> delta = error.hadamard(activation->backward(output));
        this->delta = delta;

        Matrix<T> weights_gradient = input.transpose().dot(delta);
        Matrix<T> bias_gradient(1, bias.getCols());

        // Calculate bias gradient
        for (size_t j = 0; j < delta.getCols(); ++j) {
            T sum = 0;
            for (size_t i = 0; i < delta.getRows(); ++i) {
                sum += delta.at(i, j);
            }
            bias_gradient.at(0, j) = sum;
        }

        // Update weights and bias
        weights = weights - weights_gradient * learning_rate;
        bias = bias - bias_gradient * learning_rate;

        // Propagate error to previous layer
        return delta.dot(weights.transpose());
    }

    const Matrix<T>& getWeights() const { return weights; }
    const Matrix<T>& getBias() const { return bias; }
    const Matrix<T>& getOutput() const { return output; }
    const Matrix<T>& getDelta() const { return delta; }
};