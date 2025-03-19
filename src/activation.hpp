
#pragma once
#include <cmath>
#include "matrix.hpp"

namespace Activation {
    // Base class for activation functions
    // Defines interface for forward and backward passes
    template<typename T>
    class ActivationFunction {
    public:
        // Forward pass: compute activation value
        virtual Matrix<T> forward(const Matrix<T>& x) const = 0;
        
        // Backward pass: compute activation derivative
        virtual Matrix<T> backward(const Matrix<T>& x) const = 0;
        
        virtual ~ActivationFunction() = default;
    };

    // ReLU (Rectified Linear Unit) activation
    // f(x) = max(0, x)
    // Derivative: 1 if x > 0, 0 otherwise
    template<typename T>
    class ReLU : public ActivationFunction<T> {
    public:
        // Forward pass: max(0, x)
        // Helps network learn non-linear patterns
        Matrix<T> forward(const Matrix<T>& x) const override {
            Matrix<T> result(x.getRows(), x.getCols());
            for (size_t i = 0; i < x.getRows(); ++i) {
                for (size_t j = 0; j < x.getCols(); ++j) {
                    result.at(i, j) = std::max(static_cast<T>(0), x.at(i, j));
                }
            }
            return result;
        }

        // Backward pass: derivative of ReLU
        // Returns 1 for positive inputs, 0 for negative
        Matrix<T> backward(const Matrix<T>& x) const override {
            Matrix<T> result(x.getRows(), x.getCols());
            for (size_t i = 0; i < x.getRows(); ++i) {
                for (size_t j = 0; j < x.getCols(); ++j) {
                    result.at(i, j) = x.at(i, j) > 0 ? 1 : 0;
                }
            }
            return result;
        }
    };

    // Sigmoid activation function
    // f(x) = 1 / (1 + e^(-x))
    // Useful for binary classification outputs
    template<typename T>
    class Sigmoid : public ActivationFunction<T> {
    public:
        // Forward pass: compute sigmoid value
        // Maps any input to range (0,1)
        Matrix<T> forward(const Matrix<T>& x) const override {
            Matrix<T> result(x.getRows(), x.getCols());
            for (size_t i = 0; i < x.getRows(); ++i) {
                for (size_t j = 0; j < x.getCols(); ++j) {
                    result.at(i, j) = 1.0 / (1.0 + std::exp(-x.at(i, j)));
                }
            }
            return result;
        }

        // Backward pass: derivative of sigmoid
        // f'(x) = f(x) * (1 - f(x))
        Matrix<T> backward(const Matrix<T>& x) const override {
            Matrix<T> sig = forward(x);  // Compute sigmoid values
            Matrix<T> result(x.getRows(), x.getCols());
            for (size_t i = 0; i < x.getRows(); ++i) {
                for (size_t j = 0; j < x.getCols(); ++j) {
                    T s = sig.at(i, j);
                    result.at(i, j) = s * (1 - s);  // Derivative formula
                }
            }
            return result;
        }
    };
}
