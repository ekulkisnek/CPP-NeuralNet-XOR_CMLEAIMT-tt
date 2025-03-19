
#pragma once
#include "matrix.hpp"

namespace Loss {
    // Base class for loss functions
    // Defines interface for loss calculation and derivatives
    template<typename T>
    class LossFunction {
    public:
        // Calculate loss value between predicted and expected outputs
        virtual T calculate(const Matrix<T>& predicted, const Matrix<T>& expected) const = 0;
        
        // Calculate loss derivative for backpropagation
        virtual Matrix<T> derivative(const Matrix<T>& predicted, const Matrix<T>& expected) const = 0;
        
        virtual ~LossFunction() = default;
    };

    // Mean Squared Error (MSE) loss function
    // L = 1/n * Σ(y - ŷ)²
    template<typename T>
    class MSE : public LossFunction<T> {
    public:
        // Calculate MSE loss
        // Average squared difference between predicted and expected values
        T calculate(const Matrix<T>& predicted, const Matrix<T>& expected) const override {
            T sum = 0;
            for (size_t i = 0; i < predicted.getRows(); ++i) {
                for (size_t j = 0; j < predicted.getCols(); ++j) {
                    T diff = predicted.at(i, j) - expected.at(i, j);
                    sum += diff * diff;  // Square the difference
                }
            }
            // Return average error across all elements
            return sum / (predicted.getRows() * predicted.getCols());
        }

        // Calculate MSE derivative
        // d/dx(MSE) = 2/n * (predicted - expected)
        Matrix<T> derivative(const Matrix<T>& predicted, const Matrix<T>& expected) const override {
            // Compute element-wise difference and scale by 2/n
            return (predicted - expected) * (2.0 / (predicted.getRows() * predicted.getCols()));
        }
    };
}
