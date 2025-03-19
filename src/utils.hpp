
#pragma once
#include "matrix.hpp"
#include <random>
#include <algorithm>

namespace Utils {
    // Data generation utilities for training and testing
    template<typename T>
    class DataGenerator {
    public:
        // Generate synthetic XOR training/test data
        // Returns pair of input matrix and expected output matrix
        static std::pair<Matrix<T>, Matrix<T>> generateXORData(size_t samples) {
            Matrix<T> inputs(samples, 2);    // 2D inputs (operands)
            Matrix<T> outputs(samples, 1);    // 1D outputs (XOR result)
            
            // Initialize random number generator
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, 1);  // Binary distribution

            // Generate random binary inputs and compute XOR outputs
            for (size_t i = 0; i < samples; ++i) {
                bool a = dis(gen);  // First operand
                bool b = dis(gen);  // Second operand
                inputs.at(i, 0) = a;
                inputs.at(i, 1) = b;
                outputs.at(i, 0) = a != b;  // XOR operation
            }

            return {inputs, outputs};
        }
    };

    // Evaluation metrics for model performance
    template<typename T>
    class Metrics {
    public:
        // Calculate classification accuracy
        // Compares predicted values to expected values using threshold
        static T accuracy(const Matrix<T>& predicted, const Matrix<T>& expected, T threshold = 0.5) {
            size_t correct = 0;
            size_t total = predicted.getRows() * predicted.getCols();

            // Count correct predictions
            for (size_t i = 0; i < predicted.getRows(); ++i) {
                for (size_t j = 0; j < predicted.getCols(); ++j) {
                    // Convert continuous outputs to binary using threshold
                    bool pred = predicted.at(i, j) >= threshold;
                    bool exp = expected.at(i, j) >= threshold;
                    if (pred == exp) correct++;
                }
            }

            // Return percentage of correct predictions
            return static_cast<T>(correct) / total;
        }
    };
}
