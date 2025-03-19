#pragma once
#include "matrix.hpp"
#include <random>
#include <algorithm>

namespace Utils {
    template<typename T>
    class DataGenerator {
    public:
        static std::pair<Matrix<T>, Matrix<T>> generateXORData(size_t samples) {
            Matrix<T> inputs(samples, 2);
            Matrix<T> outputs(samples, 1);
            
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, 1);

            for (size_t i = 0; i < samples; ++i) {
                bool a = dis(gen);
                bool b = dis(gen);
                inputs.at(i, 0) = a;
                inputs.at(i, 1) = b;
                outputs.at(i, 0) = a != b;
            }

            return {inputs, outputs};
        }
    };

    template<typename T>
    class Metrics {
    public:
        static T accuracy(const Matrix<T>& predicted, const Matrix<T>& expected, T threshold = 0.5) {
            size_t correct = 0;
            size_t total = predicted.getRows() * predicted.getCols();

            for (size_t i = 0; i < predicted.getRows(); ++i) {
                for (size_t j = 0; j < predicted.getCols(); ++j) {
                    bool pred = predicted.at(i, j) >= threshold;
                    bool exp = expected.at(i, j) >= threshold;
                    if (pred == exp) correct++;
                }
            }

            return static_cast<T>(correct) / total;
        }
    };
}
