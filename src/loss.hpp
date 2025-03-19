#pragma once
#include "matrix.hpp"

namespace Loss {
    template<typename T>
    class LossFunction {
    public:
        virtual T calculate(const Matrix<T>& predicted, const Matrix<T>& expected) const = 0;
        virtual Matrix<T> derivative(const Matrix<T>& predicted, const Matrix<T>& expected) const = 0;
        virtual ~LossFunction() = default;
    };

    template<typename T>
    class MSE : public LossFunction<T> {
    public:
        T calculate(const Matrix<T>& predicted, const Matrix<T>& expected) const override {
            T sum = 0;
            for (size_t i = 0; i < predicted.getRows(); ++i) {
                for (size_t j = 0; j < predicted.getCols(); ++j) {
                    T diff = predicted.at(i, j) - expected.at(i, j);
                    sum += diff * diff;
                }
            }
            return sum / (predicted.getRows() * predicted.getCols());
        }

        Matrix<T> derivative(const Matrix<T>& predicted, const Matrix<T>& expected) const override {
            return (predicted - expected) * (2.0 / (predicted.getRows() * predicted.getCols()));
        }
    };
}
