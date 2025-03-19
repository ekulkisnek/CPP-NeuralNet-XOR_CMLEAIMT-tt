#pragma once
#include <cmath>
#include "matrix.hpp"

namespace Activation {
    template<typename T>
    class ActivationFunction {
    public:
        virtual Matrix<T> forward(const Matrix<T>& x) const = 0;
        virtual Matrix<T> backward(const Matrix<T>& x) const = 0;
        virtual ~ActivationFunction() = default;
    };

    template<typename T>
    class ReLU : public ActivationFunction<T> {
    public:
        Matrix<T> forward(const Matrix<T>& x) const override {
            Matrix<T> result(x.getRows(), x.getCols());
            for (size_t i = 0; i < x.getRows(); ++i) {
                for (size_t j = 0; j < x.getCols(); ++j) {
                    result.at(i, j) = std::max(static_cast<T>(0), x.at(i, j));
                }
            }
            return result;
        }

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

    template<typename T>
    class Sigmoid : public ActivationFunction<T> {
    public:
        Matrix<T> forward(const Matrix<T>& x) const override {
            Matrix<T> result(x.getRows(), x.getCols());
            for (size_t i = 0; i < x.getRows(); ++i) {
                for (size_t j = 0; j < x.getCols(); ++j) {
                    result.at(i, j) = 1.0 / (1.0 + std::exp(-x.at(i, j)));
                }
            }
            return result;
        }

        Matrix<T> backward(const Matrix<T>& x) const override {
            Matrix<T> sig = forward(x);
            Matrix<T> result(x.getRows(), x.getCols());
            for (size_t i = 0; i < x.getRows(); ++i) {
                for (size_t j = 0; j < x.getCols(); ++j) {
                    T s = sig.at(i, j);
                    result.at(i, j) = s * (1 - s);
                }
            }
            return result;
        }
    };
}
