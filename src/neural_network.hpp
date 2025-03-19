#pragma once
#include "layer.hpp"
#include "loss.hpp"
#include <vector>
#include <memory>

template<typename T>
class NeuralNetwork {
private:
    std::vector<std::shared_ptr<Layer<T>>> layers;
    std::shared_ptr<Loss::LossFunction<T>> loss_function;

public:
    NeuralNetwork(std::shared_ptr<Loss::LossFunction<T>> loss) 
        : loss_function(loss) {}

    void addLayer(std::shared_ptr<Layer<T>> layer) {
        layers.push_back(layer);
    }

    Matrix<T> forward(const Matrix<T>& input) {
        Matrix<T> current = input;
        for (auto& layer : layers) {
            current = layer->forward(current);
        }
        return current;
    }

    T backward(const Matrix<T>& expected, T learning_rate) {
        Matrix<T> error = loss_function->derivative(layers.back()->getOutput(), expected);
        
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            error = (*it)->backward(error, learning_rate);
        }

        return loss_function->calculate(layers.back()->getOutput(), expected);
    }

    T train(const Matrix<T>& input, const Matrix<T>& expected, T learning_rate) {
        forward(input);
        return backward(expected, learning_rate);
    }

    Matrix<T> predict(const Matrix<T>& input) {
        return forward(input);
    }
};
