#include <iostream>
#include <memory>
#include "neural_network.hpp"
#include "utils.hpp"

int main() {
    using T = double;
    const size_t TRAIN_SAMPLES = 1000;
    const size_t TEST_SAMPLES = 100;
    const T LEARNING_RATE = 0.1;
    const size_t EPOCHS = 1000;

    // Create neural network
    auto nn = std::make_shared<NeuralNetwork<T>>(std::make_shared<Loss::MSE<T>>());

    // Add layers
    nn->addLayer(std::make_shared<Layer<T>>(2, 4, std::make_shared<Activation::ReLU<T>>()));
    nn->addLayer(std::make_shared<Layer<T>>(4, 1, std::make_shared<Activation::Sigmoid<T>>()));

    // Generate training data
    auto [train_x, train_y] = Utils::DataGenerator<T>::generateXORData(TRAIN_SAMPLES);

    // Training loop
    std::cout << "Training started..." << std::endl;
    for (size_t epoch = 0; epoch < EPOCHS; ++epoch) {
        T total_loss = 0;

        total_loss = nn->train(train_x, train_y, LEARNING_RATE);

        if ((epoch + 1) % 100 == 0) {
            std::cout << "Epoch " << epoch + 1 << "/" << EPOCHS 
                      << ", Loss: " << total_loss << std::endl;
        }
    }

    // Generate test data
    auto [test_x, test_y] = Utils::DataGenerator<T>::generateXORData(TEST_SAMPLES);

    // Evaluate model
    Matrix<T> predictions = nn->predict(test_x);
    T accuracy = Utils::Metrics<T>::accuracy(predictions, test_y);

    std::cout << "\nTest Results:" << std::endl;
    std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;

    // Example predictions
    std::cout << "\nExample Predictions:" << std::endl;
    for (size_t i = 0; i < 5; ++i) {
        std::cout << "Input: [" << test_x.at(i, 0) << ", " << test_x.at(i, 1) 
                  << "] Expected: " << test_y.at(i, 0)
                  << " Predicted: " << predictions.at(i, 0) << std::endl;
    }

    return 0;
}