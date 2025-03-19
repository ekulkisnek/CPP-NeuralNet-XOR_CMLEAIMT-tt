
#include <iostream>
#include <memory>
#include "neural_network.hpp"
#include "utils.hpp"

int main() {
    // Define the core network parameters as constants
    using T = double;  // Use double precision floating point for all calculations
    const size_t TRAIN_SAMPLES = 1000;  // Number of training examples to generate
    const size_t TEST_SAMPLES = 100;    // Number of test examples to evaluate on
    const T LEARNING_RATE = 0.1;        // Step size for gradient descent
    const size_t EPOCHS = 1000;         // Number of complete passes through the training data

    // Initialize the neural network with Mean Squared Error loss function
    // Using smart pointer for automatic memory management
    auto nn = std::make_shared<NeuralNetwork<T>>(std::make_shared<Loss::MSE<T>>());

    // Build the network architecture:
    // Layer 1: Input(2) -> Hidden(4) with ReLU activation
    // This layer transforms 2D input (XOR operands) into 4D intermediate representation
    nn->addLayer(std::make_shared<Layer<T>>(2, 4, std::make_shared<Activation::ReLU<T>>()));
    
    // Layer 2: Hidden(4) -> Output(1) with Sigmoid activation
    // This layer reduces 4D hidden state to single output probability
    nn->addLayer(std::make_shared<Layer<T>>(4, 1, std::make_shared<Activation::Sigmoid<T>>()));

    // Generate synthetic XOR training data
    // train_x: Matrix of input pairs (e.g., [0,1], [1,0])
    // train_y: Matrix of expected outputs (e.g., [1], [1])
    auto [train_x, train_y] = Utils::DataGenerator<T>::generateXORData(TRAIN_SAMPLES);

    // Begin training process
    std::cout << "Training started..." << std::endl;
    for (size_t epoch = 0; epoch < EPOCHS; ++epoch) {
        T total_loss = 0;

        // Perform one complete pass through training data
        // Updates weights and biases using backpropagation
        total_loss = nn->train(train_x, train_y, LEARNING_RATE);

        // Print progress every 100 epochs
        if ((epoch + 1) % 100 == 0) {
            std::cout << "Epoch " << epoch + 1 << "/" << EPOCHS 
                      << ", Loss: " << total_loss << std::endl;
        }
    }

    // Generate separate test dataset for evaluation
    auto [test_x, test_y] = Utils::DataGenerator<T>::generateXORData(TEST_SAMPLES);

    // Evaluate model performance
    Matrix<T> predictions = nn->predict(test_x);  // Forward pass without training
    T accuracy = Utils::Metrics<T>::accuracy(predictions, test_y);  // Calculate accuracy

    // Display test results
    std::cout << "\nTest Results:" << std::endl;
    std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;

    // Show example predictions for visual verification
    std::cout << "\nExample Predictions:" << std::endl;
    for (size_t i = 0; i < 5; ++i) {
        std::cout << "Input: [" << test_x.at(i, 0) << ", " << test_x.at(i, 1) 
                  << "] Expected: " << test_y.at(i, 0)
                  << " Predicted: " << predictions.at(i, 0) << std::endl;
    }

    return 0;
}
