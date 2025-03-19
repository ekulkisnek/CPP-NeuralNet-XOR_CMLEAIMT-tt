
#pragma once
#include <vector>
#include <stdexcept>
#include <random>
#include <iostream>
#include <iomanip>

// Matrix class: Core data structure for neural network operations
// Implements essential matrix operations required for forward and backward propagation
template<typename T>
class Matrix {
private:
    std::vector<std::vector<T>> data;  // 2D vector storing matrix elements
    size_t rows;                       // Number of matrix rows
    size_t cols;                       // Number of matrix columns

public:
    // Constructor: Initialize matrix with specified dimensions
    Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
        data.resize(rows, std::vector<T>(cols));
    }

    // Constructor: Initialize matrix from existing 2D vector
    Matrix(const std::vector<std::vector<T>>& input) : data(input) {
        rows = input.size();
        cols = input[0].size();
    }

    // Initialize matrix with random values in range [min, max]
    // Used for weight initialization in neural network layers
    void randomize(T min = -1, T max = 1) {
        std::random_device rd;  // Hardware random number generator
        std::mt19937 gen(rd());  // Mersenne Twister PRNG
        std::uniform_real_distribution<T> dis(min, max);  // Uniform distribution

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                data[i][j] = dis(gen);
            }
        }
    }

    // Matrix multiplication (dot product)
    // Essential for forward propagation (input * weights)
    // and backward propagation (error * weights_transpose)
    Matrix<T> dot(const Matrix<T>& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions don't match for multiplication");
        }

        Matrix<T> result(rows, other.cols);
        // Perform matrix multiplication: result[i,j] = sum(this[i,k] * other[k,j])
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                T sum = 0;
                for (size_t k = 0; k < cols; ++k) {
                    sum += data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }

    // Element-wise multiplication (Hadamard product)
    // Used in backward propagation when computing gradients
    Matrix<T> hadamard(const Matrix<T>& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions don't match for hadamard product");
        }

        Matrix<T> result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.data[i][j] = data[i][j] * other.data[i][j];
            }
        }
        return result;
    }

    // Matrix transpose operation
    // Used in backward propagation to propagate errors
    Matrix<T> transpose() const {
        Matrix<T> result(cols, rows);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.data[j][i] = data[i][j];
            }
        }
        return result;
    }

    // Element-wise matrix addition
    // Used in updating weights: weights = weights + learning_rate * gradient
    Matrix<T> operator+(const Matrix<T>& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions don't match for addition");
        }

        Matrix<T> result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.data[i][j] = data[i][j] + other.data[i][j];
            }
        }
        return result;
    }

    // Element-wise matrix subtraction
    // Used in computing error terms: error = predicted - expected
    Matrix<T> operator-(const Matrix<T>& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions don't match for subtraction");
        }

        Matrix<T> result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.data[i][j] = data[i][j] - other.data[i][j];
            }
        }
        return result;
    }

    // Scalar multiplication
    // Used in scaling gradients by learning rate
    Matrix<T> operator*(T scalar) const {
        Matrix<T> result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.data[i][j] = data[i][j] * scalar;
            }
        }
        return result;
    }

    // Access methods for matrix elements
    T& at(size_t i, size_t j) { return data[i][j]; }
    const T& at(size_t i, size_t j) const { return data[i][j]; }
    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }

    // Debug utility: Print matrix contents
    void print() const {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(4) << data[i][j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
};
