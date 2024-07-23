#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <tuple>

struct LinearModel {
    double* weights;
    double* bias;
    double* loss;
    size_t loss_size;
};

// Déclaration des fonctions
double calculate_loss(const std::vector<double>& weights, const std::vector<double>& bias,
                      const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets,
                      bool isclassification);

std::vector<double> predict(const std::vector<double>& weights, const std::vector<double>& bias,
                            const std::vector<double>& input, bool isclassification);

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> train(const std::vector<std::vector<double>>& inputs,
                                                                                const std::vector<std::vector<double>>& targets,
                                                                                size_t k, double learning_rate,
                                                                                size_t n_features, size_t num_iterations,
                                                                                bool isclassification) {
    std::vector<double> weights(n_features * k, 0.0);
    std::vector<double> bias(k, 0.0);
    std::vector<double> loss_values;

    std::random_device rd;
    std::mt19937 gen(rd());

    for (size_t iteration = 0; iteration <= num_iterations; ++iteration) {
        std::uniform_int_distribution<> dis(0, inputs.size() - 1);
        size_t i = dis(gen);
        const auto& input = inputs[i];
        const auto& target = targets[i];

        std::vector<double> output(k, 0.0);

        for (size_t j = 0; j < n_features; ++j) {
            for (size_t c = 0; c < k; ++c) {
                output[c] += input[j] * weights[j + n_features * c];
            }
        }

        for (size_t c = 0; c < k; ++c) {
            output[c] += bias[c];
        }

        std::vector<double> errors(k, 0.0);

        for (size_t c = 0; c < k; ++c) {
            errors[c] = target[c] - output[c];
        }

        for (size_t j = 0; j < n_features; ++j) {
            for (size_t c = 0; c < k; ++c) {
                weights[j + n_features * c] += learning_rate * input[j] * errors[c];
            }
        }

        for (size_t c = 0; c < k; ++c) {
            bias[c] += learning_rate * errors[c];
        }

        if (iteration % 10000 == 0) {
            double loss = calculate_loss(weights, bias, inputs, targets, isclassification);
            loss_values.push_back(loss);
            std::cout << "Iteration: " << iteration << ", Average Loss: " << loss << std::endl;
        }
    }

    return {weights, bias, loss_values};
}

double calculate_loss(const std::vector<double>& weights, const std::vector<double>& bias,
                      const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets,
                      bool isclassification) {
    size_t k = bias.size();
    size_t n_samples = inputs.size();
    double loss = 0.0;

    for (size_t i = 0; i < n_samples; ++i) {
        const auto& input = inputs[i];
        const auto& target = targets[i];
        std::vector<double> prediction = predict(weights, bias, input, isclassification);

        for (size_t c = 0; c < k; ++c) {
            double error = target[c] - prediction[c];
            loss += error * error;
        }
    }

    return loss / (n_samples * k);
}

std::vector<double> predict(const std::vector<double>& weights, const std::vector<double>& bias,
                            const std::vector<double>& input, bool isclassification) {
    size_t k = bias.size();
    size_t n_features = input.size();
    std::vector<double> prediction(k, 0.0);

    for (size_t j = 0; j < n_features; ++j) {
        for (size_t c = 0; c < k; ++c) {
            prediction[c] += input[j] * weights[j + n_features * c];
        }
    }

    for (size_t c = 0; c < k; ++c) {
        prediction[c] += bias[c];
    }

    if (isclassification) {
        for (size_t c = 0; c < k; ++c) {
            prediction[c] = std::tanh(prediction[c]);
        }
    }

    return prediction;
}

extern "C" double* predict_linear_model(const double* features, const double* weights, const double* bias,
                                        size_t num_samples, size_t num_features, size_t k, bool isclassification) {
    std::vector<double> features_vec(features, features + num_samples * num_features);
    std::vector<double> weights_vec(weights, weights + num_features * k);
    std::vector<double> bias_vec(bias, bias + k);

    std::vector<double> inputs(features_vec.begin(), features_vec.begin() + num_features);

    std::vector<double> target = predict(weights_vec, bias_vec, inputs, isclassification);
    double* target_ptr = new double[target.size()];
    std::copy(target.begin(), target.end(), target_ptr);
    return target_ptr;
}

extern "C" LinearModel train_linear_model(const double* features, const double* outputs, size_t num_samples,
                                          size_t num_features, double learning_rate, size_t num_iterations,
                                          size_t k, bool isclassification) {
    std::vector<std::vector<double>> inputs(num_samples, std::vector<double>(num_features));
    std::vector<std::vector<double>> targets(num_samples, std::vector<double>(k));

    for (size_t i = 0; i < num_samples; ++i) {
        for (size_t j = 0; j < num_features; ++j) {
            inputs[i][j] = features[i * num_features + j];
        }
    }

    for (size_t i = 0; i < num_samples; ++i) {
        for (size_t j = 0; j < k; ++j) {
            targets[i][j] = outputs[i * k + j];
        }
    }

    auto [weights, bias, loss_values] = train(inputs, targets, k, learning_rate, num_features, num_iterations, isclassification);

    LinearModel model;
    model.weights = new double[weights.size()];
    model.bias = new double[bias.size()];
    model.loss = new double[loss_values.size()];
    model.loss_size = loss_values.size();


    std::copy(weights.begin(), weights.end(), model.weights);
    std::copy(bias.begin(), bias.end(), model.bias);
    std::copy(loss_values.begin(), loss_values.end(), model.loss);

    return model;
}
