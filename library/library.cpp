#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <algorithm>

#ifdef WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

extern "C" {
class MyMLP {
private:
    std::vector<int64_t> d;
    std::vector<std::vector<std::vector<double>>> W;
    std::vector<std::vector<double>> X, deltas;
    int64_t L;
    double alpha;

    double tanh_derivative(double x) {
        return 1.0 - x * x;
    }

public:
    MyMLP(const std::vector<int64_t> &npl) : d(npl), L(npl.size() - 1), alpha(0.01) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);

        // Initialize weights
        W.resize(L + 1);
        for (int64_t l = 1; l <= L; ++l) {
            W[l].resize(d[l - 1] + 1, std::vector<double>(d[l] + 1));
            for (int64_t i = 0; i <= d[l - 1]; ++i) {
                for (int64_t j = 1; j <= d[l]; ++j) {
                    W[l][i][j] = dis(gen);
                }
            }
        }

        X.resize(L + 1);
        deltas.resize(L + 1);
        for (int64_t l = 0; l <= L; ++l) {
            X[l].resize(d[l] + 1, 0.0);
            X[l][0] = 1.0; // Bias neuron
            deltas[l].resize(d[l] + 1, 0.0);
        }
    }

    void propagate(const std::vector<double> &sample_inputs, bool is_classification) {
        for (size_t j = 0; j < sample_inputs.size(); ++j) {
            X[0][j + 1] = sample_inputs[j];
        }

        for (int64_t l = 1; l <= L; ++l) {
            for (int64_t j = 1; j <= d[l]; ++j) {
                double total = 0.0;
                for (int64_t i = 0; i <= d[l - 1]; ++i) {
                    total += W[l][i][j] * X[l - 1][i];
                }
                X[l][j] = is_classification || l < L ? std::tanh(total) : total;
            }
        }
    }

    std::vector<double> predict(const std::vector<double> &sample_inputs, bool is_classification) {
        propagate(sample_inputs, is_classification);
        return std::vector<double>(X[L].begin() + 1, X[L].end());
    }

    void backpropagate(const std::vector<double> &sample_expected_outputs, bool is_classification) {
        // Output layer error
        for (int64_t j = 1; j <= d[L]; ++j) {
            double output = X[L][j];
            double target = sample_expected_outputs[j - 1];
            deltas[L][j] = output - target;
            if (is_classification) {
                deltas[L][j] *= tanh_derivative(output);
            }
        }

        // Hidden layer errors
        for (int64_t l = L - 1; l >= 1; --l) {
            for (int64_t i = 1; i <= d[l]; ++i) {
                double error = 0.0;
                for (int64_t j = 1; j <= d[l + 1]; ++j) {
                    error += W[l + 1][i][j] * deltas[l + 1][j];
                }
                deltas[l][i] = error * tanh_derivative(X[l][i]);
            }
        }

        // Update weights
        for (int64_t l = 1; l <= L; ++l) {
            for (int64_t i = 0; i <= d[l - 1]; ++i) {
                for (int64_t j = 1; j <= d[l]; ++j) {
                    W[l][i][j] -= alpha * deltas[l][j] * X[l - 1][i];
                }
            }
        }
    }

    void train(const std::vector<std::vector<double>> &all_samples_inputs,
               const std::vector<std::vector<double>> &all_samples_expected_outputs,
               double alpha, int64_t nb_iter, bool is_classification) {
        this->alpha = alpha;
        int64_t num_samples = all_samples_inputs.size();

        for (int64_t it = 0; it < nb_iter; ++it) {
            int64_t k = std::rand() % num_samples;
            const std::vector<double> &sample_inputs = all_samples_inputs[k];
            const std::vector<double> &sample_expected_outputs = all_samples_expected_outputs[k];

            propagate(sample_inputs, is_classification);
            backpropagate(sample_expected_outputs, is_classification);
        }
    }

    const std::vector<int64_t>& get_d() const { return d; }
    int64_t get_L() const { return L; }
    std::vector<std::vector<std::vector<double>>>& get_W() { return W; }
    std::vector<std::vector<double>>& get_X() { return X; }
};

DLLEXPORT MyMLP *create_mlp(const int64_t *layer_sizes, int64_t num_layers) {
    std::vector<int64_t> layers(layer_sizes, layer_sizes + num_layers);
    return new MyMLP(layers);
}

DLLEXPORT void destroy_mlp(MyMLP *mlp) {
    delete mlp;
}

DLLEXPORT void predict(MyMLP *mlp, double *inputs, int64_t num_inputs, double *output, bool is_classification) {
    std::vector<double> input_vector(inputs, inputs + num_inputs);
    std::vector<double> results = mlp->predict(input_vector, is_classification);
    std::copy(results.begin(), results.end(), output);
}

DLLEXPORT void train(MyMLP *mlp, const double *inputs, int64_t num_samples, int64_t input_size, const double *outputs,
                     int64_t output_size, double alpha, int64_t iterations, bool is_classification) {
    std::vector<std::vector<double>> training_inputs(num_samples, std::vector<double>(input_size));
    std::vector<std::vector<double>> training_outputs(num_samples, std::vector<double>(output_size));

    for (int64_t i = 0; i < num_samples; ++i) {
        std::copy(inputs + i * input_size, inputs + (i + 1) * input_size, training_inputs[i].begin());
        std::copy(outputs + i * output_size, outputs + (i + 1) * output_size, training_outputs[i].begin());
    }

    mlp->train(training_inputs, training_outputs, alpha, iterations, is_classification);
}
}