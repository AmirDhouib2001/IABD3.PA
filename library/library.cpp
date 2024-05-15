#include <vector>
#include <cmath>
#include <random>
#include <iostream>

#ifdef WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

extern "C" {
class MyMLP {
private:
    std::vector<int> d;
    std::vector<std::vector<std::vector<double>>> W;
    std::vector<std::vector<double>> X, deltas;
    int L;

    double tanh_derivative(double x) {
        return 1.0 - x * x;
    }

public:
    MyMLP(const std::vector<int> &npl) : d(npl), L(npl.size() - 1) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);

        // Initialize weights
        W.resize(L + 1);
        for (int l = 1; l <= L; ++l) {
            W[l].resize(d[l - 1] + 1, std::vector<double>(d[l] + 1));
            for (int i = 0; i <= d[l - 1]; ++i) {
                for (int j = 1; j <= d[l]; ++j) {
                    W[l][i][j] = dis(gen);
                }
            }
        }

        X.resize(L + 1);
        deltas.resize(L + 1);
        for (int l = 0; l <= L; ++l) {
            X[l].resize(d[l] + 1, 0.0);
            X[l][0] = 1.0; // Bias neuron
            deltas[l].resize(d[l] + 1, 0.0);
        }
    }

    void propagate(const std::vector<double> &sample_inputs, bool is_classification) {
        for (size_t j = 0; j < sample_inputs.size(); ++j) {
            X[0][j + 1] = sample_inputs[j];
        }

        for (int l = 1; l <= L; ++l) {
            for (int j = 1; j <= d[l]; ++j) {
                double total = 0.0;
                for (int i = 0; i <= d[l - 1]; ++i) {
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

    const std::vector<int>& get_d() const { return d; }
    int get_L() const { return L; }
    std::vector<std::vector<std::vector<double>>>& get_W() { return W; }
    std::vector<std::vector<double>>& get_X() { return X; }
};

DLLEXPORT MyMLP *create_mlp(const int *layer_sizes, int num_layers) {
    std::vector<int> layers(layer_sizes, layer_sizes + num_layers);
    return new MyMLP(layers);
}

DLLEXPORT void destroy_mlp(MyMLP *mlp) {
    delete mlp;
}

DLLEXPORT void predict(MyMLP *mlp, double *inputs, int num_inputs, double *output, bool is_classification) {
    std::vector<double> input_vector(inputs, inputs + num_inputs);
    std::vector<double> results = mlp->predict(input_vector, is_classification);
    std::copy(results.begin(), results.end(), output);
}

DLLEXPORT void train(MyMLP *mlp, const double *inputs, int num_samples, int input_size, const double *outputs,
                     int output_size, double alpha, int iterations, bool is_classification) {
    std::vector<std::vector<double> > training_inputs(num_samples, std::vector<double>(input_size));
    std::vector<std::vector<double> > training_outputs(num_samples, std::vector<double>(output_size));

    for (int i = 0; i < num_samples; i++) {
        std::copy(inputs + i * input_size, inputs + (i + 1) * input_size, training_inputs[i].begin());
        std::copy(outputs + i * output_size, outputs + (i + 1) * output_size, training_outputs[i].begin());
    }

    for (int it = 0; it < iterations; ++it) {
        for (int idx = 0; idx < num_samples; ++idx) {
            mlp->propagate(training_inputs[idx], is_classification);


            for (size_t j = 1; j <= mlp->get_d()[mlp->get_L()]; ++j) {
                double error = training_outputs[idx][j - 1] - mlp->get_X()[mlp->get_L()][j];
                for (size_t i = 0; i <= mlp->get_d()[mlp->get_L() - 1]; ++i) {
                    mlp->get_W()[mlp->get_L()][i][j] += alpha * error * mlp->get_X()[mlp->get_L() - 1][i];
                }
            }
        }
    }
}
}
