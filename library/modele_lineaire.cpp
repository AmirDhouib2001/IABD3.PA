#include <vector>
#include <random>
#include <iostream>
#include <algorithm>

#ifdef WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

extern "C" {
class modele_lineaire {
private:
    std::vector<double> W;
    double b;
    double alpha;

public:
    modele_lineaire(int64_t input_size) : W(input_size), b(0.0), alpha(0.01) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);

        // Initialize weights
        for (auto& weight : W) {
            weight = dis(gen);
        }
        b = dis(gen);  // Initialize bias
    }

    std::vector<double> predict(const std::vector<double> &inputs) {
        double total = b;
        for (size_t i = 0; i < inputs.size(); ++i) {
            total += W[i] * inputs[i];
        }
        return { total };
    }

    void train(const std::vector<std::vector<double>> &all_samples_inputs,
               const std::vector<double> &all_samples_outputs,
               double alpha, int64_t nb_iter) {
        this->alpha = alpha;
        int64_t num_samples = all_samples_inputs.size();

        for (int64_t it = 0; it < nb_iter; ++it) {
            int64_t k = std::rand() % num_samples;
            const std::vector<double> &sample_inputs = all_samples_inputs[k];
            double expected_output = all_samples_outputs[k];

            // Predict
            double prediction = predict(sample_inputs)[0];

            // Compute error
            double error = prediction - expected_output;

            // Update
            for (size_t i = 0; i < W.size(); ++i) {
                W[i] -= alpha * error * sample_inputs[i];
            }
            b -= alpha * error;
        }
    }
};

DLLEXPORT modele_lineaire *create_linear_model(int64_t input_size) {
    return new modele_lineaire(input_size);
}

DLLEXPORT void destroy_linear_model(modele_lineaire *model) {
    delete model;
}

DLLEXPORT void predict(modele_lineaire *model, double *inputs, int64_t num_inputs, double *output) {
    std::vector<double> input_vector(inputs, inputs + num_inputs);
    std::vector<double> results = model->predict(input_vector);
    std::copy(results.begin(), results.end(), output);
}

DLLEXPORT void train(modele_lineaire *model, const double *inputs, int64_t num_samples, int64_t input_size, const double *outputs,
                     double alpha, int64_t iterations) {
    std::vector<std::vector<double>> training_inputs(num_samples, std::vector<double>(input_size));
    std::vector<double> training_outputs(num_samples);

    for (int64_t i = 0; i < num_samples; ++i) {
        std::copy(inputs + i * input_size, inputs + (i + 1) * input_size, training_inputs[i].begin());
        training_outputs[i] = outputs[i];
    }

    model->train(training_inputs, training_outputs, alpha, iterations);
}
}
