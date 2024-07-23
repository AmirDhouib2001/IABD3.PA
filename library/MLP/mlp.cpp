#include <vector>
#include <random>
#include <iostream>
#include <string>
#include <cstring>
#include <cmath>
#include <fstream>


// Définition de la structure du modèle MLP
class MyMLP {
public:
    std::vector<int64_t> d;
    int64_t nbr_layers;
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> outputs_values;
    std::vector<std::vector<double>> deltas;
    std::vector<double> loss_values;

    MyMLP(const std::vector<int64_t>& npl) {
        // Copier les dimensions du MLP dans un nouveau vecteur d
        d = npl;
        // Nbr de couches du MLP
        nbr_layers = npl.size() - 1;

        weights.resize(nbr_layers + 1);
        for (int64_t l = 0; l <= nbr_layers; ++l) {
            if (l == 0) continue;
            weights[l].resize(npl[l - 1] + 1, std::vector<double>(npl[l] + 1));
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(-1.0, 1.0);
            for (int64_t i = 0; i <= npl[l - 1]; ++i) {
                for (int64_t j = 0; j <= npl[l]; ++j) {
                    weights[l][i][j] = (j == 0) ? 0.0 : dis(gen);
                }
            }
        }

        // Création de l'espace mémoire pour stocker plus tard les valeurs de sorties de chaque neurone
        outputs_values.resize(nbr_layers + 1);
        for (int64_t l = 0; l <= nbr_layers; ++l) {
            outputs_values[l].resize(npl[l] + 1, (l == 0) ? 1.0 : -1.0);
        }

        // Création de l'espace mémoire pour stocker plus tard les semi-gradients associés à chaque neurone
        deltas.resize(nbr_layers + 1);
        for (int64_t l = 0; l <= nbr_layers; ++l) {
            deltas[l].resize(npl[l] + 1, 0.0);
        }
    }

    // Méthode interne pour effectuer la propagation avant et mettre à jour les valeurs de sortie de chaque neurone à partir des entrées d'un exemple
    void propagate(const std::vector<double>& inputs, bool is_classification) {
        if (inputs.size() != static_cast<size_t>(d[0])) {
            std::cerr << "Error: Input size does not match the expected input dimension. Expected: " << d[0] << ", Got: " << inputs.size() << std::endl;
            return;
        }

        // Copier les entrées (inputs) dans la couche d'entrée du modèle
        for (int64_t j = 0; j < d[0]; ++j) {
            outputs_values[0][j + 1] = inputs[j];
        }

        // Mise à jour récursive des valeurs de sortie des neurones, couche après couche
        for (int64_t l = 1; l <= nbr_layers; ++l) {
            for (int64_t j = 1; j <= d[l]; ++j) {
                double total = 0.0;
                for (int64_t i = 0; i <= d[l - 1]; ++i) {
                    total += weights[l][i][j] * outputs_values[l - 1][i];
                }
                if (l <= nbr_layers || is_classification) {
                    total = tanh(total);
                }
                outputs_values[l][j] = total;
            }
        }
    }

    std::vector<double> predict(const std::vector<double>& inputs, bool is_classification) {
        propagate(inputs, is_classification);
        std::vector<double> outputs(outputs_values[nbr_layers].begin() + 1, outputs_values[nbr_layers].begin() + 1 + d[nbr_layers]);
        std::cout << "Predict: Outputs size: " << outputs.size() << std::endl;
        return outputs;
    }

    static double calculate_loss(const std::vector<double>& outputs, const std::vector<double>& expected_outputs) {
        if (outputs.size() != expected_outputs.size()) {
            std::cerr << "Error: Output size does not match the expected output size. Outputs size: " << outputs.size() << ", Expected size: " << expected_outputs.size() << std::endl;
            return -1.0;
        }

        double loss = 0.0;
        for (size_t i = 0; i < outputs.size(); ++i) {
            loss += pow(outputs[i] - expected_outputs[i], 2);
        }
        return loss / (2.0 * outputs.size());
    }

    // Méthode pour entraîner le modèle à partir d'un dataset étiqueté
    void train(const std::vector<std::vector<double>>& all_samples_inputs,
               const std::vector<std::vector<double>>& all_samples_expected_outputs,
               bool is_classification,
               int64_t iteration_count,
               double alpha) {
        if (all_samples_inputs.size() != all_samples_expected_outputs.size()) {
            std::cerr << "Error: Number of samples does not match number of expected outputs." << std::endl;
            return;
        }

        for (int64_t it = 0; it < iteration_count; ++it) {
            int64_t k = rand() % all_samples_inputs.size();
            const std::vector<double>& inputs_k = all_samples_inputs[k];
            const std::vector<double>& y_k = all_samples_expected_outputs[k];

            propagate(inputs_k, is_classification);
            // Calcul des deltas pour la couche de sortie

            for (int64_t j = 1; j <= d[nbr_layers]; ++j) {
                deltas[nbr_layers][j] = outputs_values[nbr_layers][j] - y_k[j - 1];
                if (is_classification) {
                    deltas[nbr_layers][j] *= 1.0 - pow(outputs_values[nbr_layers][j], 2);
                }
            }
            // Propagation des deltas en arrière à travers les couches du réseau

            for (int64_t l = nbr_layers; l > 0; --l) {
                for (int64_t i = 1; i <= d[l - 1]; ++i) {
                    double total = 0.0;
                    for (int64_t j = 1; j <= d[l]; ++j) {
                        total += weights[l][i][j] * deltas[l][j];
                    }
                    deltas[l - 1][i] = (1.0 - pow(outputs_values[l - 1][i], 2)) * total;
                }
            }
            // Mise à jour des poids en utilisant les deltas et le taux d'apprentissage

            for (int64_t l = 1; l <= nbr_layers; ++l) {
                for (int64_t i = 0; i <= d[l - 1]; ++i) {
                    for (int64_t j = 1; j <= d[l]; ++j) {
                        weights[l][i][j] -= alpha * outputs_values[l - 1][i] * deltas[l][j];
                    }
                }
            }
            // Calcul de la perte moyenne à intervalles réguliers et à la fin de l'entraînement

            if (it % 20000 == 0 || it == (iteration_count - 1)) {
                double total_loss = 0.0;
                for (size_t i = 0; i < all_samples_inputs.size(); ++i) {
                    propagate(all_samples_inputs[i], is_classification);
                    double example_loss = MyMLP::calculate_loss(
                        std::vector<double>(outputs_values[nbr_layers].begin() + 1, outputs_values[nbr_layers].begin() + 1 + d[nbr_layers]),
                        all_samples_expected_outputs[i]
                    );
                    if (example_loss < 0) {
                        std::cerr << "Iteration: " << it << ", Invalid loss encountered. Skipping this sample." << std::endl;
                        continue;
                    }
                    total_loss += example_loss;
                }
                double average_loss = total_loss / all_samples_inputs.size();
                loss_values.push_back(average_loss);
                std::cout << "Iteration: " << it << ", Average Loss: " << average_loss << std::endl;
            }
        }
    }
void save_model(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    for (int64_t l = 1; l <= nbr_layers; ++l) {
        for (int64_t i = 0; i <= d[l - 1]; ++i) {
            for (int64_t j = 1; j <= d[l]; ++j) {
                file.write(reinterpret_cast<char*>(&weights[l][i][j]), sizeof(double));
            }
        }
    }
    file.close();
}

void load_model(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file for reading: " << filename << std::endl;
        return;
    }

    for (int64_t l = 1; l <= nbr_layers; ++l) {
        for (int64_t i = 0; i <= d[l - 1]; ++i) {
            for (int64_t j = 1; j <= d[l]; ++j) {
                file.read(reinterpret_cast<char*>(&weights[l][i][j]), sizeof(double));
            }
        }
    }

    file.close();
}
};

// Fonctions d'interface C pour communiquer avec le modèle depuis d'autres langages
extern "C" {

#ifdef _WIN32
    #define DLL_EXPORT __declspec(dllexport)
#else
    #define DLL_EXPORT
#endif

DLL_EXPORT MyMLP* create_mlp(const int64_t* npl, size_t npl_len) {
    std::vector<int64_t> npl_vec(npl, npl + npl_len);
    std::cout << "Creating MLP with layers: ";
    for (size_t i = 0; i < npl_len; ++i) {
        std::cout << npl_vec[i] << " ";
    }
    std::cout << std::endl;
    return new MyMLP(npl_vec);
}

DLL_EXPORT void train_mlp(MyMLP* mlp, const double* all_samples_inputs, size_t all_samples_inputs_row_len,
                          size_t input_dim, const double* all_samples_outputs, size_t output_dim,
                          double alpha, int64_t iteration_count, bool is_classification) {
    // Debugging information
    std::cout << "Starting training..." << std::endl;
    std::cout << "Number of training samples: " << all_samples_inputs_row_len << std::endl;
    std::cout << "Input dimension: " << input_dim << std::endl;
    std::cout << "Output dimension: " << output_dim << std::endl;
    std::cout << "Iteration count: " << iteration_count << std::endl;

    std::vector<std::vector<double>> inputs(all_samples_inputs_row_len, std::vector<double>(input_dim));
    for (size_t i = 0; i < all_samples_inputs_row_len; ++i) {
        for (size_t j = 0; j < input_dim; ++j) {
            inputs[i][j] = all_samples_inputs[i * input_dim + j];
        }
    }

    std::vector<std::vector<double>> outputs(all_samples_inputs_row_len, std::vector<double>(output_dim));
    for (size_t i = 0; i < all_samples_inputs_row_len; ++i) {
        for (size_t j = 0; j < output_dim; ++j) {
            outputs[i][j] = all_samples_outputs[i * output_dim + j];
        }
    }

    std::cout << "Data prepared, starting the training process..." << std::endl;
    mlp->train(inputs, outputs, is_classification, iteration_count, alpha);
    std::cout << "Training completed." << std::endl;
}

DLL_EXPORT double* mlp_predict(MyMLP* mlp, const double* inputs, size_t inputs_row_len, size_t input_dim, bool is_classification) {
    if (inputs_row_len != 1) {
        std::cerr << "Error: inputs_row_len should be 1 but got " << inputs_row_len << std::endl;
        return nullptr;
    }
    if (input_dim != mlp->d[0]) {
        std::cerr << "Error: Input dimension does not match the expected input dimension. Expected: " << mlp->d[0] << ", Got: " << input_dim << std::endl;
        return nullptr;
    }

    std::vector<double> inputs_vec(inputs, inputs + input_dim);
    std::vector<double> outputs = mlp->predict(inputs_vec, is_classification);
    double* result = new double[outputs.size()];
    std::copy(outputs.begin(), outputs.end(), result);
    std::cout << "Prediction completed with result size: " << outputs.size() << std::endl;
    return result;
}

DLL_EXPORT void mlp_free(void* ptr) {
    delete[] static_cast<double*>(ptr);
}
// Ajout dans le bloc extern "C"
DLL_EXPORT const double* get_loss_values(MyMLP* mlp, size_t* length) {
    *length = mlp->loss_values.size();
    double* loss_array = new double[*length];
    std::copy(mlp->loss_values.begin(), mlp->loss_values.end(), loss_array);
    return loss_array;
}
DLL_EXPORT void save_mlp(MyMLP* mlp, const char* filename) {
mlp->save_model(filename);
}

DLL_EXPORT void load_mlp(MyMLP* mlp, const char* filename) {
mlp->load_model(filename);
}
}
