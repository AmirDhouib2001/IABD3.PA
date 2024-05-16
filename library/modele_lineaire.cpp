//
// Created by Damien Nadjar on 16/05/2024.
//
#include <iostream>
#include <vector>
#include <cmath>

// Données TEST
std::vector<double> X = {1.0, 2.0, 3.0, 4.0, 5.0}; // Features
std::vector<double> y = {2.0, 4.0, 6.0, 8.0, 10.0}; // Target

//Fonction de prédiction
double predict(double b0, double b1, double x) {
    return b0 + b1 * x;
}

//Fonction de coût
double computeCost(const std::vector<double>& X, const std::vector<double>& y, double b0, double b1) {
    double totalCost = 0.0;
    int m = X.size();

    for(int i = 0; i < m; ++i) {
        double prediction = predict(b0, b1, X[i]);
        double error = prediction - y[i];
        totalCost += error * error;
    }

    return totalCost / (2 * m);
}

//Descente de gradient
void gradientDescent(std::vector<double>& X, std::vector<double>& y, double& b0, double& b1, double learningRate, int iterations) {
    int m = X.size();

    for(int iter = 0; iter < iterations; ++iter) {
        double b0_gradient = 0.0;
        double b1_gradient = 0.0;

        for(int i = 0; i < m; ++i) {
            double prediction = predict(b0, b1, X[i]);
            double error = prediction - y[i];
            b0_gradient += error;
            b1_gradient += error * X[i];
        }

        b0 -= (learningRate / m) * b0_gradient;
        b1 -= (learningRate / m) * b1_gradient;

        // Afficher le coût toutes les 100 itérations
        if (iter % 100 == 0) {
            std::cout << "Iteration " << iter << ", Cost: " << computeCost(X, y, b0, b1) << std::endl;
        }
    }
}

//TEST

int main() {
    // Hyperparamètres
    double learningRate = 0.01;
    int iterations = 1000;

    // Paramètres du modèle
    double b0 = 0.0; // Ordonnée à l'origine
    double b1 = 0.0; // Pente

    // Entraîner le modèle
    gradientDescent(X, y, b0, b1, learningRate, iterations);

    // Afficher les paramètres finaux
    std::cout << "Final parameters: b0 = " << b0 << ", b1 = " << b1 << std::endl;

    // Faire des prédictions
    double x_new = 6.0;
    double y_pred = predict(b0, b1, x_new);
    std::cout << "Prediction for x = " << x_new << " : y = " << y_pred << std::endl;

    return 0;
}
