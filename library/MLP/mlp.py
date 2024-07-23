import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import ctypes
from ctypes import POINTER, c_double, c_int64, c_void_p, c_bool
from ctypes import c_double

# Load the DLL
mlp_lib = ctypes.CDLL("../library/mlp_save.dll")

# Define types for the functions
create_mlp = mlp_lib.create_mlp
create_mlp.argtypes = [POINTER(c_int64), c_int64]
create_mlp.restype = c_void_p

destroy_mlp = mlp_lib.mlp_free
destroy_mlp.argtypes = [c_void_p]
destroy_mlp.restype = None

train_mlp = mlp_lib.train_mlp
train_mlp.argtypes = [c_void_p, POINTER(c_double), c_int64, c_int64, POINTER(c_double), c_int64, c_double, c_int64, c_bool]
train_mlp.restype = None

predict_mlp = mlp_lib.mlp_predict
predict_mlp.argtypes = [c_void_p, POINTER(c_double), c_int64, c_int64, c_bool]
predict_mlp.restype = POINTER(c_double)

get_loss_values = mlp_lib.get_loss_values
get_loss_values.argtypes = [c_void_p, POINTER(c_int64)]
get_loss_values.restype = POINTER(c_double)

save_mlp = mlp_lib.save_mlp
save_mlp.argtypes = [c_void_p, ctypes.c_char_p]
save_mlp.restype = None

load_mlp = mlp_lib.load_mlp
load_mlp.argtypes = [c_void_p, ctypes.c_char_p]
load_mlp.restype = None
def prepare_flower_data():
    # Load vectors from files
    marguerite_vectors = np.load("../../src/vector_data/marguerite_fleur_vectors.npy")
    rose_vectors = np.load("../../src/vector_data/rose_rouge_vectors.npy")
    tulipe_vectors = np.load("../../src/vector_data/tulipe_jaune_vectors.npy")

    # Create labels for classes
    marguerite_labels = np.array([[1.0, 0.0, 0.0]] * len(marguerite_vectors))
    rose_labels = np.array([[0.0, 1.0, 0.0]] * len(rose_vectors))
    tulipe_labels = np.array([[0.0, 0.0, 1.0]] * len(tulipe_vectors))

    # Combine the data and labels
    X = np.concatenate((marguerite_vectors, rose_vectors, tulipe_vectors))
    y = np.concatenate((marguerite_labels, rose_labels, tulipe_labels))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Flatten and convert training and testing data to c_double arrays
    train_data = (c_double * X_train.size)(*X_train.flatten())
    test_data = (c_double * X_test.size)(*X_test.flatten())
    train_labels = (c_double * y_train.size)(*y_train.flatten())
    print(f"Number of training points: {len(X_train)}")
    print(f"Number of test points: {len(X_test)}")

    return train_data, test_data, train_labels, X_train, X_test, y_train, y_test

train_data, test_data, train_labels, X_train, X_test, y_train, y_test = prepare_flower_data()

# Define the structure of the MLP
layers = (c_int64 * 4)(X_train.shape[1], 32, 16, 3)  # Example structure with one hidden layer of 10 neurons
mlp_instance = create_mlp(layers, len(layers))

print("MLP network created")

alpha = 0.01
print("MLP network created with : ")
for i in range(len(layers)):
    print(layers[i], end=" ")
print("Alpha : ", alpha)

# Train the MLP network
try:
    train_mlp(mlp_instance, train_data, len(X_train), X_train.shape[1], train_labels, y_train.shape[1], alpha, 500000, True)
except Exception as e:
    print("Error during training:", e)

print("MLP network trained")

save_mlp(mlp_instance, b'mon_mlp_model.dat')
print("MLP model saved")

load_mlp(mlp_instance, b'mon_mlp_model.dat')
print("MLP model loaded")

# Predict using the MLP network
def predict(mlp_instance, X):
    predictions = np.zeros((len(X), y_test.shape[1]), dtype=np.float64)
    for i in range(len(X)):
        test_sample = (c_double * X.shape[1])(*X[i])
        try:
            result_ptr = predict_mlp(mlp_instance, test_sample, 1, X.shape[1], True)
            if result_ptr:
                result_array = np.ctypeslib.as_array(result_ptr, shape=(1, y_test.shape[1]))
                predictions[i] = result_array
                mlp_lib.mlp_free(result_ptr)
            else:
                print(f"Error: result_ptr is NULL for sample {i}")
        except Exception as e:
            print(f"Error during prediction for sample {i}: {e}")
    return predictions

train_predictions = predict(mlp_instance, X_train)
test_predictions = predict(mlp_instance, X_test)

# Calculate accuracies
train_accuracy = accuracy_score(np.argmax(y_train, axis=1), np.argmax(train_predictions, axis=1))
test_accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(test_predictions, axis=1))

print(f"Train accuracy: {train_accuracy:.2f}")
print(f"Test accuracy: {test_accuracy:.2f}")
