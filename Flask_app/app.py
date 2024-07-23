from flask import Flask, request, jsonify, render_template
import numpy as np
import ctypes
from ctypes import POINTER, c_double, c_int64, c_void_p, c_bool
from PIL import Image
import io

app = Flask(__name__)

mlp_lib = ctypes.CDLL("../library/mlp_save.dll")

create_mlp = mlp_lib.create_mlp
create_mlp.argtypes = [POINTER(c_int64), c_int64]
create_mlp.restype = c_void_p

destroy_mlp = mlp_lib.mlp_free
destroy_mlp.argtypes = [c_void_p]
destroy_mlp.restype = None

predict_mlp = mlp_lib.mlp_predict
predict_mlp.argtypes = [c_void_p, POINTER(c_double), c_int64, c_int64, c_bool]
predict_mlp.restype = POINTER(c_double)

load_mlp = mlp_lib.load_mlp
load_mlp.argtypes = [c_void_p, ctypes.c_char_p]
load_mlp.restype = None

layers = (c_int64 * 4)(9408, 32, 16, 3)
mlp_instance = create_mlp(layers, len(layers))

load_mlp(mlp_instance, b'C:/Users/webazza/Desktop/Projet_annual/PA_Pycha/loaded_lib/mon_mlp_model.bin')
print("MLP model loaded")

def predict(mlp_instance, X):
    predictions = np.zeros((len(X), 3), dtype=np.float64)  # Adjust the output size accordingly
    for i in range(len(X)):
        test_sample = (c_double * X.shape[1])(*X[i])
        try:
            result_ptr = predict_mlp(mlp_instance, test_sample, 1, X.shape[1], True)
            if result_ptr:
                result_array = np.ctypeslib.as_array(result_ptr, shape=(1, 3))
                predictions[i] = result_array
                mlp_lib.mlp_free(result_ptr)
            else:
                print(f"Error: result_ptr is NULL for sample {i}")
        except Exception as e:
            print(f"Error during prediction for sample {i}: {e}")
    return predictions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return 'Aucun fichier sélectionné', 400

    file = request.files['image']
    if file.filename == '':
        return 'Aucun fichier sélectionné', 400

    try:
        image = Image.open(file).convert('RGB')
        image = image.resize((56, 56))
        image_array = np.array(image, dtype=np.float64) / 255.0
        image_array = image_array.flatten()
        image_array = np.expand_dims(image_array, axis=0)

        predictions = predict(mlp_instance, image_array)
        print("predictions", predictions)
        prediction = np.argmax(predictions, axis=1)[0]

        print("prediction : ", prediction)

        flower_map = {0: 'Marguerite', 1: 'Rose', 2: 'Tulipe'}
        flower_name = flower_map.get(prediction, 'Inconnu')

        return jsonify({'prediction': flower_name})
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True)
