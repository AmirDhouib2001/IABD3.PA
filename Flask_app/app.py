from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return 'Aucun fichier sélectionné'

    file = request.files['image']

    # Traiter l'image ici (à remplacer par le code de classification en Rust)

    return 'Image uploadée avec succès'


if __name__ == '__main__':
    app.run(debug=True)
