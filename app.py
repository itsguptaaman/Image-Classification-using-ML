from flask import Flask, render_template, request, redirect, flash
import os
from werkzeug.utils import secure_filename
import base64
from prediction import load_model, model_path, predict_image

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'supersecretkey'

model_data = load_model(model_path)
clf = model_data['classifier']
pca = model_data['pca']
le = model_data['label_encoder']

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        predicted_label = predict_image(filepath, clf, pca, le)
        print(f"Predicted label: {predicted_label}")
        with open(filepath, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return render_template('index.html', filename=filename, image_data=encoded_string, prediction=predicted_label)
    flash('Invalid file format')
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
