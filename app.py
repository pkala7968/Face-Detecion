from flask import Flask, render_template, request, jsonify, redirect, url_for
import numpy as np
import cv2
import base64
import re
from keras_facenet import FaceNet
from sklearn.preprocessing import Normalizer
import joblib
from PIL import Image
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)

# Load RandomForest model + label encoder
model = joblib.load("model.pkl")
encoder = joblib.load("label_encoder.pkl")

# Load FaceNet embedder
embedder = FaceNet()
l2_normalizer = Normalizer('l2')

@app.route('/')
def index():
    return render_template('index.html')  # HTML page with webcam capture

@app.route('/verify', methods=['POST'])
def verify():
    data = request.get_json()
    img_data = re.sub('^data:image/.+;base64,', '', data['image'])
    img = Image.open(BytesIO(base64.b64decode(img_data)))
    img = img.convert('RGB').resize((160, 160))
    img_np = np.array(img)

    # Compute embedding
    embedding = embedder.embeddings([img_np])[0]
    embedding = l2_normalizer.transform([embedding])[0]

    # Predict using RandomForest
    pred = model.predict([embedding])[0]
    prob = np.max(model.predict_proba([embedding])[0])
    name = encoder.inverse_transform([pred])[0]

    if prob >= 0.5:
        return jsonify({'result': 'success', 'name': name, 'confidence': float(prob), 'redirect': url_for('dashboard')})
    else:
        return jsonify({'result': 'fail', 'message': 'Unknown or low confidence', 'confidence': float(prob)})

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True)