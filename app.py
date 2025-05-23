from flask import Flask, render_template, request, redirect, url_for, jsonify
import numpy as np
import cv2
import base64
import re
from keras_facenet import FaceNet
from sklearn.preprocessing import Normalizer
from tensorflow.keras.models import load_model
import joblib
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Load CNN model and label encoder
model = load_model("cnn_face_classifier.h5")
encoder = joblib.load("label_encoder.pkl")
embedder = FaceNet()
l2_normalizer = Normalizer('l2')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/verify', methods=['POST'])
def verify():
    data = request.get_json()
    img_data = re.sub('^data:image/.+;base64,', '', data['image'])
    
    img = Image.open(BytesIO(base64.b64decode(img_data)))
    img = img.convert('RGB').resize((160, 160))
    img_np = np.array(img)

    # Get embedding and normalize
    embedding = embedder.embeddings([img_np])[0]
    embedding = l2_normalizer.transform([embedding])[0]

    # Predict using CNN model
    prediction = model.predict(np.array([embedding]))[0]
    predicted_class = encoder.inverse_transform([np.argmax(prediction)])[0]
    confidence = np.max(prediction)

    if confidence > 0.8:
        return jsonify({'result': 'success', 'name': predicted_class, 'redirect': url_for('dashboard')})
    else:
        return jsonify({'result': 'fail'})

@app.route('/dashboard')
def dashboard():
    return "<h1>Access Granted!</h1><p>Welcome to the secure page.</p>"

if __name__ == '__main__':
    app.run(debug=True)