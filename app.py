from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

model = load_model('models/cnn_model.h5')
CLASS_NAMES = ['COVID', 'Normal', 'Pneumonia', 'Tuberculosis']  # Update if needed

def preprocess_image(image):
    image = image.resize((224, 224)).convert('RGB')
    img_array = img_to_array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            image = Image.open(file.stream)
            processed = preprocess_image(image)
            preds = model.predict(processed)
            prediction = CLASS_NAMES[np.argmax(preds)]
            confidence = f"{np.max(preds) * 100:.2f}%"

    return render_template('index.html', prediction=prediction, confidence=confidence)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
