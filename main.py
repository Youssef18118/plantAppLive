import os
import h5py  # Add this import
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
import cv2
from PIL import Image, ImageEnhance

app = Flask(__name__)

# Load models
model = load_model('plant_disease_model_inception.h5')
leaf_model = load_model('leaf-nonleaf.h5')

index_to_class = {
    0: 'Cherry_(including_sour)___Powdery_mildew',
    1: 'Cherry_(including_sour)___healthy',
    2: 'Grape___Black_rot',
    3: 'Grape___Esca_(Black_Measles)',
    4: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    5: 'Grape___healthy',
    6: 'Peach___Bacterial_spot',
    7: 'Peach___healthy',
    8: 'Pepper,_bell___Bacterial_spot',
    9: 'Pepper,_bell___healthy',
    10: 'Potato___Early_blight',
    11: 'Potato___Late_blight',
    12: 'Potato___healthy',
    13: 'Tomato__Bacterial_Spot',
    14: 'Tomato__Early_Blight',
    15: 'Tomato__Healthy',
    16: 'Tomato__Late_Blight',
    17: 'Tomato__Septoria_Leaf_Spot',
    18: 'Tomato__Yellow_Leaf_Curl_Virus'
}

# Function to check brightness level
def is_low_brightness(img_cv, threshold=50):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray)
    return brightness < threshold

# Function to check noise level
def is_noisy(img_cv, noise_threshold=100):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    noise_level = cv2.Laplacian(gray, cv2.CV_64F).var()
    return noise_level > noise_threshold

# Function to adjust brightness
def adjust_brightness(img):
    img_cv = np.array(img)
    if is_low_brightness(img_cv):
        # print("Adjusting brightness for a dark image")
        img_pil = Image.fromarray(img_cv)
        enhancer = ImageEnhance.Brightness(img_pil)
        img_pil = enhancer.enhance(1.2)
        return np.array(img_pil)
    return img_cv

# Function to reduce noise
def reduce_noise(img_cv):
    if is_noisy(img_cv):
        # print("Reducing noise using Gaussian Blur")
        img_cv = cv2.GaussianBlur(img_cv, (3, 3), 0)
    return img_cv

# Function to predict if an image is a leaf
def is_not_leaf(img_path):
    img_array = preprocess_image(img_path, target_size=(224, 224))
    prediction = leaf_model.predict(img_array)
    return prediction[0][0] > 0.5

# Define your image processing and prediction functions here
def preprocess_image(img_path, target_size=(299, 299)):
    img = Image.open(img_path).convert('RGB')
    img_cv = np.array(img)
    img_cv = adjust_brightness(img_cv)
    img_cv = reduce_noise(img_cv)
    img_pil = Image.fromarray(img_cv)
    img_resized = img_pil.resize(target_size)
    img_array = np.array(img_resized)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

def predict_single_image(img_path):
    if not is_not_leaf(img_path):
        img_array = preprocess_image(img_path)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = prediction[0][predicted_class]
        class_name = index_to_class.get(predicted_class, "Unknown")
        return {'class': class_name, 'confidence': float(confidence)}
    else:
        return {'error': 'Image is not classified as a leaf'}

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    img_path = 'temp.jpg'
    file.save(img_path)

    # Call your prediction function
    result = predict_single_image(img_path)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
