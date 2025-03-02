import os
import h5py  # Ensure this import is included
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import uvicorn

app = FastAPI()

# Load models
model_path = 'plant_disease_model_inception.h5'
leaf_model_path = 'leaf-nonleaf.h5'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
if not os.path.exists(leaf_model_path):
    raise FileNotFoundError(f"Leaf model file not found at {leaf_model_path}")
print('Both models are loaded')

model = load_model(model_path)
leaf_model = load_model(leaf_model_path)

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
        img_pil = Image.fromarray(img_cv)
        enhancer = ImageEnhance.Brightness(img_pil)
        img_pil = enhancer.enhance(1.2)
        return np.array(img_pil)
    return img_cv

# Function to reduce noise
def reduce_noise(img_cv):
    if is_noisy(img_cv):
        img_cv = cv2.GaussianBlur(img_cv, (3, 3), 0)
    return img_cv

# Function to preprocess an image
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

# Function to check if image is a leaf
def is_not_leaf(img_path):
    img_array = preprocess_image(img_path, target_size=(224, 224))
    prediction = leaf_model.predict(img_array)
    return prediction[0][0] > 0.5

# Function to predict plant disease
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

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_path = 'temp.jpg'
        with open(img_path, "wb") as buffer:
            buffer.write(await file.read())
        result = predict_single_image(img_path)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
