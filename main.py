import os
import h5py  # Add this import
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
from PIL import Image, ImageEnhance, UnidentifiedImageError
import io
import logging
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Model path
model_path = "plant_disease_model_inception.h5"

# Validate the HDF5 file before loading
try:
    logger.info("Validating HDF5 file...")
    with h5py.File(model_path, "r") as f:
        logger.info("HDF5 file is valid.")
except Exception as e:
    logger.error(f"HDF5 file is invalid: {e}")
    raise HTTPException(status_code=500, detail=f"HDF5 file is invalid: {e}")

# Load model
try:
    logger.info(f"Attempting to load model from: {model_path}")
    model = load_model(model_path)
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise HTTPException(status_code=500, detail=f"Error loading model: {e}")

# Class names
class_names = [
    "Tomato__Septoria_Leaf_Spot",
    "Grape___healthy",
    "Tomato__Healthy",
    "Potato___healthy",
    "Tomato__Early_Blight",
    "Peach___healthy",
    "Tomato__Yellow_Leaf_Curl_Virus",
    "Cherry_(including_sour)___Powdery_mildew",
    "Peach___Bacterial_spot",
    "Tomato__Bacterial_Spot",
    "Pepper,_bell___healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Potato___Late_blight",
    "Grape___Black_rot",
    "Tomato__Late_Blight",
    "Potato___Early_blight",
    "Cherry_(including_sour)___healthy",
    "Grape___Esca_(Black_Measles)",
    "Pepper,_bell___Bacterial_spot"
]

def is_low_brightness(img_array):
    """Check if the image has low brightness"""
    return np.mean(img_array) < 100

def is_noisy(img_array):
    """Check if the image is noisy"""
    return np.var(img_array) > 1000

def enhance_image(img_pil):
    """Apply preprocessing enhancements to the image"""
    img_array = np.array(img_pil)

    # Brightness correction
    if is_low_brightness(img_array):
        enhancer = ImageEnhance.Brightness(img_pil)
        img_pil = enhancer.enhance(1.2)
        img_array = np.array(img_pil)

    # Noise reduction
    if is_noisy(img_array):
        img_array = cv2.GaussianBlur(img_array, (3, 3), 5)
        img_pil = Image.fromarray(img_array)

    return img_pil

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and validate image file
        contents = await file.read()
        img_buffer = io.BytesIO(contents)

        try:
            img_pil = Image.open(img_buffer)
        except UnidentifiedImageError:
            raise HTTPException(
                status_code=400,
                detail="Unsupported image format or corrupted file"
            )

        # Resize and enhance image
        img_pil = img_pil.resize((299, 299), Image.BILINEAR)
        processed_img = enhance_image(img_pil)

        # Preprocess for model
        img_array = image.img_to_array(processed_img)
        img_array = preprocess_input(np.expand_dims(img_array, axis=0))

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        return JSONResponse(content={
            "predicted_class": predicted_class,
            "confidence": confidence
        })

    except HTTPException as he:
        logger.error(f"HTTPException: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
