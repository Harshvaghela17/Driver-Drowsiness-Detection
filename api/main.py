
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import tensorflow as tf
import numpy as np
import io
import os

app = FastAPI(
    title="Driver Drowsiness Detection API",
    description="CNN-based Drowsiness Detection"
)

# Model path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "MODEL",
    "drowsiness_model (1).keras"
)

# Check model exists BEFORE loading
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# Load model only once
model = tf.keras.models.load_model(MODEL_PATH)

# IMPORTANT:
# Change this according to train_ds.class_names in Colab
class_names = ["Drowsy", "Non Drowsy"]


@app.get("/")
def home():
    return {
        "message": "Driver Drowsiness Detection API is running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Upload JPG or PNG image"
        )

    try:
        contents = await file.read()

        # Read image
        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

        # Resize to training image size
        image = image.resize((224, 224))

        # Convert to NumPy array
        image_array = np.array(
            image,
            dtype=np.float32
        )

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # Predict
        prediction = model.predict(
            image_array,
            verbose=0
        )[0]

        # Get highest probability index
        predicted_class = int(np.argmax(prediction))

        # Get class name
        result = class_names[predicted_class]

        # Confidence
        confidence = float(prediction[predicted_class])

        return {
            "prediction": result,
            "confidence": round(confidence, 4),
            "probabilities": {
                class_names[0]: round(float(prediction[0]), 4),
                class_names[1]: round(float(prediction[1]), 4)
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
