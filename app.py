import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "pneumonia_model.h5")

# Load trained model
model = load_model(model_path)
IMG_SIZE = 224

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/")
def home():
    return "Pneumonia Detection API is running"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = os.path.join(script_dir, "temp.jpg")
    file.save(file_path)

    img_array = preprocess_image(file_path)
    pred = model.predict(img_array)[0][0]
    os.remove(file_path)

    if pred > 0.5:
        result = "PNEUMONIA"
        confidence = pred * 100
    else:
        result = "NORMAL"
        confidence = (1 - pred) * 100

    return jsonify({
        "prediction": result,
        "confidence": round(float(confidence), 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
