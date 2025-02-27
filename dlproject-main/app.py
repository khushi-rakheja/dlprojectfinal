from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import io

app = Flask(__name__)

model = load_model("mask_model.h5")  # Load your model

# Get the expected input shape from the model
input_shape = (128, 128)  # Change this to match your model's expected input size

# Define class labels
class_labels = ["No Mask", "Mask"]  # Adjust based on your dataset

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files["image"]
    image = Image.open(io.BytesIO(file.read()))  # Open image
    image = image.convert("RGB")  # Ensure it's in RGB mode
    image = image.resize(input_shape)  # Resize to match model input
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    prediction = model.predict(image)  # Get model prediction
    predicted_class = class_labels[np.argmax(prediction)]  # Get class label

    return jsonify({"prediction": predicted_class})

if __name__ == "__main__":
        # app.run(debug=True, host="0.0.0.0", port=10000)
     port = int(os.environ.get("PORT", 5000))  # Railway provides a PORT environment variable
     app.run(host="0.0.0.0", port=port)