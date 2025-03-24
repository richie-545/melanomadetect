import os
import io
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Use environment variable for the model path (with a fallback)
MODEL_PATH = os.environ.get("MODEL_PATH", "model.tflite")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ["Benign", "Malignant"]

def preprocess_image(image):
    """
    Preprocess the input image:
      - Resize to the expected model size (e.g., 224x224)
      - Convert to numpy array and use EfficientNet preprocessing to scale pixel values to [-1, 1]
      - Expand dims to add batch dimension
    """
    target_size = (224, 224)
    image = image.resize(target_size)
    image = np.array(image).astype("float32")
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

def predict_tflite(processed_image):
    interpreter.set_tensor(input_details[0]['index'], processed_image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/disclaimer")
def disclaimer():
    return render_template("disclaimer.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/analysis")
def analysis():
    return render_template("analysis.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": "Invalid image format", "details": str(e)}), 400

    processed_image = preprocess_image(image)
    prediction = predict_tflite(processed_image)

    predicted_index = np.argmax(prediction)
    predicted_label = class_names[predicted_index]
    predicted_prob = float(prediction[0][predicted_index]) * 100

    return jsonify({"prediction": predicted_label, "confidence": f"{predicted_prob:.2f}%"})

if __name__ == "__main__":
    # For production, set debug=False
    app.run(debug=True)
