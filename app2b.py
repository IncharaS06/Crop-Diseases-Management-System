from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import json
import os
import requests

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# Load the trained model and class labels
model = tf.keras.models.load_model("C:/Users/harsh/Downloads/archive/banana_leaf_disease_model.h5")

with open("C:/Users/harsh/Downloads/archive/banana.json", "r") as f:
    class_labels = json.load(f)
class_labels = list(class_labels.keys())  # Convert dictionary keys to list

# Preprocess image function
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array

# Fetch farming-related information from Groq API based on the deficiency or disease
def get_farming_info(query):
    api_url = "https://api.groq.com/openai/v1/chat/completions"
    api_key = "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"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    prompt = (
        f"Provide detailed information about the following farming-related query:\n"
        f"Question: '{query}'"
    )

    data = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    try:
        response = requests.post(api_url, json=data, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

    result = response.json()
    return result["choices"][0]["message"]["content"]

# Route for home page
@app.route('/')
def home():
    return render_template('banana-disease.html')  # Renders the HTML page

# Route for processing image upload
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Ensure the 'uploads' folder exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    # Save the uploaded file
    img_path = os.path.join('uploads', file.filename)
    file.save(img_path)

    # Preprocess image and make prediction
    input_image = preprocess_image(img_path)
    prediction = model.predict(input_image)

    # Get predicted class index
    predicted_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_index] if predicted_index < len(class_labels) else "Unknown Disease"

    # Fetch farming-related info based on the predicted class
    query = f"Causes of {predicted_class} deficiency in banana plants, its effects, and how to overcome it."
    disease_info = get_farming_info(query)

    # Return results as JSON
    result = {
        'disease': predicted_class,
        'confidence': np.max(prediction) * 100,
        'disease_info': disease_info
    }

    return render_template('banana-disease.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
