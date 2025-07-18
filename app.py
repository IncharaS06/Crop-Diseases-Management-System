from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import json
import os
import requests

app = Flask(__name__)

# Load the trained model and class labels
model = tf.keras.models.load_model("C:/Users/incha/OneDrive/Desktop/frontend2/plant_leaf_disease_model.h5")

with open("C:/Users/incha/OneDrive/Desktop/frontend2/class_indices.json", "r") as f:
    class_indices = json.load(f)
class_labels = {v: k for k, v in class_indices.items()}

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array

def get_farming_info(disease_name):
    api_url = "https://api.groq.com/openai/v1/chat/completions"
    api_key = "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"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    prompt = f"Provide detailed information about the following plant disease:\nDisease: {disease_name}\n\n- Causes\n- How it spreads\n- How much crop yield it decreases\n- Preventive measures"

    data = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    try:
        response = requests.post(api_url, json=data, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return f"Error fetching data from Groq API: {e}"

    result = response.json()
    return result["choices"][0]["message"]["content"]

@app.route('/')
def home():
    return render_template('plant-disease.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('plant-disease.html', error="No file part")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('plant-disease.html', error="No selected file")
    
    # Save the uploaded file
    img_path = os.path.join('uploads', file.filename)
    file.save(img_path)

    # Preprocess image and make prediction
    input_image = preprocess_image(img_path)
    prediction = model.predict(input_image)

    # Get predicted class index
    predicted_index = np.argmax(prediction)
    predicted_class = class_labels.get(predicted_index, "Unknown Disease")

    # Fetch disease-related farming info
    disease_info = get_farming_info(predicted_class)

    # Pass results to template
    result = {
        'disease': predicted_class,
        'confidence': np.max(prediction) * 100,
        'disease_info': disease_info
    }

    return render_template('plant-disease.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
