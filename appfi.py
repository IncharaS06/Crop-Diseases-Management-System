from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import json
import os
import requests

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# Load models and class labels for each type
# Plant Disease Model
plant_model = tf.keras.models.load_model("C:/Users/incha/OneDrive/Desktop/frontend2/plant_leaf_disease_model.h5")
with open("C:/Users/incha/OneDrive/Desktop/frontend2/class_indices.json", "r") as f:
    plant_class_labels = json.load(f)
plant_class_labels = list(plant_class_labels.keys())

# Fruit Disease Model
fruit_model = tf.keras.models.load_model("C:/Users/incha/OneDrive/Desktop/frontend2/fruit_veg_disease_model.h5")
with open("C:/Users/incha/OneDrive/Desktop/frontend2/fruit.json", "r") as f:
    fruit_class_labels = json.load(f)
fruit_class_labels = list(fruit_class_labels.keys())

# Fruit & Veg Quality Grading Model
quality_model = tf.keras.models.load_model("C:/Users/incha/OneDrive/Desktop/frontend2/banana_leaf_disease_model.h5")
with open("C:/Users/incha/OneDrive/Desktop/frontend2/banana.json", "r") as f:
    quality_class_labels = json.load(f)
quality_class_labels = list(quality_class_labels.keys())

# Preprocess image function
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array

# Fetch farming-related information from Groq API
def get_farming_info(query):
    api_url = "https://api.groq.com/openai/v1/chat/completions"
    api_key = "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"  # Replace with your Groq API key
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    prompt = f"Provide detailed information about the following farming-related query: '{query}'"
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

# Route for home page (home.html)
@app.route('/')
def home():
    return render_template('home.html')  # Renders the home page

# Route for index page (index.html)
@app.route('/index')
def index():
    return render_template('index.html')  # Renders the index page

@app.route('/irrigation')
def irrigation():
    return render_template('irrigation.html')

# Route for processing image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('home.html', error="No file selected")

    file = request.files['file']
    if file.filename == '':
        return render_template('home.html', error="No selected file")

    # Ensure the 'uploads' folder exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    # Save the uploaded file
    img_path = os.path.join('uploads', file.filename)
    file.save(img_path)

    # Get the model type selected by the user
    model_type = request.form.get('model_type')

    # Initialize variables for prediction results
    predicted_class = "Unknown"
    disease_info = "No information available."
    confidence = 0

    # Prediction logic based on selected model
    if model_type == 'plant_disease':
        model = plant_model
        class_labels = plant_class_labels
    elif model_type == 'fruit_veg_quality':
        model = fruit_model
        class_labels = fruit_class_labels
    elif model_type == 'Banana_leaf_deficiency':
        model = quality_model
        class_labels = quality_class_labels
    else:
        return render_template('home.html', error="Invalid model type selected.")

    # Preprocess image and make prediction
    input_image = preprocess_image(img_path)
    prediction = model.predict(input_image)
    predicted_index = np.argmax(prediction)

    if predicted_index < len(class_labels):
        predicted_class = class_labels[predicted_index]
    confidence = np.max(prediction) * 100

    # Fetch detailed farming information based on the prediction
    query = f"Causes of {predicted_class} and its effects on crops"
    disease_info = get_farming_info(query)
    


    # Return results as JSON
    result = {
        'disease': predicted_class,
        'confidence': confidence,
        'disease_info': disease_info
    }

    return render_template('home.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
