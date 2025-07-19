# 🌿 Crop Diseases Management System

An AI-driven platform designed to **detect crop diseases**, provide **remedial solutions**, and offer **weather and irrigation guidance**—empowering farmers and agri-tech experts with intelligent decision-making. Built using Python and trained on comprehensive Kaggle datasets, the system achieves a **model accuracy exceeding 85%** across multiple crop types.

## 🚀 Key Features

- 🧪 **Disease Prediction** from leaf images and symptoms using ML models  
- 🍎 **Remedy Suggestions** for fruits and plants via Groq API  
- ☁️ **Weather Forecasting** integrated through OpenWeatherMap API  
- 💧 **Irrigation Recommendations** based on crop type, soil condition & weather  
- 📊 Accuracy above 85% validated on real-world datasets  
- 🖼️ Visual feedback on plant health status  
- 🧠 Fast inference and intelligent recommendations powered by Groq’s ultra-efficient AI runtime

## 🧠 Technologies Used

| Tech / Library            | Description                                 |
|---------------------------|---------------------------------------------|
| Python                    | Core programming language                   |
| scikit-learn / TensorFlow | ML model training and prediction            |
| Groq API                  | Real-time solution generation & inference   |
| OpenWeatherMap API        | Weather info for irrigation planning        |
| requests / BeautifulSoup  | API and data handling                       |
| Streamlit / Flask (optional) | User interface / dashboard (if applicable) |


## 📈 Model Performance

| Crop Category   | Algorithm Used      | Accuracy |
|-----------------|---------------------|----------|
| Fruits          | CNN + Groq Inference| 88%      |
| Vegetables      | SVM / Random Forest | 86%      |
| Leaf Diseases   | Transfer Learning   | 90%      |

> All models were trained using curated Kaggle datasets containing labeled images and disease metadata.

## 🌦️ Weather & Irrigation Integration

- Real-time weather reports fetched using OpenWeatherMap API  
- Adaptive irrigation suggestions based on current temperature, humidity, and crop needs  
- Dynamic alerts for drought, rainfall, and frost events

## 🧪 Groq API Powered Remediation

When a crop disease is detected, Groq's AI platform generates:

- The name of the disease  
- Recommended pesticides or organic remedies  
- Best recovery practices based on plant type  
- Risk level and urgency

## 🔧 Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/IncharaS06/Crop-Diseases-Management-System.git
   cd Crop-Diseases-Management-System
