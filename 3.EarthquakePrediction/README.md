# Earthquake Risk Prediction AI

## 📌 Description
Predict high-risk and low-risk earthquake events based on geospatial and seismic features like latitude, longitude, depth, and magnitude. The system also visualizes affected areas on an interactive map and provides results through a web interface.

## 📂 Dataset
- `data/weather_data.csv`

## 🛠 Model Used
- Artificial Neural Network (ANN) using TensorFlow/Keras
- Deep learning classification model

## ✨ Steps
1. Load and clean data
2. Convert date & time to timestamp
3. Preprocess features (scaling)
4. Split dataset into training and testing sets
5. Train neural network model with early stopping
6. Evaluate model accuracy on test data
7. Visualize earthquake locations on an interactive map
8. Optional: Use Streamlit app to upload CSV and view predictions

## ▶ How to Run
- Run model training:
```bash
python src/train.py
streamlit run app.py
## 📌 Features

High-risk vs low-risk classification

Neural network with optimized hyperparameters

Geospatial visualization using Folium

Interactive web interface with Streamlit