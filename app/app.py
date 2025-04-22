#!/usr/bin/env python3
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from pathlib import Path
import argparse
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the model using absolute path
model_path = Path("/Users/gabegiancarlo/Projects/CyberSentryAI/models/lstm_model.h5")
model = load_model(model_path)

# Initialize the scaler (same as used in training)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.data_min_ = np.array([0])  # Minimum packet length
scaler.data_max_ = np.array([65535])  # Maximum packet length for IP packets

@app.route('/')
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['GET'])
def predict():
    # Get the length from the query parameter
    length = float(request.args.get('length'))
    # Preprocess the input
    X = np.array([[length]])
    X_scaled = scaler.transform(X)
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))  # Reshape for LSTM [samples, timesteps, features]
    # Make prediction
    prediction = model.predict(X_scaled)[0][0]
    label = 'attack' if prediction >= 0.5 else 'benign'
    return render_template('index.html', prediction=f"Prediction: {label} (probability: {prediction:.2f})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Flask app on a specified port.')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the Flask app on.')
    args = parser.parse_args()
    app.run(debug=True, host='0.0.0.0', port=args.port)
