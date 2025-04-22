#!/usr/bin/env python3
from scapy.all import sniff
from tensorflow.keras.models import load_model
import numpy as np
import time
from pathlib import Path

# Load the trained model using absolute path
model_path = Path("/Users/gabegiancarlo/Projects/CyberSentryAI/models/lstm_model.h5")
model = load_model(model_path)

def packet_callback(packet):
    # Extract packet length (or other features)
    if packet.haslayer('IP'):
        length = packet['IP'].len
        # Preprocess the feature
        X = np.array([[length]])
        X = X.reshape((X.shape[0], 1, X.shape[1]))  # Reshape for LSTM [samples, timesteps, features]
        # Make prediction
        prediction = model.predict(X, verbose=0)[0][0]
        label = 'attack' if prediction >= 0.5 else 'benign'
        print(f"Packet Length: {length} | Prediction: {label} (probability: {prediction:.2f})")

# Capture packets (run as root if on macOS)
print("Starting real-time detection... (Press Ctrl+C to stop)")
try:
    sniff(prn=packet_callback, store=0, count=100)  # Capture 100 packets
except PermissionError:
    print("Error: You may need to run this script with sudo on macOS (e.g., 'sudo python3 src/realtime_detection.py')")
