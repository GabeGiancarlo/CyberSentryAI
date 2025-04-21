import pyshark
from time import sleep
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained model and scaler
model = load_model('../models/lstm_model.h5')
scaler = MinMaxScaler()  # Load or fit scaler as needed

def extract_features(packet):
    # Placeholder: Extract features like packet length, protocol, etc.
    # This will depend on your dataset features
    features = [float(packet.length), float(packet.protocol)]  # Example
    return features

# Capture live packets
capture = pyshark.LiveCapture(interface='eth0')
for packet in capture.sniff_continuously(packet_count=10):
    features = extract_features(packet)
    features = scaler.transform([features])
    features = np.array(features).reshape((1, 1, len(features[0])))
    prediction = model.predict(features)
    if prediction > 0.5:
        print("Intrusion detected!")
    sleep(1)  # Simulate real-time processing
