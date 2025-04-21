from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = load_model('../models/lstm_model.h5')
scaler = MinMaxScaler()  # Load or fit scaler as needed

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']
    data = scaler.transform([data])
    data = np.array(data).reshape((1, 1, len(data[0])))
    prediction = model.predict(data)
    return jsonify({'prediction': 'Intrusion' if prediction > 0.5 else 'Normal'})

if __name__ == '__main__':
    app.run(debug=True)
