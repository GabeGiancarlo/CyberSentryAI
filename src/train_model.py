#!/usr/bin/env python3
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from pathlib import Path

# Define absolute paths
project_dir = Path("/Users/gabegiancarlo/Projects/CyberSentryAI")
data_path = project_dir / "data" / "combined_data.csv"
model_path = project_dir / "models" / "lstm_model.h5"

# Create models directory if it doesn't exist
os.makedirs(project_dir / "models", exist_ok=True)

# Initialize the model
model = Sequential()
model.add(LSTM(64, input_shape=(1, 1), return_sequences=True))
model.add(Dropout(0.3))  # Increased dropout to reduce overfitting
model.add(LSTM(32))
model.add(Dropout(0.3))  # Increased dropout
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Process the dataset in chunks
chunk_size = 100000
scaler = MinMaxScaler()
first_chunk = True

print("Training model on chunks...")
for chunk in pd.read_csv(data_path, chunksize=chunk_size):
    print(f"Processing chunk of {chunk_size} rows...")
    
    # Prepare features and labels
    chunk['label'] = chunk['label'].map({'benign': 0, 'attack': 1})
    chunk.fillna(chunk.mean(numeric_only=True), inplace=True)

    X = scaler.fit_transform(chunk[['length']]) if first_chunk else scaler.transform(chunk[['length']])
    y = chunk['label']

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reshape for LSTM [samples, timesteps, features]
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Train the model on this chunk
    model.fit(X_train, y_train, epochs=2, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    first_chunk = False

# Save the model
model.save(model_path)
print(f"Model saved to {model_path}")
