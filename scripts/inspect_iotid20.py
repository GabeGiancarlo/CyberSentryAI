#!/usr/bin/env python3
import pandas as pd
import os

# Dynamically compute the path to the data directory
script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script (CyberSentryAI/scripts/)
project_dir = os.path.dirname(script_dir)  # Parent directory (CyberSentryAI/)
iotid20_path = os.path.join(project_dir, 'data', 'iotid20.csv')
print(f"IoTID20 file path: {iotid20_path}")

# Load the dataset
print("Loading IoTID20 dataset...")
iot = pd.read_csv(iotid20_path)

# Display basic information
print("\nDataset Info:")
print(iot.info())

# Display the first few rows
print("\nFirst 5 Rows:")
print(iot.head())

# Display the shape (number of rows and columns)
print("\nDataset Shape:")
print(iot.shape)

# Display the columns
print("\nColumns:")
print(list(iot.columns))

# Display label distribution
print("\nLabel Distribution:")
print(iot['label'].value_counts())
