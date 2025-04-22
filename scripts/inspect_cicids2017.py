#!/usr/bin/env python3
import pandas as pd
import glob
import os

# Dynamically compute the path to the data directory
script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script (CyberSentryAI/scripts/)
project_dir = os.path.dirname(script_dir)  # Parent directory (CyberSentryAI/)
traffic_dir = os.path.join(project_dir, 'data', 'TrafficLabelling')  # CyberSentryAI/data/TrafficLabelling/
combined_path = os.path.join(project_dir, 'data', 'cicids2017.csv')

print(f"TrafficLabelling directory: {traffic_dir}")
print(f"Combined file path: {combined_path}")

# Check if the combined file exists; if not, combine the files
if not pd.io.common.file_exists(combined_path):
    print("Combining CICIDS2017 files...")
    cicids_files = glob.glob(traffic_dir + '/*.csv')
    dfs = []
    for file in cicids_files:
        print(f"Loading {file}...")
        df = pd.read_csv(file)
        dfs.append(df)
    cicids_combined = pd.concat(dfs, ignore_index=True)
    cicids_combined.to_csv(combined_path, index=False)
    print(f"Combined dataset saved to {combined_path}")
else:
    print("Combined CICIDS2017 file already exists.")

# Load the combined dataset
print("\nLoading CICIDS2017 dataset...")
cicids = pd.read_csv(combined_path)

# Display basic information
print("\nDataset Info:")
print(cicids.info())

# Display the first few rows
print("\nFirst 5 Rows:")
print(cicids.head())

# Display the shape
print("\nDataset Shape:")
print(cicids.shape)

# Display the columns
print("\nColumns:")
print(list(cicids.columns))

# Display label distribution
print("\nLabel Distribution:")
print(cicids['Label'].value_counts())
