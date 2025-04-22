#!/usr/bin/env python3
import pandas as pd
import glob
import os
from pathlib import Path

# Use absolute paths with Path for better handling
project_dir = Path("/Users/gabegiancarlo/Projects/CyberSentryAI")
traffic_dir = project_dir / "data" / "TrafficLabelling"
combined_path = project_dir / "data" / "cicids2017.csv"

# Convert to string for compatibility with glob and pandas
traffic_dir = str(traffic_dir)
combined_path = str(combined_path)

# Print paths for debugging
print(f"TrafficLabelling directory: {traffic_dir}")
print(f"Combined file path: {combined_path}")

# Verify the directory exists
if not os.path.exists(traffic_dir):
    print(f"Error: Directory {traffic_dir} does not exist.")
    data_dir = os.path.dirname(traffic_dir)
    print(f"Contents of data directory: {os.listdir(data_dir)}")
    exit(1)

# Find all CSV files
cicids_files = glob.glob(traffic_dir + "/*.csv")
print(f"Found {len(cicids_files)} CSV files in {traffic_dir}:")
for file in cicids_files:
    print(f" - {file}")

# Check if any files were found
if not cicids_files:
    print("Error: No CSV files found in the directory.")
    exit(1)

# Process files incrementally to save memory
first_file = True
for file in cicids_files:
    print(f"Processing {file}...")
    try:
        # Read the CSV file
        df = pd.read_csv(file, encoding='latin1', low_memory=False)
        print(f"Loaded {file} with shape: {df.shape}")
        
        # Write to the combined file (append mode after the first file)
        if first_file:
            df.to_csv(combined_path, index=False, mode='w')
            first_file = False
        else:
            df.to_csv(combined_path, index=False, mode='a', header=False)
    except Exception as e:
        print(f"Error processing {file}: {e}")
        continue

print(f"Combined dataset saved to {combined_path}")
# Load the final combined file to get its shape
combined_df = pd.read_csv(combined_path, encoding='latin1', low_memory=False)
print(f"Final dataset shape: {combined_df.shape}")
