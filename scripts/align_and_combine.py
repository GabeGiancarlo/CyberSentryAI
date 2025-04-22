#!/usr/bin/env python3
import pandas as pd
import os
from pathlib import Path

# Paths to the datasets
project_dir = Path("/Users/gabegiancarlo/Projects/CyberSentryAI")
iotid20_path = project_dir / "data" / "iotid20.csv"
cicids2017_path = project_dir / "data" / "cicids2017.csv"
combined_path = project_dir / "data" / "combined_data.csv"

# Process datasets in chunks
chunk_size = 100000  # Process 100,000 rows at a time
first_chunk = True

print("Processing datasets in chunks...")
for chunk_iotid20, chunk_cicids2017 in zip(
    pd.read_csv(iotid20_path, chunksize=chunk_size),
    pd.read_csv(cicids2017_path, chunksize=chunk_size, low_memory=False)
):
    print(f"Processing chunk of {chunk_size} rows...")
    
    # Strip whitespace from column names
    chunk_iotid20.columns = chunk_iotid20.columns.str.strip()
    chunk_cicids2017.columns = chunk_cicids2017.columns.str.strip()

    # IoTID20: select features
    chunk_iotid20 = chunk_iotid20[['length', 'label']]

    # CICIDS2017: map features and labels
    chunk_cicids2017 = chunk_cicids2017.rename(columns={
        'Total Length of Fwd Packets': 'length',
        'Label': 'label'
    })
    chunk_cicids2017['label'] = chunk_cicids2017['label'].apply(lambda x: 'benign' if x == 'BENIGN' else 'attack')
    chunk_cicids2017 = chunk_cicids2017[['length', 'label']]

    # Combine the chunk
    combined_chunk = pd.concat([chunk_iotid20, chunk_cicids2017], ignore_index=True)
    combined_chunk.fillna(combined_chunk.mean(numeric_only=True), inplace=True)

    # Write to file (append mode after the first chunk)
    if first_chunk:
        combined_chunk.to_csv(combined_path, index=False, mode='w')
        first_chunk = False
    else:
        combined_chunk.to_csv(combined_path, index=False, mode='a', header=False)

print(f"Combined dataset saved to {combined_path}")

# Load the final dataset to get its shape and label distribution
combined_data = pd.read_csv(combined_path)
print(f"Final shape: {combined_data.shape}")
print("Label distribution:")
print(combined_data['label'].value_counts())
