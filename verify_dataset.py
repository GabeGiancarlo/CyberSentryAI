#!/usr/bin/env python3
"""
Script to verify that the CIC-IDS-2017 dataset has been downloaded correctly
and can be accessed and read.
"""

import pandas as pd
import os
from pathlib import Path

def verify_dataset():
    """Verify the dataset files are accessible and readable."""
    
    print("CIC-IDS-2017 Dataset Verification")
    print("=" * 40)
    
    # Check if data directory exists
    data_dir = Path("data/raw")
    if not data_dir.exists():
        print("❌ Data directory not found!")
        return False
    
    print("✅ Data directory found")
    
    # Check GeneratedLabelledFlows directory
    flows_dir = data_dir / "GeneratedLabelledFlows" / "TrafficLabelling "
    if not flows_dir.exists():
        print("❌ GeneratedLabelledFlows directory not found!")
        return False
    
    print("✅ GeneratedLabelledFlows directory found")
    
    # List all CSV files
    csv_files = list(flows_dir.glob("*.csv"))
    print(f"✅ Found {len(csv_files)} CSV files:")
    
    total_size = 0
    for csv_file in csv_files:
        file_size = csv_file.stat().st_size
        total_size += file_size
        print(f"   - {csv_file.name} ({file_size / (1024*1024):.1f} MB)")
    
    print(f"\nTotal dataset size: {total_size / (1024*1024*1024):.2f} GB")
    
    # Try to read a sample file
    print("\nTesting file reading...")
    try:
        # Read the first file as a sample
        sample_file = csv_files[0]
        print(f"Reading sample file: {sample_file.name}")
        
        # Read just the first few rows to check structure
        df_sample = pd.read_csv(sample_file, nrows=5)
        print(f"✅ Successfully read {len(df_sample)} rows from sample file")
        print(f"   Columns: {len(df_sample.columns)}")
        print(f"   Column names: {list(df_sample.columns)[:5]}...")  # Show first 5 columns
        
        # Check if it has the expected structure
        if 'Label' in df_sample.columns:
            print("✅ Found 'Label' column - dataset appears to be properly formatted")
        else:
            print("⚠️  'Label' column not found - checking for alternative label columns")
            label_cols = [col for col in df_sample.columns if 'label' in col.lower() or 'class' in col.lower()]
            if label_cols:
                print(f"   Found potential label columns: {label_cols}")
            else:
                print("   No obvious label columns found")
        
    except Exception as e:
        print(f"❌ Error reading sample file: {e}")
        return False
    
    print("\n✅ Dataset verification completed successfully!")
    return True

if __name__ == "__main__":
    verify_dataset()
