#!/usr/bin/env python3
"""
Script to download the CIC-IDS-2017 dataset from the official repository.
"""

import os
import requests
import zipfile
from pathlib import Path
import sys

def download_file(url, local_path, chunk_size=8192):
    """Download a file with progress indication."""
    print(f"Downloading {url}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}%", end='', flush=True)
        
        print(f"\nDownloaded: {local_path}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"\nError downloading {url}: {e}")
        return False

def extract_zip(zip_path, extract_to):
    """Extract a zip file to the specified directory."""
    print(f"Extracting {zip_path}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted to: {extract_to}")
        return True
    except zipfile.BadZipFile as e:
        print(f"Error extracting {zip_path}: {e}")
        return False

def main():
    """Main function to download and extract the CIC-IDS-2017 dataset."""
    
    # Base URLs
    base_url = "http://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/"
    
    # Files to download
    files_to_download = [
        "GeneratedLabelledFlows.zip",
        "MachineLearningCSV.zip"
    ]
    
    # Create directories
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    print("CIC-IDS-2017 Dataset Downloader")
    print("=" * 40)
    
    # Download files
    for filename in files_to_download:
        url = base_url + filename
        local_path = raw_dir / filename
        
        if local_path.exists():
            print(f"File {filename} already exists, skipping...")
            continue
            
        success = download_file(url, local_path)
        if not success:
            print(f"Failed to download {filename}")
            continue
    
    # Extract files
    print("\nExtracting downloaded files...")
    for filename in files_to_download:
        zip_path = raw_dir / filename
        if zip_path.exists():
            extract_dir = raw_dir / filename.replace('.zip', '')
            extract_dir.mkdir(exist_ok=True)
            extract_zip(zip_path, extract_dir)
    
    print("\nDataset download and extraction completed!")
    print(f"Data location: {raw_dir.absolute()}")

if __name__ == "__main__":
    main()
