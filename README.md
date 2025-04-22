CyberSentryAI
A Network Intrusion Detection System Using LSTM for Real-Time Traffic Classification
CyberSentryAI is an advanced network intrusion detection system (NIDS) designed to classify network traffic as either benign or malicious using a Long Short-Term Memory (LSTM) neural network. This project leverages large-scale network traffic datasets to train a robust model capable of identifying potential attacks in real-time. It features a Flask-based web interface for manual predictions and a real-time packet capturing module using Scapy, making it a versatile tool for network security monitoring.
Table of Contents

Project Overview
Features
Datasets
Prerequisites
Installation
Usage
Training the Model
Running the Web Interface
Real-Time Detection


Directory Structure
Troubleshooting
Future Improvements
Contributing
License
Contact

Project Overview
CyberSentryAI aims to provide a scalable and efficient solution for network intrusion detection, particularly for environments with high traffic volumes, such as enterprise networks and IoT ecosystems. The system uses an LSTM model trained on a combined dataset of 5.82 million network traffic samples, sourced from the IoTID20 and CICIDS2017 datasets. The project includes two main components:

A Flask web application for manual prediction of network packet classifications.
A real-time detection script that captures live network packets and classifies them as benign or attack.

This project was developed as part of a cybersecurity research initiative to explore the application of deep learning in network security, with potential use cases in both academic research and practical deployment.
Features

LSTM-Based Classification: Utilizes a deep learning model to accurately classify network traffic.
Web Interface: A user-friendly Flask application to input packet lengths and receive predictions.
Real-Time Detection: Captures live network packets using Scapy and provides instant classifications.
Scalable Data Processing: Handles large datasets (5.82M rows) using chunked processing during training.
Extensive Documentation: Detailed setup, usage, and troubleshooting guides for users and contributors.

Datasets
CyberSentryAI uses two primary datasets for training, which are combined into a single dataset for model training. Due to their large size, these datasets are not included in the GitHub repository but are available for download via Google Drive links.

IoTID20: IoT network traffic dataset containing 2.98 million rows of labeled traffic data.Download IoTID20 Dataset

CICIDS2017: General network traffic dataset with 2.83 million rows, widely used for intrusion detection research.Download CICIDS2017 Dataset

Combined Dataset: A preprocessed dataset combining IoTID20 and CICIDS2017, totaling 5.82 million rows, used for training the LSTM model.Download Combined Dataset


Dataset Notes

Ensure you have sufficient storage space (~10-15 GB) to download and extract these datasets.
Place the downloaded CSV files in the data/ directory of the project before proceeding with the setup.

Prerequisites
Before setting up CyberSentryAI, ensure you have the following installed on your system:

Operating System: macOS, Linux, or Windows (macOS recommended for this setup guide).
Python: Version 3.9 or higher.
pip: Python package manager.
Git: For cloning the repository.
sudo Access: Required for real-time packet capturing with Scapy on macOS/Linux.
Hardware Requirements:
At least 16 GB of RAM (for training on the full dataset).
20 GB of free disk space (for datasets and model files).


Network Access: Required for real-time detection; ensure you have permission to capture packets on your network.

Installation
Follow these steps to set up CyberSentryAI on your machine.
1. Clone the Repository
Clone the CyberSentryAI repository from GitHub to your local machine:
git clone https://github.com/yourusername/CyberSentryAI.git
cd CyberSentryAI

Replace yourusername with your actual GitHub username.
2. Download the Datasets
Download the datasets using the links provided in the Datasets section. Place the CSV files in the data/ directory:

data/IoTID20.csv
data/CICIDS2017.csv
data/combined_data.csv

3. Install Dependencies
Install the required Python packages using the provided requirements.txt file:
pip3 install -r requirements.txt

Common Dependencies
The requirements.txt includes:

tensorflow (for LSTM model training and inference)
flask (for the web interface)
scapy (for real-time packet capturing)
pandas, numpy, scikit-learn (for data preprocessing)

If you encounter issues with scapy during real-time detection, you may need to install it separately for the root user:
sudo pip3 install scapy

4. Verify Directory Structure
Ensure your project directory matches the following structure:
CyberSentryAI/
├── app/
│   ├── app.py
│   └── templates/
│       └── index.html
├── data/
│   ├── IoTID20.csv
│   ├── CICIDS2017.csv
│   └── combined_data.csv
├── models/
│   └── lstm_model.h5
├── notebooks/
│   └── eda.ipynb
├── scripts/
│   └── align_and_combine.py
├── src/
│   ├── train_model.py
│   └── realtime_detection.py
├── README.md
└── requirements.txt

Usage
CyberSentryAI can be used in three main ways: combining datasets, training the model, and running detection (either via the web interface or in real-time).
1. Combining Datasets
The align_and_combine.py script merges the IoTID20 and CICIDS2017 datasets into a single combined_data.csv file for training.
Run the script:
python3 scripts/align_and_combine.py


Input: data/IoTID20.csv, data/CICIDS2017.csv
Output: data/combined_data.csv
Time Estimate: ~5-10 minutes depending on your hardware.

2. Training the Model
Train the LSTM model using the combined dataset with the train_model.py script.
python3 src/train_model.py


Input: data/combined_data.csv
Output: models/lstm_model.h5
Time Estimate: ~60-90 minutes for the full dataset (5.82M rows) with 2 epochs per chunk.
Notes:
The script processes the dataset in chunks of 100,000 rows to manage memory usage.
Training logs will show accuracy and loss for each chunk.



3. Running the Web Interface
The Flask web application allows you to input packet lengths and receive predictions on whether the traffic is benign or malicious.
Start the Flask app:
python3 app/app.py --port 5001


Access: Open your browser and go to http://localhost:5001.
Usage:
Enter a packet length (e.g., 500) in the input field.
Click “Test Prediction” to see the classification (e.g., “Prediction: benign (probability: 0.45)”).


Notes:
Use --port 5001 if port 5000 is in use (common on macOS due to AirPlay Receiver).
To free up port 5000, disable AirPlay Receiver in System Preferences -> General -> AirDrop & Handoff and restart your Mac.



4. Real-Time Detection
The realtime_detection.py script captures live network packets and classifies them in real-time using the trained model.
Run the script (requires root privileges):
sudo python3 src/realtime_detection.py


Output Example:
Starting real-time detection... (Press Ctrl+C to stop)
Packet Length: 84 | Prediction: benign (probability: 0.45)
Packet Length: 96 | Prediction: attack (probability: 0.60)


Notes:

Requires sudo on macOS/Linux for packet capturing.
Generate network traffic (e.g., ping google.com) in another terminal to see predictions.
Press Ctrl+C to stop capturing packets.



Directory Structure

app/: Contains the Flask application.
app.py: Main Flask script for the web interface.
templates/index.html: HTML template for the web interface.


data/: Directory for datasets (not included in Git; download from links).
models/: Stores the trained LSTM model (lstm_model.h5).
notebooks/: Jupyter notebooks for exploratory data analysis.
eda.ipynb: Notebook for initial data exploration.


scripts/: Utility scripts for data preprocessing.
align_and_combine.py: Script to combine IoTID20 and CICIDS2017 datasets.


src/: Core scripts for training and detection.
train_model.py: Script to train the LSTM model.
realtime_detection.py: Script for real-time packet capturing and classification.


README.md: Project documentation (this file).
requirements.txt: List of Python dependencies.

Troubleshooting
Common Issues

Port Conflict on 5000 (Flask App):

Error: Address already in use

Solution: Run the Flask app on a different port:
python3 app/app.py --port 5001


Alternatively, free up port 5000 by disabling AirPlay Receiver:

Go to System Preferences -> General -> AirDrop & Handoff.
Uncheck AirPlay Receiver.
Restart your Mac.




Permission Error with Scapy (Real-Time Detection):

Error: PermissionError: You may need to run this script with sudo

Solution: Run the script with sudo:
sudo python3 src/realtime_detection.py


If Scapy isn’t found, install it for the root user:
sudo pip3 install scapy




Uniform Predictions (All Benign or Attack):

Issue: Predictions are consistently benign with low probabilities.

Solution:

Ensure preprocessing is consistent (scaling is applied in both app.py and realtime_detection.py).

Retrain the model with more epochs or additional features:

Edit src/train_model.py to increase epochs=2 to epochs=3.
Add more features (e.g., protocol type, packet rate) to the dataset.


Rerun training:
python3 src/train_model.py






Memory Issues During Training:

Error: Out of memory when processing the dataset.
Solution:
Reduce the chunk size in train_model.py (e.g., change chunk_size = 100000 to 50000).
Ensure you have at least 16 GB of RAM available.





Debugging Tips

Check training logs in the terminal output for accuracy and loss trends.
Use top or Activity Monitor to monitor memory usage during training.
Verify dataset integrity by opening combined_data.csv in a text editor or Jupyter notebook.

Future Improvements

Enhanced Feature Extraction: Incorporate additional features like protocol type, packet rate, and source/destination ports for better classification accuracy.
Model Optimization: Experiment with different architectures (e.g., CNN-LSTM hybrid) or hyperparameters to improve performance.
User Interface: Add more interactive elements to the Flask app, such as a dashboard for real-time detection results.
Cross-Platform Support: Test and document setup instructions for Windows and Linux.
Alert System: Implement notifications (e.g., email or SMS) for detected attacks in real-time mode.

Contributing
Contributions are welcome! To contribute to CyberSentryAI:

Fork the repository on GitHub.

Clone your fork to your local machine:
git clone https://github.com/yourusername/CyberSentryAI.git


Create a new branch for your feature or bugfix:
git checkout -b feature/your-feature-name


Make your changes and commit them with a descriptive message:
git commit -m "Add feature: your feature description"


Push your changes to your fork:
git push origin feature/your-feature-name


Open a pull request on the main repository, describing your changes in detail.


Please ensure your code follows PEP 8 style guidelines and includes appropriate documentation.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Contact
For questions, suggestions, or collaboration opportunities, please reach out:

Author: Gabe Giancarlo
Email: giancarlo@chapmam.edu
GitHub: gabegiancarlo


Last updated: April 22, 2025

