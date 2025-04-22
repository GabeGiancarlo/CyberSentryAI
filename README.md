🌐 CyberSentryAI
CyberSentryAI is an innovative AI-powered cybersecurity tool designed to monitor network traffic, detect anomalies, and identify potential threats in real time. Built with Python and leveraging machine learning, it empowers users to enhance network security through automation and intelligent analysis.

🚀 Features

Real-Time Monitoring 📡: Analyze network packets as they flow.
Anomaly Detection 🛡️: Identify unusual patterns using AI models.
Customizable Alerts 🚨: Get notified of potential threats instantly.
Extensible Framework 🛠️: Easily integrate with tools like Wireshark or Suricata.
User-Friendly CLI 💻: Manage and configure via simple commands.


📋 Table of Contents

Installation
Setup
Usage
Contributing
Contact


🛠️ Installation
Get CyberSentryAI up and running in just a few steps.
# Clone the repository
git clone https://github.com/GabeGiancarlo/CyberSentryAI.git
cd CyberSentryAI

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt


Note: Ensure you have Python 3.8+ installed. Check with python3 --version.


⚙️ Setup
Configure CyberSentryAI to suit your environment.

Install Network Tools (optional, for packet capture):
# On Ubuntu/Debian
sudo apt-get install wireshark tshark
# On macOS
brew install wireshark


Set Environment Variables: Create a .env file in the root directory:
echo "MODEL_PATH=models/anomaly_detector.pkl" > .env
echo "LOG_LEVEL=INFO" >> .env


Prepare Data (if training models): Place your dataset in the data/ directory or update config.yaml to point to your data source.



💻 Usage
Run CyberSentryAI via the command line to start monitoring or analyzing.
# Start real-time network monitoring
python3 src/main.py --mode monitor

# Analyze a PCAP file
python3 src/main.py --mode analyze --file data/sample.pcap

# Train a new model
python3 src/train.py --dataset data/network_traffic.csv

Example Output:
[INFO] Starting CyberSentryAI...
[INFO] Monitoring interface eth0...
[ALERT] Anomaly detected: Unusual traffic from 192.168.1.100


Tip: Use --help for more CLI options: python3 src/main.py --help


🤝 Contributing
We welcome contributions to make CyberSentryAI even better! 🙌

Fork the repository.
Create a feature branch (git checkout -b feature/awesome-feature).
Commit your changes (git commit -m "Add awesome feature").
Push to the branch (git push origin feature/awesome-feature).
Open a Pull Request.

Check out our Contributing Guidelines for more details.

📬 Contact
Have questions or ideas? Reach out!

Name: Gabriel Giancarlo
Email: giancarlo@chapman.edu
LinkedIn: gabe-giancarlo
GitHub: GabeGiancarlo


🌟 Thank you for exploring CyberSentryAI! Let's secure the digital world together.

