# CyberSentryAI - CIC-IDS-2017 Dataset

This repository contains the CIC-IDS-2017 dataset for cybersecurity research and machine learning applications.

## Dataset Information

- **Source**: [CIC-IDS-2017 Dataset](http://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/)
- **Total Size**: ~1.12 GB
- **Files**: 8 CSV files containing network traffic data
- **Features**: 85 columns per record
- **Label Column**: ' Label' (note the leading space)

## Dataset Structure

```
data/
└── raw/
    ├── GeneratedLabelledFlows/
    │   └── TrafficLabelling /
    │       ├── Monday-WorkingHours.pcap_ISCX.csv (256.2 MB)
    │       ├── Tuesday-WorkingHours.pcap_ISCX.csv (166.6 MB)
    │       ├── Wednesday-workingHours.pcap_ISCX.csv (272.4 MB)
    │       ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv (87.8 MB)
    │       ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv (103.7 MB)
    │       ├── Friday-WorkingHours-Morning.pcap_ISCX.csv (71.9 MB)
    │       ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv (91.6 MB)
    │       └── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv (97.2 MB)
    └── MachineLearningCSV/
        └── MachineLearningCVE/
            └── [Same 8 files as above]
```

## Files Description

- **Monday-WorkingHours**: Normal traffic
- **Tuesday-WorkingHours**: Normal traffic  
- **Wednesday-workingHours**: Normal traffic
- **Thursday-WorkingHours-Morning-WebAttacks**: Web attacks
- **Thursday-WorkingHours-Afternoon-Infilteration**: Infiltration attacks
- **Friday-WorkingHours-Morning**: Normal traffic
- **Friday-WorkingHours-Afternoon-DDos**: DDoS attacks
- **Friday-WorkingHours-Afternoon-PortScan**: Port scan attacks

## Setup

1. **Virtual Environment**: A Python virtual environment is already set up in `venv/`
2. **Dependencies**: Install required packages with:
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Usage

### Download Dataset
```bash
python download_dataset.py
```

### Verify Dataset
```bash
python verify_dataset.py
```

### Load Data in Python
```python
import pandas as pd

# Load a single file
df = pd.read_csv('data/raw/GeneratedLabelledFlows/TrafficLabelling /Monday-WorkingHours.pcap_ISCX.csv')

# Check the data
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Label distribution: {df[' Label'].value_counts()}")
```

## Dataset Features

The dataset contains 85 features including:
- Flow ID
- Source IP and Port
- Destination IP and Port
- Protocol information
- Flow duration and statistics
- Packet counts and sizes
- Various network statistics
- Attack labels

## Notes

- The label column has a leading space: `' Label'`
- Files are quite large (70-270 MB each)
- Consider using chunked reading for large-scale processing
- The dataset contains both normal and attack traffic patterns

## Next Steps

This dataset is ready for:
- Exploratory Data Analysis (EDA)
- Feature engineering
- Machine learning model training
- Intrusion detection system development
- Network security research
