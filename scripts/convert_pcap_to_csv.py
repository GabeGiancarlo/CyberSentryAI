import pandas as pd
import pyshark
import os

# Directory containing .pcap files
pcap_dir = '/Users/gabegiancarlo/Downloads/iot_intrusion_dataset'  # Update this to your actual path
output_dir = '/Users/gabegiancarlo/projects/CyberSentryAI'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# List to store all packet data
all_packets = []

# Define a function to determine the label based on the file name
def get_label(filename):
    if 'benign' in filename.lower():
        return 'benign'
    else:
        return 'attack'

# Process each .pcap file
for pcap_file in os.listdir(pcap_dir):
    if pcap_file.endswith('.pcap'):
        print(f"Processing {pcap_file}...")
        label = get_label(pcap_file)
        cap = pyshark.FileCapture(os.path.join(pcap_dir, pcap_file))
        
        for packet in cap:
            try:
                packet_data = {
                    'timestamp': packet.sniff_time,
                    'src_ip': packet.ip.src if 'ip' in packet else '',
                    'dst_ip': packet.ip.dst if 'ip' in packet else '',
                    'protocol': packet.highest_layer,
                    'length': int(packet.length),
                    'label': label
                }
                all_packets.append(packet_data)
            except AttributeError:
                continue
        cap.close()

# Convert to DataFrame and save as CSV
df = pd.DataFrame(all_packets)
df.to_csv(os.path.join(output_dir, 'iotid20.csv'), index=False)
print("Saved to CyberSentryAI/data/iotid20.csv")
