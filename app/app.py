# app/app.py
# Main Flask/Dash application for CyberSentryAI web interface
# Provides real-time packet capture, PCAP file upload, attack type inference, and visualization

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import scapy.all as scapy
from flask import Flask
import threading
import time
import os
import joblib
import base64

# Initialize Flask and Dash app
server = Flask(__name__)
app = dash.Dash(__name__, server=server, url_base_pathname='/')

# Check if running in a hosted environment (e.g., Render)
IS_HOSTED = os.getenv('IS_HOSTED', 'false').lower() == 'true'

# Load the trained LSTM model
model_path = os.path.join("models", "lstm_model.h5")
try:
    model = load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Load the saved scaler from training
try:
    scaler = joblib.load("models/scaler.pkl")
except Exception as e:
    print(f"Error loading scaler: {e}")
    # Fallback to a dummy scaler if loading fails (not ideal)
    scaler = StandardScaler()
    scaler.fit([[0], [1500]])  # Dummy fit for packet lengths (min: 0, max: 1500)

# Global variables for real-time capture
capturing = False
predictions = []
packet_lengths = []
stop_event = threading.Event()

def capture_packets():
    """Capture packets in a separate thread and classify them using the LSTM model."""
    global capturing, predictions, packet_lengths
    print("Starting packet capture...")
    while not stop_event.is_set():
        try:
            # Sniff packets (timeout ensures we can stop gracefully)
            packets = scapy.sniff(count=1, timeout=1)
            for packet in packets:
                if capturing and 'IP' in packet:
                    # Extract packet length
                    pkt_len = len(packet)
                    # Scale the packet length
                    scaled_len = scaler.transform([[pkt_len]])[0][0]
                    # Predict using the model
                    pred = model.predict(np.array([[scaled_len]]), verbose=0)[0][0]
                    label = 'attack' if pred > 0.5 else 'benign'
                    prob = pred if pred > 0.5 else 1 - pred
                    # Store results
                    packet_lengths.append(pkt_len)
                    predictions.append({'label': label, 'probability': prob})
                    # Keep only the last 50 predictions for display
                    if len(predictions) > 50:
                        predictions.pop(0)
                        packet_lengths.pop(0)
        except Exception as e:
            print(f"Error during packet capture: {e}")
        time.sleep(0.1)  # Prevent excessive CPU usage

def infer_attack_type(packet_length):
    """Infer attack type based on packet length (simplified for demo)."""
    if packet_length > 1000:
        return 'ddos', "DDoS: High packet lengths often indicate flooding attacks."
    elif packet_length > 500:
        return 'portscan', "Port Scan: Medium packet lengths may suggest scanning activity."
    else:
        return 'webattack', "Web Attack: Smaller packets may indicate web-based exploits."

# Define the Dash app layout with a sidebar and main content area
app.layout = html.Div([
    html.H1("CyberSentryAI - Network Intrusion Detection", style={'textAlign': 'center', 'color': '#2c3e50'}),
    html.Div([
        html.Div([
            html.H3("Controls", style={'color': '#34495e'}),
            html.Button('Start Monitoring', id='start-button', n_clicks=0, disabled=IS_HOSTED, style={'margin': '5px'}),
            html.Button('Stop Monitoring', id='stop-button', n_clicks=0, disabled=IS_HOSTED, style={'margin': '5px'}),
            html.Div(id='capture-status', children='Monitoring: Stopped', style={'margin': '10px'}),
            html.Div(
                "Real-time monitoring is disabled on the hosted version due to platform restrictions. Please use PCAP file uploads or run locally for real-time capture.",
                style={'color': 'red', 'margin': '10px'},
                hidden=not IS_HOSTED
            ),
            dcc.Upload(
                id='upload-pcap',
                children=html.Button('Upload PCAP File', style={'margin': '5px'}),
                multiple=False
            ),
            html.H3("Analysis Options", style={'color': '#34495e', 'marginTop': '20px'}),
            dcc.Dropdown(
                id='attack-type-filter',
                options=[
                    {'label': 'All', 'value': 'all'},
                    {'label': 'DDoS', 'value': 'ddos'},
                    {'label': 'Port Scan', 'value': 'portscan'},
                    {'label': 'Web Attack', 'value': 'webattack'}
                ],
                value='all',
                style={'margin': '5px'}
            ),
            html.Button('Analyze Packet Patterns', id='analyze-button', n_clicks=0, style={'margin': '5px'})
        ], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px', 'backgroundColor': '#ecf0f1'}),
        html.Div([
            html.H3("Analysis Results", style={'color': '#34495e'}),
            html.Div(id='pcap-output', style={'margin': '20px'}),
            dcc.Graph(id='live-graph', style={'height': '400px'}),
            html.H3("Attack Details", style={'color': '#34495e'}),
            html.Div(id='attack-details', style={'margin': '20px'}),
            html.H3("Alerts", style={'color': '#34495e'}),
            html.Div(id='alerts', style={'margin': '20px'})
        ], style={'width': '75%', 'display': 'inline-block', 'padding': '20px'})
    ]),
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0)
])

@app.callback(
    [Output('capture-status', 'children'),
     Output('start-button', 'disabled'),
     Output('stop-button', 'disabled')],
    [Input('start-button', 'n_clicks'),
     Input('stop-button', 'n_clicks')]
)
def toggle_monitoring(start_clicks, stop_clicks):
    """Handle start/stop monitoring button clicks."""
    global capturing
    if start_clicks > 0 and not capturing:
        capturing = True
        stop_event.clear()
        threading.Thread(target=capture_packets, daemon=True).start()
        return 'Monitoring: Running', True, False
    elif stop_clicks > 0 and capturing:
        capturing = False
        stop_event.set()
        return 'Monitoring: Stopped', False, True
    return 'Monitoring: Stopped' if not capturing else 'Monitoring: Running', capturing, not capturing

@app.callback(
    [Output('live-graph', 'figure'),
     Output('alerts', 'children'),
     Output('attack-details', 'children')],
    [Input('interval-component', 'n_intervals'),
     Input('attack-type-filter', 'value')]
)
def update_graph_and_alerts(n, attack_filter):
    """Update the live graph, alerts, and attack details with recent predictions, filtered by attack type."""
    if not predictions:
        return {
            'data': [],
            'layout': go.Layout(title='Real-Time Packet Classifications', xaxis={'title': 'Packet Index'}, yaxis={'title': 'Packet Length'})
        }, "No alerts yet.", "No attack details available."

    filtered_predictions = predictions
    filtered_lengths = packet_lengths
    if attack_filter != 'all':
        filtered_predictions = []
        filtered_lengths = []
        for i, (pred, pkt_len) in enumerate(zip(predictions, packet_lengths)):
            if pred['label'] == 'attack':
                attack_type, _ = infer_attack_type(pkt_len)
                if attack_type == attack_filter:
                    filtered_predictions.append(pred)
                    filtered_lengths.append(pkt_len)

    colors = ['red' if p['label'] == 'attack' else 'green' for p in filtered_predictions]
    scatter = go.Scatter(
        x=list(range(len(filtered_lengths))),
        y=filtered_lengths,
        mode='lines+markers',
        marker={'color': colors},
        name='Packet Lengths'
    )

    alerts = [f"Alert: {p['label'].capitalize()} detected (Probability: {p['probability']:.2f})" for p in filtered_predictions[-5:] if p['label'] == 'attack']
    alerts_text = html.Ul([html.Li(alert) for alert in alerts]) if alerts else "No recent attack alerts."

    attack_info = []
    for pkt_len, pred in zip(filtered_lengths, filtered_predictions):
        if pred['label'] == 'attack':
            attack_type, description = infer_attack_type(pkt_len)
            attack_info.append(f"{attack_type.upper()}: {description} (Packet Length: {pkt_len})")
    attack_details = html.Ul([html.Li(info) for info in attack_info]) if attack_info else "No attack details available."

    return {
        'data': [scatter],
        'layout': go.Layout(
            title='Real-Time Packet Classifications',
            xaxis={'title': 'Packet Index'},
            yaxis={'title': 'Packet Length'},
            showlegend=True
        )
    }, alerts_text, attack_details

@app.callback(
    [Output('pcap-output', 'children'),
     Output('attack-details', 'children', allow_duplicate=True)],
    Input('upload-pcap', 'contents'),
    State('upload-pcap', 'filename'),
    prevent_initial_call=True
)
def analyze_pcap(contents, filename):
    """Analyze an uploaded PCAP file and display predictions with attack details."""
    if contents is None:
        return "No PCAP file uploaded.", "No attack details available."

    try:
        # Decode and save the uploaded PCAP file
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        with open("temp.pcap", "wb") as f:
            f.write(decoded)

        # Read packets from the PCAP file
        packets = scapy.rdpcap("temp.pcap")
        results = []
        attack_info = []
        for packet in packets:
            if 'IP' in packet:
                pkt_len = len(packet)
                scaled_len = scaler.transform([[pkt_len]])[0][0]
                pred = model.predict(np.array([[scaled_len]]), verbose=0)[0][0]
                label = 'attack' if pred > 0.5 else 'benign'
                prob = pred if pred > 0.5 else 1 - pred
                if label == 'attack':
                    attack_type, description = infer_attack_type(pkt_len)
                    attack_info.append(f"{attack_type.upper()}: {description} (Packet Length: {pkt_len})")
                results.append(f"Packet Length: {pkt_len} | Prediction: {label} (Probability: {prob:.2f})")

        # Clean up temporary file
        os.remove("temp.pcap")
        return html.Ul([html.Li(result) for result in results]), html.Ul([html.Li(info) for info in attack_info])
    except Exception as e:
        return f"Error analyzing PCAP file: {e}", "No attack details available."

@app.callback(
    Output('pcap-output', 'children', allow_duplicate=True),
    Input('analyze-button', 'n_clicks'),
    State('upload-pcap', 'contents'),
    prevent_initial_call=True
)
def analyze_patterns(n_clicks, contents):
    """Analyze packet patterns in the uploaded PCAP file (e.g., average packet length, attack ratio)."""
    if n_clicks == 0 or contents is None:
        return "No analysis performed."

    try:
        # Decode and save the uploaded PCAP file
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        with open("temp.pcap", "wb") as f:
            f.write(decoded)

        # Analyze packets
        packets = scapy.rdpcap("temp.pcap")
        lengths = [len(pkt) for pkt in packets if 'IP' in pkt]
        attacks = 0
        for pkt in packets:
            if 'IP' in pkt:
                pkt_len = len(pkt)
                scaled_len = scaler.transform([[pkt_len]])[0][0]
                pred = model.predict(np.array([[scaled_len]]), verbose=0)[0][0]
                if pred > 0.5:
                    attacks += 1

        # Clean up temporary file
        os.remove("temp.pcap")
        avg_length = sum(lengths) / len(lengths) if lengths else 0
        attack_ratio = attacks / len(lengths) if lengths else 0
        return html.Ul([
            html.Li(f"Average Packet Length: {avg_length:.2f} bytes"),
            html.Li(f"Attack Ratio: {attack_ratio:.2f} (Attacks: {attacks}, Total Packets: {len(lengths)})")
        ])
    except Exception as e:
        return f"Error analyzing patterns: {e}"

if __name__ == '__main__':
    # Run the Dash app (use port 5001 to avoid conflicts on macOS)
    app.run_server(debug=True, port=5001)
