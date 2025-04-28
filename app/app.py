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
    scaler.fit([[0, 0, 0], [1500, 1, 1]])  # Dummy fit for packet length, inter-arrival time, protocol

# Global variables for real-time capture
capturing = False
predictions = []
packet_data = []  # Stores packet length, inter-arrival time, protocol
stop_event = threading.Event()

# Styling variables
DARK_BG = '#1C2526'
PRIMARY_BLUE = '#1E90FF'
TEXT_COLOR = '#E0E0E0'
FONT_FAMILY = 'Montserrat, sans-serif'
BOX_BG = '#2A3439'

def capture_packets():
    """Capture packets in a separate thread and classify them using the LSTM model."""
    global capturing, predictions, packet_data
    print("Starting packet capture...")
    last_time = None
    while not stop_event.is_set():
        try:
            packets = scapy.sniff(count=1, timeout=1)
            for packet in packets:
                if capturing and 'IP' in packet:
                    # Extract features
                    pkt_len = len(packet)
                    current_time = packet.time
                    inter_arrival = (current_time - last_time) if last_time else 0
                    last_time = current_time
                    protocol = 1 if 'TCP' in packet else 0  # Simplified protocol encoding (matches training)
                    # Scale the features
                    scaled_features = scaler.transform([[pkt_len, inter_arrival, protocol]])[0]
                    # Predict using the model
                    pred = model.predict(np.array([scaled_features]), verbose=0)[0][0]
                    label = 'attack' if pred > 0.5 else 'benign'
                    prob = pred if pred > 0.5 else 1 - pred
                    # Store results
                    packet_data.append([pkt_len, inter_arrival, protocol])
                    predictions.append({'label': label, 'probability': prob})
                    # Keep only the last 50 predictions for display
                    if len(predictions) > 50:
                        predictions.pop(0)
                        packet_data.pop(0)
        except Exception as e:
            print(f"Error during packet capture: {e}")
        time.sleep(0.1)

def infer_attack_type(packet_length):
    """Infer attack type based on packet length (simplified for demo)."""
    if packet_length > 1000:
        return 'DDoS', "DDoS: High packet lengths often indicate flooding attacks."
    elif packet_length > 500:
        return 'Port Scan', "Port Scan: Medium packet lengths may suggest scanning activity."
    else:
        return 'Web Attack', "Web Attack: Smaller packets may indicate web-based exploits."

# Include Montserrat font via Google Fonts
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>CyberSentryAI</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700;900&style=italic&display=swap" rel="stylesheet">
        <style>
            /* Custom styling for dropdown options */
            .Select-control {
                background-color: #2A3439 !important;
                color: #E0E0E0 !important;
                border: 1px solid #1E90FF !important;
            }
            .Select-menu-outer {
                background-color: #2A3439 !important;
                color: #E0E0E0 !important;
            }
            .Select-option {
                background-color: #2A3439 !important;
                color: #E0E0E0 !important;
            }
            .Select-option:hover {
                background-color: #1E90FF !important;
            }
            .Select-value-label {
                color: #E0E0E0 !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Define the Dash app layout with a fixed, non-scrollable page
app.layout = html.Div([
    # Navigation Bar
    html.Div([
        html.H1("CyberSentryAI", style={
            'color': PRIMARY_BLUE,
            'fontFamily': FONT_FAMILY,
            'fontSize': '32px',
            'fontWeight': 'bold',
            'fontStyle': 'italic',
            'textAlign': 'left',  # Moved to top left
            'padding': '10px',
            'backgroundColor': DARK_BG,
            'margin': '0'
        })
    ], style={'width': '100%', 'position': 'fixed', 'top': '0', 'zIndex': '1000'}),

    # Main content (fixed layout to prevent scrolling)
    html.Div([
        # Left Section (Monitoring Options, Alerts, Attack Details)
        html.Div([
            # Monitoring and Analysis Options
            html.Div([
                # Controls
                html.Div([
                    html.Button('Start Monitoring', id='start-button', n_clicks=0, style={
                        'backgroundColor': PRIMARY_BLUE,
                        'color': TEXT_COLOR,
                        'fontFamily': FONT_FAMILY,
                        'fontSize': '14px',
                        'fontWeight': 'bold',
                        'padding': '10px',
                        'margin': '5px',
                        'border': 'none',
                        'borderRadius': '5px'
                    }),
                    html.Button('Stop Monitoring', id='stop-button', n_clicks=0, style={
                        'backgroundColor': PRIMARY_BLUE,
                        'color': TEXT_COLOR,
                        'fontFamily': FONT_FAMILY,
                        'fontSize': '14px',
                        'fontWeight': 'bold',
                        'padding': '10px',
                        'margin': '5px',
                        'border': 'none',
                        'borderRadius': '5px'
                    }),
                    html.Div(id='capture-status', children='Monitoring: Stopped', style={
                        'color': TEXT_COLOR,
                        'fontFamily': FONT_FAMILY,
                        'fontSize': '14px',
                        'margin': '10px'
                    }),
                ], style={'textAlign': 'center'}),

                # Analysis Options
                html.Div([
                    html.H3("Analysis Options", style={
                        'color': TEXT_COLOR,
                        'fontFamily': FONT_FAMILY,
                        'fontSize': '18px',
                        'fontWeight': 'bold'
                    }),
                    dcc.Dropdown(
                        id='attack-type-filter',
                        options=[
                            {'label': 'All', 'value': 'all'},
                            {'label': 'DDoS', 'value': 'ddos'},
                            {'label': 'Port Scan', 'value': 'portscan'},
                            {'label': 'Web Attack', 'value': 'webattack'}
                        ],
                        value='all',
                        style={
                            'backgroundColor': BOX_BG,
                            'color': TEXT_COLOR,
                            'fontFamily': FONT_FAMILY,
                            'fontSize': '14px',
                            'margin': '5px'
                        },
                        clearable=False,
                        className='Select'  # For custom CSS styling
                    ),
                    dcc.Upload(
                        id='upload-pcap',
                        children=html.Button('Upload PCAP File', style={
                            'backgroundColor': PRIMARY_BLUE,
                            'color': TEXT_COLOR,
                            'fontFamily': FONT_FAMILY,
                            'fontSize': '14px',
                            'fontWeight': 'bold',
                            'padding': '10px',
                            'margin': '5px',
                            'border': 'none',
                            'borderRadius': '5px'
                        }),
                        multiple=False
                    ),
                    html.Button('Analyze Packet Patterns', id='analyze-button', n_clicks=0, style={
                        'backgroundColor': PRIMARY_BLUE,
                        'color': TEXT_COLOR,
                        'fontFamily': FONT_FAMILY,
                        'fontSize': '14px',
                        'fontWeight': 'bold',
                        'padding': '10px',
                        'margin': '5px',
                        'border': 'none',
                        'borderRadius': '5px'
                    })
                ], style={'marginTop': '10px'}),
            ], style={'marginBottom': '20px'}),

            # Alerts and Attack Details (stacked)
            html.Div([
                # Alerts
                html.Div([
                    html.H3("Alerts", style={
                        'color': TEXT_COLOR,
                        'fontFamily': FONT_FAMILY,
                        'fontSize': '18px',
                        'fontWeight': 'bold'
                    }),
                    html.Div(id='alerts', style={
                        'color': TEXT_COLOR,
                        'fontFamily': FONT_FAMILY,
                        'fontSize': '14px',
                        'height': '150px',  # Fixed height for scrollable window
                        'overflowY': 'auto',
                        'margin': '10px',
                        'padding': '10px',
                        'backgroundColor': BOX_BG,
                        'borderRadius': '5px'
                    })
                ], style={'marginBottom': '20px'}),

                # Attack Details (same styling as Alerts)
                html.Div([
                    html.H3("Attack Details", style={
                        'color': TEXT_COLOR,
                        'fontFamily': FONT_FAMILY,
                        'fontSize': '18px',
                        'fontWeight': 'bold'
                    }),
                    html.Div(id='attack-details', style={
                        'color': TEXT_COLOR,
                        'fontFamily': FONT_FAMILY,
                        'fontSize': '14px',
                        'height': '150px',  # Fixed height for scrollable window
                        'overflowY': 'auto',
                        'margin': '10px',
                        'padding': '10px',
                        'backgroundColor': BOX_BG,
                        'borderRadius': '5px'
                    })
                ])
            ], style={'height': 'calc(100vh - 300px)', 'overflow': 'hidden'})  # Adjusted height to fit
        ], style={
            'width': '60%',
            'display': 'inline-block',
            'verticalAlign': 'top',
            'padding': '20px',
            'height': 'calc(100vh - 60px)',  # Adjust for nav bar height
            'overflow': 'hidden'
        }),

        # Right Section (Graph at Top)
        html.Div([
            dcc.Graph(id='live-graph', style={'height': '400px', 'width': '100%'}),  # Increased height for visibility
        ], style={
            'width': '40%',
            'display': 'inline-block',
            'verticalAlign': 'top',
            'padding': '20px',
            'height': 'calc(100vh - 60px)',  # Adjust for nav bar height
            'overflow': 'hidden'
        })
    ], style={
        'backgroundColor': DARK_BG,
        'height': '100vh',
        'margin': '0',
        'paddingTop': '60px',  # Space for nav bar
        'overflow': 'hidden'
    }),

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
            'layout': go.Layout(
                title='Real-Time Packet Classifications',
                titlefont={'family': FONT_FAMILY, 'size': 18, 'color': TEXT_COLOR},
                xaxis={'title': 'Packet Index', 'titlefont': {'family': FONT_FAMILY, 'color': TEXT_COLOR}, 'tickfont': {'family': FONT_FAMILY, 'color': TEXT_COLOR}},
                yaxis={'title': 'Packet Length', 'titlefont': {'family': FONT_FAMILY, 'color': TEXT_COLOR}, 'tickfont': {'family': FONT_FAMILY, 'color': TEXT_COLOR}},
                paper_bgcolor=DARK_BG,
                plot_bgcolor=DARK_BG
            )
        }, "No alerts yet.", "No attack details available."

    filtered_predictions = predictions
    filtered_data = packet_data
    if attack_filter != 'all':
        filtered_predictions = []
        filtered_data = []
        for pred, data in zip(predictions, packet_data):
            if pred['label'] == 'attack':
                attack_type, _ = infer_attack_type(data[0])
                if attack_type.lower() == attack_filter:
                    filtered_predictions.append(pred)
                    filtered_data.append(data)

    colors = ['red' if p['label'] == 'attack' else 'green' for p in filtered_predictions]
    scatter = go.Scatter(
        x=list(range(len(filtered_data))),
        y=[d[0] for d in filtered_data],
        mode='lines+markers',
        marker={'color': colors},
        name='Packet Lengths',
        line={'color': PRIMARY_BLUE}
    )

    # Generate alerts (same for both sections)
    alerts = [f"Alert: {p['label'].capitalize()} detected (Probability: {p['probability']:.2f})" for p in predictions[-10:] if p['label'] == 'attack']
    alerts_text = html.Ul([html.Li(alert) for alert in alerts]) if alerts else "No recent attack alerts."

    # Generate attack details (same alerts with additional details)
    attack_info = []
    for data, pred in zip(packet_data[-10:], predictions[-10:]):  # Match the last 10 alerts
        if pred['label'] == 'attack':
            attack_type, description = infer_attack_type(data[0])
            alert_text = f"Alert: {pred['label'].capitalize()} detected (Probability: {pred['probability']:.2f})"
            attack_info.append(f"{alert_text} - {attack_type}: {description} (Packet Length: {data[0]})")
    attack_details = html.Ul([html.Li(info) for info in attack_info]) if attack_info else "No attack details available."

    return {
        'data': [scatter],
        'layout': go.Layout(
            title='Real-Time Packet Classifications',
            titlefont={'family': FONT_FAMILY, 'size': 18, 'color': TEXT_COLOR},
            xaxis={'title': 'Packet Index', 'titlefont': {'family': FONT_FAMILY, 'color': TEXT_COLOR}, 'tickfont': {'family': FONT_FAMILY, 'color': TEXT_COLOR}},
            yaxis={'title': 'Packet Length', 'titlefont': {'family': FONT_FAMILY, 'color': TEXT_COLOR}, 'tickfont': {'family': FONT_FAMILY, 'color': TEXT_COLOR}},
            paper_bgcolor=DARK_BG,
            plot_bgcolor=DARK_BG,
            showlegend=True,
            legend={'font': {'family': FONT_FAMILY, 'color': TEXT_COLOR}}
        )
    }, alerts_text, attack_details

@app.callback(
    [Output('alerts', 'children', allow_duplicate=True),
     Output('attack-details', 'children', allow_duplicate=True)],
    Input('upload-pcap', 'contents'),
    State('upload-pcap', 'filename'),
    prevent_initial_call=True
)
def analyze_pcap(contents, filename):
    """Analyze an uploaded PCAP file and display predictions with attack details."""
    if contents is None:
        return "No alerts yet.", "No attack details available."

    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        with open("temp.pcap", "wb") as f:
            f.write(decoded)

        packets = scapy.rdpcap("temp.pcap")
        results = []
        attack_info = []
        last_time = None
        for packet in packets:
            if 'IP' in packet:
                pkt_len = len(packet)
                current_time = packet.time
                inter_arrival = (current_time - last_time) if last_time else 0
                last_time = current_time
                protocol = 1 if 'TCP' in packet else 0
                scaled_features = scaler.transform([[pkt_len, inter_arrival, protocol]])[0]
                pred = model.predict(np.array([scaled_features]), verbose=0)[0][0]
                label = 'attack' if pred > 0.5 else 'benign'
                prob = pred if pred > 0.5 else 1 - pred
                if label == 'attack':
                    attack_type, description = infer_attack_type(pkt_len)
                    alert_text = f"Alert: {label.capitalize()} detected (Probability: {prob:.2f})"
                    attack_info.append(f"{alert_text} - {attack_type}: {description} (Packet Length: {pkt_len})")
                    results.append(alert_text)
                # Update global predictions for real-time monitoring consistency
                packet_data.append([pkt_len, inter_arrival, protocol])
                predictions.append({'label': label, 'probability': prob})
                if len(predictions) > 50:
                    predictions.pop(0)
                    packet_data.pop(0)

        os.remove("temp.pcap")
        alerts_text = html.Ul([html.Li(result) for result in results[-10:]]) if results else "No recent attack alerts."
        attack_details = html.Ul([html.Li(info) for info in attack_info[-10:]]) if attack_info else "No attack details available."
        return alerts_text, attack_details
    except Exception as e:
        return f"Error analyzing PCAP file: {e}", f"Error analyzing PCAP file: {e}"

@app.callback(
    Output('alerts', 'children', allow_duplicate=True),
    Input('analyze-button', 'n_clicks'),
    State('upload-pcap', 'contents'),
    prevent_initial_call=True
)
def analyze_patterns(n_clicks, contents):
    """Analyze packet patterns in the uploaded PCAP file (e.g., average packet length, attack ratio)."""
    if n_clicks == 0 or contents is None:
        return "No alerts yet."

    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        with open("temp.pcap", "wb") as f:
            f.write(decoded)

        packets = scapy.rdpcap("temp.pcap")
        lengths = []
        attacks = 0
        last_time = None
        for pkt in packets:
            if 'IP' in pkt:
                pkt_len = len(pkt)
                lengths.append(pkt_len)
                current_time = pkt.time
                inter_arrival = (current_time - last_time) if last_time else 0
                last_time = current_time
                protocol = 1 if 'TCP' in pkt else 0
                scaled_features = scaler.transform([[pkt_len, inter_arrival, protocol]])[0]
                pred = model.predict(np.array([scaled_features]), verbose=0)[0][0]
                if pred > 0.5:
                    attacks += 1

        os.remove("temp.pcap")
        avg_length = sum(lengths) / len(lengths) if lengths else 0
        attack_ratio = attacks / len(lengths) if lengths else 0
        results = [
            f"Average Packet Length: {avg_length:.2f} bytes",
            f"Attack Ratio: {attack_ratio:.2f} (Attacks: {attacks}, Total Packets: {len(lengths)})"
        ]
        return html.Ul([html.Li(result) for result in results])
    except Exception as e:
        return f"Error analyzing patterns: {e}"

if __name__ == '__main__':
    app.run_server(debug=True, port=5001)
