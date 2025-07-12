# preprocess_data.py
import pandas as pd
import numpy as np
import joblib
import json

# Load preprocessing objects and selected features
imputer = joblib.load('model_files/imputer.joblib')
scaler = joblib.load('model_files/scaler.joblib')
with open('model_files/selected_features.json', 'r') as f:
    selected_features = json.load(f)['features']

# Load Wireshark CSV
wireshark_df = pd.read_csv('data/raw/wireshark_capture.csv')

# Map Wireshark columns to selected features (adjust based on your Wireshark export)
# Example mapping (modify based on your Wireshark CSV column names)
wireshark_mapping = {
    'Length': 'Length',  # Packet length
    'Source': 'Source',  # Source IP
    'Destination': 'Destination',  # Destination IP
    'Protocol': 'Protocol',  # Protocol type
    'Time': 'Time',  # Packet timestamp
    # Add more mappings for the remaining 5 features (e.g., 'Info', 'TTL', etc.)
    # For simplicity, assume the first 10 columns match; adjust as needed
}
available_columns = [col for col in wireshark_df.columns if col in wireshark_mapping.values()]
wireshark_df = wireshark_df[available_columns]

# Ensure the order matches the selected features
feature_df = wireshark_df[selected_features[:len(available_columns)]]

# Handle missing values and scale
X_processed = imputer.transform(feature_df)
X_processed = scaler.transform(X_processed)

# Save to processed_features.csv
pd.DataFrame(X_processed, columns=selected_features[:len(available_columns)]).to_csv('data/processed/processed_features.csv', index=False)
print("Processed features saved to data/processed/processed_features.csv")


# Add to preprocess_data.py after saving to CSV
import websocket
df = pd.read_csv('data/processed/processed_features.csv')
data = df.to_dict(orient='records')
ws = websocket.WebSocket()
ws.connect('ws://localhost:8765')
ws.send(json.dumps(data))
response = ws.recv()
print(f"Inference result: {response}")
ws.close()