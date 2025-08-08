import json
import asyncio
import websockets
from nfstream import NFStreamer
import numpy as np

# Load selected features from the JSON file
with open('model_files/selected_features.json', 'r') as f:
    selected_features = json.load(f)['features']

# Function to send features to WebSocket server
async def send_to_websocket(features):
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps(features))
        print(f"Sent features: {features}")

# Function to compute features from a flow
def compute_features(flow):
    features = {}

    features['rst_flag_counts'] = getattr(flow, 'rst_count', 0)
    features['packet_IAT_total'] = getattr(flow, 'flow_duration', 0)
    features['bwd_payload_bytes_variance'] = getattr(flow, 'dst2src_stddev_ps', 0) ** 2
    features['bwd_packets_rate'] = (getattr(flow, 'dst2src_packets', 0) /
                                    (getattr(flow, 'flow_duration', 1) / 1000))
    features['min_header_bytes'] = min(
        getattr(flow, 'src2dst_min_ip_size', 0),
        getattr(flow, 'dst2src_min_ip_size', 0)
    )
    features['bwd_packets_IAT_min'] = getattr(flow, 'dst2src_min_iat', 0)
    features['fwd_ack_flag_counts'] = getattr(flow, 'ack_count', 0)
    features['payload_bytes_std'] = np.mean([
        getattr(flow, 'src2dst_stddev_ps', 0),
        getattr(flow, 'dst2src_stddev_ps', 0)
    ])
    features['mean_header_bytes'] = 20  # Placeholder
    features['dst_port'] = getattr(flow, 'destination_port', 0)

    
    # Ensure all selected features are included, even if not computed
    for feat in selected_features:
        if feat not in features:
            features[feat] = 0  # Default value for uncomputed features
    
    return features

# Function to capture traffic and process flows
def capture_traffic():
    print("[+] Starting traffic capture...")

    try:
        # Initialize NFStreamer to capture traffic
        streamer = NFStreamer(
            source="\\Device\\NPF_{E63C9493-7134-4236-867B-0056329F1B86}",
            statistical_analysis=True
        )

        for flow in streamer:
            features = compute_features(flow)
            print(f"[>] Flow captured: {features}")  # DEBUG: Show the computed features
            asyncio.run(send_to_websocket(features))

    except Exception as e:
        print(f"[!] Error during capture: {e}")

# Main execution
if __name__ == "__main__":
    capture_traffic()