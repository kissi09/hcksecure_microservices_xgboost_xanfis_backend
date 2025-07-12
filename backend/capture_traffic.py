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
    
    # Map nfstream attributes to the selected features
    features['rst_flag_counts'] = flow.rst_count
    features['packet_IAT_total'] = flow.flow_duration  # Total flow duration as an approximation
    features['bwd_payload_bytes_variance'] = (flow.dst2src_stddev_ps ** 2) if hasattr(flow, 'dst2src_stddev_ps') else 0  # Variance derived from stddev
    features['bwd_packets_rate'] = flow.dst2src_packets / (flow.flow_duration / 1000) if flow.flow_duration > 0 else 0  # Rate in packets per second
    features['min_header_bytes'] = min(flow.src2dst_min_ip_size, flow.dst2src_min_ip_size) if hasattr(flow, 'src2dst_min_ip_size') else 0  # Approximation
    features['bwd_packets_IAT_min'] = flow.dst2src_min_iat if hasattr(flow, 'dst2src_min_iat') else 0  # Minimum inter-arrival time for backward packets
    features['fwd_ack_flag_counts'] = flow.ack_count  # Assuming this represents forward ACK flags
    features['payload_bytes_std'] = np.mean([flow.src2dst_stddev_ps, flow.dst2src_stddev_ps]) if hasattr(flow, 'src2dst_stddev_ps') else 0  # Average stddev as approximation
    features['mean_header_bytes'] = 20  # Placeholder: assuming average TCP header size (not accurate, adjust if needed)
    features['dst_port'] = flow.destination_port
    
    # Ensure all selected features are included, even if not computed
    for feat in selected_features:
        if feat not in features:
            features[feat] = 0  # Default value for uncomputed features
    
    return features

# Function to capture traffic and process flows
def capture_traffic():
    # Initialize NFStreamer to capture traffic from the loopback interface
    streamer = NFStreamer(source="lo", statistical_analysis=True)
    for flow in streamer:
        features = compute_features(flow)
        asyncio.run(send_to_websocket(features))

# Main execution
if __name__ == "__main__":
    capture_traffic()