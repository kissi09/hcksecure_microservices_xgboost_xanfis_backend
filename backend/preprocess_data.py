import numpy as np
import joblib
import json

# Load preprocessing objects
imputer = joblib.load('model_files/imputer.joblib')
scaler = joblib.load('model_files/scaler.joblib')

# Load selected features
with open('model_files/selected_features.json', 'r') as f:
    selected_features = json.load(f)['features']

def preprocess_features(feature_dict):
    """
    Preprocess a dictionary of features for model input.
    
    Args:
        feature_dict (dict): Dictionary containing feature names and values from capture_traffic.py.
    
    Returns:
        np.ndarray: Preprocessed feature array ready for model inference.
    """
    # Extract features in the order specified by selected_features
    features = [feature_dict.get(feat, 0) for feat in selected_features]
    features = np.array(features).reshape(1, -1)
    
    # Apply imputation and scaling
    features = imputer.transform(features)
    features = scaler.transform(features)
    
    return features

# Example usage (for testing)
if __name__ == "__main__":
    # Simulate a feature dictionary received from WebSocket
    sample_feature_dict = {
        "rst_flag_counts": 0,
        "packet_IAT_total": 100,
        "bwd_payload_bytes_variance": 0,
        "bwd_packets_rate": 0,
        "min_header_bytes": 20,
        "bwd_packets_IAT_min": 0,
        "fwd_ack_flag_counts": 1,
        "payload_bytes_std": 0,
        "mean_header_bytes": 20,
        "dst_port": 5000
    }
    preprocessed = preprocess_features(sample_feature_dict)
    print("Preprocessed features:", preprocessed)