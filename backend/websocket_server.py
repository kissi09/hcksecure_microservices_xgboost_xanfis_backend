import asyncio
import websockets
import json
import torch
import numpy as np
import joblib
from xanfis import GdAnfisClassifier
from datetime import datetime
import requests
import xgboost as xgb
from preprocess_data import preprocess_features
import pandas as pd

# Define selected features based on your preprocessing pipeline (must match training)
selected_features = [
    'rst_flag_counts',
    'packet_IAT_total',
    'bwd_payload_bytes_variance',
    'bwd_packets_rate',
    'min_header_bytes',
    'bwd_packets_IAT_min',
    'fwd_ack_flag_counts',
    'payload_bytes_std',
    'mean_header_bytes',
    'dst_port'
]

# Supabase configuration
SUPABASE_URL = "https://gjybvdpxopxforyufwdt.supabase.co/rest/v1/intrusion_alerts"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdqeWJ2ZHB4b3B4Zm9yeXVmd2R0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTI0MDQ4NDcsImV4cCI6MjA2Nzk4MDg0N30.LUIcyYV9F-Upky3ynt0OQiQU4DTO_53mggSv2tJHbYw"
headers = {
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "apiKey": SUPABASE_KEY,
    "Content-Type": "application/json",
    "Prefer": "return=minimal"
}

# Load models and preprocessing objects once at startup
imputer = joblib.load('model_files/imputer.joblib')
scaler = joblib.load('model_files/scaler.joblib')

# Rebuild ANFIS model architecture
anfis_model = GdAnfisClassifier(
    num_rules=4,
    mf_class="Gaussian",
    act_output=None,
    vanishing_strategy="blend",
    reg_lambda=None,
    epochs=0,
    batch_size=1,
    optim="Adam",
    optim_params={"lr": 0.01},
    early_stopping=False,
    n_patience=10,
    epsilon=0.001,
    valid_rate=0.1,
    seed=42,
    verbose=False
)

n_features = 10  # ← Set this to your actual number of features
dummy_X = np.random.rand(100, n_features)
dummy_y = np.random.randint(0, 2, size=100)
anfis_model.fit(dummy_X, dummy_y)

# Now load your real weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state_dict = torch.load('model_files/nf_model_1.pth', map_location=device)
anfis_model.network.load_state_dict(state_dict)
anfis_model.network.to(device).eval()

xgb_model = xgb.XGBClassifier()
xgb_model.load_model("model_files/xgb_model_1.json")
meta_learner = joblib.load('model_files/meta_learner.joblib')

async def handle_connection(websocket):
    async for message in websocket:
        try:
            print(f"Received data: {message}")
            feature_dict = json.loads(message)

            # === STEP 1: Preprocess the features (imputation + scaling) ===
            df_features = pd.DataFrame([{feat: feature_dict.get(feat, 0) for feat in selected_features}])
            X_imputed = imputer.transform(df_features)
            X_scaled = scaler.transform(X_imputed)

            # === STEP 2: Convert to tensor for ANFIS ===
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

            # === STEP 3: Get predictions ===
            with torch.no_grad():
                anfis_preds = torch.sigmoid(anfis_model.network(X_tensor)).cpu().numpy().flatten()
            xgb_preds = xgb_model.predict_proba(X_scaled)[:, 1]
            stacked_preds = np.column_stack((anfis_preds, xgb_preds))
            meta_pred = meta_learner.predict(stacked_preds)[0]

            # === STEP 4: Prepare record for Supabase ===
            record = {
                "timestamp": str(datetime.now()),
                "data": feature_dict,
                "prediction": int(meta_pred),
                "message": "Attack detected" if meta_pred == 1 else "Benign"
            }

            # === STEP 5: Send to Supabase ===
            response = requests.post(SUPABASE_URL, headers=headers, json=record)
            if response.status_code != 201:
                print(f"Failed to insert into Supabase: {response.text}")

            # === STEP 6: Return prediction to frontend ===
            await websocket.send(json.dumps({
                "prediction": int(meta_pred),
                "message": record["message"]
            }))

        except Exception as e:
            print(f"Error: {e}")
            await websocket.send(json.dumps({"error": str(e)}))

async def main():
    async with websockets.serve(handle_connection, 'localhost', 8765):
        print("WebSocket server started on ws://localhost:8765")
        await asyncio.Future()  # run forever

asyncio.run(main())
