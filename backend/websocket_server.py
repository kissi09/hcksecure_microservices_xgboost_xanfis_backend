import asyncio
import websockets
import json
import torch
import numpy as np
import joblib
from xanfis import GdAnfisClassifier
from datetime import datetime
import requests
from preprocess_data import preprocess_features

# Supabase configuration
SUPABASE_URL = "https://weuyfkdrvrrqbuyadswd.supabase.co/rest/v1/intrusion_alerts"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndldXlma2RydnJycWJ1eWFkc3dkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDk0Nzk0NDYsImV4cCI6MjA2NTA1NTQ0Nn0.FvvluOEUDohvBhWy8lhAi2DmT3OJG7fOc8x6e68tu3o"
headers = {
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "apiKey": SUPABASE_KEY,
    "Content-Type": "application/json",
    "Prefer": "return=minimal"
}

# Load models and preprocessing objects once at startup
imputer = joblib.load('model_files/imputer.joblib')
scaler = joblib.load('model_files/scaler.joblib')
anfis_model = GdAnfisClassifier(n_inputs=10, n_rules=10, mf_type='gaussian')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
anfis_model.load_state_dict(torch.load('model_files/nf_model.pth'))
anfis_model.to(device).eval()
xgb_model = joblib.load('model_files/xgb_model.joblib')
meta_learner = joblib.load('model_files/meta_learner.joblib')

async def handle_connection(websocket, path):
    async for message in websocket:
        try:
            print(f"Received data: {message}")
            feature_dict = json.loads(message)
            
            # Preprocess the features using preprocess_data.py
            preprocessed_features = preprocess_features(feature_dict)
            
            # Convert to tensor for ANFIS
            X_tensor = torch.tensor(preprocessed_features, dtype=torch.float32).to(device)
            
            # Get predictions
            with torch.no_grad():
                anfis_preds = torch.sigmoid(anfis_model(X_tensor)).cpu().numpy().flatten()
            xgb_preds = xgb_model.predict_proba(preprocessed_features)[:, 1]
            stacked_preds = np.column_stack((anfis_preds, xgb_preds))
            meta_pred = meta_learner.predict(stacked_preds)[0]
            
            # Prepare record for Supabase
            record = {
                "timestamp": str(datetime.now()),
                "data": feature_dict,
                "prediction": int(meta_pred),
                "message": "Attack detected" if meta_pred == 1 else "Benign"
            }
            
            # Send to Supabase
            response = requests.post(SUPABASE_URL, headers=headers, json=record)
            if response.status_code != 201:
                print(f"Failed to insert into Supabase: {response.text}")
            
            # Send prediction back to client
            await websocket.send(json.dumps({"prediction": int(meta_pred), "message": record["message"]}))
        
        except Exception as e:
            print(f"Error: {e}")
            await websocket.send(json.dumps({"error": str(e)}))

# Start WebSocket server
start_server = websockets.serve(handle_connection, 'localhost', 8765)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()