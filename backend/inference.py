# inference.py
import sys
import json
import torch
import numpy as np
import joblib
from ANFIS.anfis import AnfisNet
import requests
from datetime import datetime

# Load models and preprocessing objects
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
imputer = joblib.load('model_files/imputer.joblib')
scaler = joblib.load('model_files/scaler.joblib')
anfis_model = AnfisNet("IDS Model", invardefs, outvarnames, rule_base).to(device)
anfis_model.load_state_dict(torch.load('model_files/anfis_model.pth'))
anfis_model.to(device).eval()
xgb_model = joblib.load('model_files/xgb_model.joblib')
meta_learner = joblib.load('model_files/meta_learner.joblib')

# Supabase configuration
SUPABASE_URL = "https://weuyfkdrvrrqbuyadswd.supabase.co/rest/v1/intrusion_alerts"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndldXlma2RydnJycWJ1eWFkc3dkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDk0Nzk0NDYsImV4cCI6MjA2NTA1NTQ0Nn0.FvvluOEUDohvBhWy8lhAi2DmT3OJG7fOc8x6e68tu3o"  # Replace with your Supabase API key
headers = {
    "Authorization": f"Bearer {eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndldXlma2RydnJycWJ1eWFkc3dkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDk0Nzk0NDYsImV4cCI6MjA2NTA1NTQ0Nn0.FvvluOEUDohvBhWy8lhAi2DmT3OJG7fOc8x6e68tu3o}",
    "apiKey": eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndldXlma2RydnJycWJ1eWFkc3dkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDk0Nzk0NDYsImV4cCI6MjA2NTA1NTQ0Nn0.FvvluOEUDohvBhWy8lhAi2DmT3OJG7fOc8x6e68tu3o,
    "Content-Type": "application/json",
    "Prefer": "return=minimal"
}

# Process input data
data = json.loads(sys.argv[1])
features = np.array(data)  # Assume data is a list of feature vectors

# Preprocess
X = imputer.transform(features)
X = scaler.transform(X)
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

# Get predictions
with torch.no_grad():
    anfis_preds = torch.sigmoid(anfis_model(X_tensor)).cpu().numpy().flatten()
xgb_preds = xgb_model.predict_proba(X)[:, 1]
stacked_preds = np.column_stack((anfis_preds, xgb_preds))
meta_preds = meta_learner.predict(stacked_preds)

# Prepare and send to Supabase
for i, pred in enumerate(meta_preds):
    record = {
        "timestamp": str(datetime.now()),
        "data": data[i].tolist() if isinstance(data[i], np.ndarray) else data[i],
        "prediction": int(pred),
        "message": "Attack detected" if pred == 1 else "Benign"
    }
    response = requests.post(SUPABASE_URL, headers=headers, json=record)
    if response.status_code != 201:
        print(f"Failed to insert into Supabase: {response.text}")

print(json.dumps({"predictions": meta_preds.tolist()}))