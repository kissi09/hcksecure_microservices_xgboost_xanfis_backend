# web_socket.py (batch-capable, ANFIS init fixed)

import asyncio
import json
import logging
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import requests
import shap
import torch
import websockets
import xgboost as xgb
import lightgbm as lgb
from xanfis import GdAnfisClassifier

# Optional GCS client
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except Exception:
    GCS_AVAILABLE = False

# -----------------------------
# Config
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("websocket_server")

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model_files")

# FIXED: Remove trailing spaces from URL
SUPABASE_URL = "https://gjybvdpxopxforyufwdt.supabase.co/rest/v1/intrusion_alerts"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdqeWJ2ZHB4b3B4Zm9yeXVmd2R0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTI0MDQ4NDcsImV4cCI6MjA2Nzk4MDg0N30.LUIcyYV9F-Upky3ynt0OQiQU4DTO_53mggSv2tJHbYw"

SUPABASE_HEADERS = {
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "apiKey": SUPABASE_KEY,
    "Content-Type": "application/json",
    "Prefer": "return=minimal"
}

# -----------------------------
# Helpers
# -----------------------------
def try_joblib_load(path):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            logger.exception("Failed to joblib.load(%s): %s", path, e)
    return None

def compute_meta_prediction(stacked_row, meta_learner=None):
    row = np.asarray(stacked_row).reshape(1, -1)
    if meta_learner is not None:
        try:
            prob = float(meta_learner.predict_proba(row)[:, 1][0])
            lab = int(meta_learner.predict(row)[0])
            return lab, prob
        except Exception:
            pass
    avg = float(np.mean(row))
    return int(avg > 0.5), avg

async def persist_async(record):
    try:
        # FIXED: Properly format the record for Supabase
        supabase_record = {
            "timestamp": record.get("timestamp"),
            "data": record.get("data", {}),
            "prediction": record.get("prediction"),
            "probability": record.get("probability"),
            "message": record.get("message"),
            "shap_values": record.get("shap_values", {})
        }
        
        r = await asyncio.to_thread(
            requests.post, 
            SUPABASE_URL, 
            json=supabase_record,  # Don't double-encode JSON
            headers=SUPABASE_HEADERS
        )
        if r.status_code not in (200, 201):
            logger.warning("Supabase insert failed: %s", r.text)
        else:
            logger.debug("Successfully inserted record into Supabase")
    except Exception as e:
        logger.exception("Supabase insert error: %s", e)

# -----------------------------
# Load features & models
# -----------------------------
with open(os.path.join(MODEL_DIR, "selected_features.json"), "r") as fh:
    sel = json.load(fh)
selected_features = sel["features"] if isinstance(sel, dict) else sel
top_15_features = selected_features[:15]

imputer = try_joblib_load(os.path.join(MODEL_DIR, "imputer.joblib"))
scaler = try_joblib_load(os.path.join(MODEL_DIR, "scaler.joblib"))
anfis_scaler = try_joblib_load(os.path.join(MODEL_DIR, "anfis_scaler.joblib"))

xgb_model = try_joblib_load(os.path.join(MODEL_DIR, "xgb_model.joblib"))
if xgb_model is None:
    raise FileNotFoundError("XGBoost model not found")

lgb_model = try_joblib_load(os.path.join(MODEL_DIR, "lgb_model.joblib"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
anfis_model = GdAnfisClassifier(num_rules=20, mf_class='Gaussian', epochs=0, batch_size=1, verbose=False)

# ANFIS init with dummy fit
n_features_anfis = len(top_15_features)
try:
    dummy_X = np.random.rand(20, n_features_anfis)
    dummy_y = np.array([0, 1] * 10)
    anfis_model.fit(dummy_X, dummy_y)
except Exception as e:
    logger.exception("ANFIS dummy init failed: %s", e)

anfis_state_path = os.path.join(MODEL_DIR, "anfis_state.pth")
if os.path.exists(anfis_state_path):
    state_dict = torch.load(anfis_state_path, map_location=device)
    anfis_model.network.load_state_dict(state_dict)
    anfis_model.network.to(device).eval()
    logger.info("Loaded ANFIS state dict.")

shap_explainer = shap.TreeExplainer(xgb_model)
meta_learner = try_joblib_load(os.path.join(MODEL_DIR, "meta_learner.joblib"))

# -----------------------------
# Inference
# -----------------------------
def run_batch_inference(batch):
    df = pd.DataFrame([{feat: row.get(feat, 0) for feat in selected_features} for row in batch])
    X_imputed = imputer.transform(df)
    X_scaled = scaler.transform(X_imputed)
    X_scaled_df = pd.DataFrame(X_scaled, columns=selected_features)

    X_top15 = pd.DataFrame(X_imputed, columns=selected_features)[top_15_features]
    if anfis_scaler is not None:
        X_anfis_scaled = anfis_scaler.transform(X_top15)
    else:
        X_anfis_scaled = X_scaled_df[top_15_features].values

    # ANFIS batch
    X_tensor = torch.tensor(np.asarray(X_anfis_scaled, dtype=np.float32), dtype=torch.float32).to(device)
    with torch.no_grad():
        raw = anfis_model.network(X_tensor).cpu().numpy().reshape(-1)
        anfis_probs = 1.0 / (1.0 + np.exp(-raw))

    # XGBoost batch
    xgb_probs = xgb_model.predict_proba(X_scaled_df)[:, 1]

    # LightGBM batch
    if lgb_model is not None:
        lgb_probs = lgb_model.predict_proba(X_scaled_df)[:, 1]
    else:
        lgb_probs = np.zeros_like(xgb_probs)

    stacked = np.column_stack((xgb_probs, anfis_probs, lgb_probs))
    if meta_learner is not None:
        probs = meta_learner.predict_proba(stacked)[:, 1]
        labels = meta_learner.predict(stacked)
    else:
        probs = np.mean(stacked, axis=1)
        labels = (probs > 0.5).astype(int)

    # SHAP batch (explainer handles multiple rows)
    try:
        shap_vals = shap_explainer(X_scaled_df).values
        shap_dicts = [{feat: float(shap_vals[j, i]) for i, feat in enumerate(selected_features)} for j in range(len(batch))]
    except Exception:
        shap_dicts = [{feat: 0.0 for feat in selected_features} for _ in batch]

    records = []
    for i in range(len(batch)):
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "data": batch[i],
            "prediction": int(labels[i]),
            "probability": float(probs[i]),
            "message": "Attack detected" if labels[i] == 1 else "Benign",
            "shap_values": shap_dicts[i]
        }
        records.append(record)
    return records

# -----------------------------
# WebSocket handler
# -----------------------------
MAX_BATCH_SIZE = 10000  # Limit batch size to prevent memory issues

async def handle_connection(websocket):
    logger.info("New WebSocket connection established")
    try:
        async for message in websocket:
            try:
                payload = json.loads(message)

                if isinstance(payload, list):  # Batch
                    batch_size = len(payload)
                    logger.info(f"Processing batch of {batch_size} rows")
                    
                    # Limit batch size to prevent memory issues
                    if batch_size > MAX_BATCH_SIZE:
                        error_msg = f"Batch size {batch_size} exceeds maximum allowed size {MAX_BATCH_SIZE}"
                        logger.warning(error_msg)
                        try:
                            await websocket.send(json.dumps({
                                "error": error_msg,
                                "max_allowed_batch_size": MAX_BATCH_SIZE,
                                "suggestion": "Split your data into smaller batches"
                            }))
                        except:
                            pass
                        continue
                    
                    results = await asyncio.to_thread(run_batch_inference, payload)
                    for rec in results:
                        asyncio.create_task(persist_async(rec))

                    attack_count = sum(r["prediction"] for r in results)
                    out = {"total_rows": len(results), "attack_count": attack_count, "benign_count": len(results)-attack_count}
                    
                    # Send with proper exception handling
                    try:
                        await websocket.send(json.dumps(out))
                        logger.info(f"Sent batch response: {len(results)} rows processed")
                    except websockets.exceptions.ConnectionClosed:
                        logger.info("WebSocket connection closed while sending batch response")
                    except Exception as e:
                        logger.error(f"Error sending WebSocket message: {e}")

                else:  # Single row
                    # Treat single row as a batch of one
                    results = await asyncio.to_thread(run_batch_inference, [payload])
                    record = results[0]
                    asyncio.create_task(persist_async(record))
                    
                    # Send with proper exception handling
                    try:
                        await websocket.send(json.dumps({
                            "prediction": record["prediction"],
                            "probability": record["probability"],
                            "message": record["message"]
                        }))
                        logger.info("Sent single row response")
                    except websockets.exceptions.ConnectionClosed:
                        logger.info("WebSocket connection closed while sending single response")
                    except Exception as e:
                        logger.error(f"Error sending WebSocket message: {e}")
                        
            except json.JSONDecodeError:
                logger.error("Invalid JSON received")
                try:
                    await websocket.send(json.dumps({"error": "Invalid JSON format"}))
                except:
                    pass
            except Exception as e:
                logger.exception("Error handling message: %s", e)
                try:
                    await websocket.send(json.dumps({"error": str(e)}))
                except:
                    pass
                    
    except websockets.exceptions.ConnectionClosedError as e:
        if "message too big" in str(e).lower():
            logger.warning(f"Client sent message too large: {e}")
        else:
            logger.info(f"WebSocket connection closed: {e}")
    except websockets.exceptions.ConnectionClosed:
        logger.info("WebSocket connection closed normally")
    except Exception as e:
        logger.exception("Unexpected error in WebSocket connection: %s", e)
    finally:
        logger.info("WebSocket connection handler finished")

# -----------------------------
# Main
# -----------------------------
async def main():
    host = os.environ.get("WS_HOST", "localhost")
    port = int(os.environ.get("WS_PORT", 8765))
    logger.info("Starting WebSocket server on %s:%d", host, port)
    
    # FIXED: Increase message size limits to handle large batches
    async with websockets.serve(
        handle_connection, 
        host, 
        port, 
        max_size=2**30,  # 1GB limit for large messages
        max_queue=32
    ):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())