# web_socket.py
import asyncio
import websockets
import json
import os
import logging
from datetime import datetime
import joblib
import json
import numpy as np
import pandas as pd
import torch
import requests
import shap
import xgboost as xgb
import lightgbm as lgb
from xanfis import GdAnfisClassifier

# Optional GCS client (for uploading logs)
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except Exception:
    GCS_AVAILABLE = False

# -----------------------------
# Configuration & logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("websocket_server")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_files")

# -----------------------------
# Inline Supabase configuration (hardcoded as requested)
# -----------------------------
SUPABASE_URL = "https://gjybvdpxopxforyufwdt.supabase.co/rest/v1/intrusion_alerts"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdqeWJ2ZHB4b3B4Zm9yeXVmd2R0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTI0MDQ4NDcsImV4cCI6MjA2Nzk4MDg0N30.LUIcyYV9F-Upky3ynt0OQiQU4DTO_53mggSv2tJHbYw"

SUPABASE_HEADERS = {
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "apiKey": SUPABASE_KEY,
    "Content-Type": "application/json",
    "Prefer": "return=minimal"
}

# -----------------------------
# Inline Google Cloud bucket name (hardcoded)
# -----------------------------
GCS_BUCKET_NAME = "hcksecure_bucket"

if GCS_BUCKET_NAME and not GCS_AVAILABLE:
    logger.warning("google-cloud-storage not available; GCS uploads will be skipped.")

# -----------------------------
# Helper: safe joblib load
# -----------------------------
def try_joblib_load(path):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            logger.exception("Failed to joblib.load(%s): %s", path, e)
    return None

# -----------------------------
# Load selected_features.json (expected 60 features)
BASE_DIR = os.path.dirname(__file__)
selected_features_path = os.path.join(BASE_DIR, "model_files", "selected_features.json")

if not os.path.exists(selected_features_path):
    logger.error("selected_features.json not found in model_files/. This file is required.")
    raise FileNotFoundError("selected_features.json not found")
with open(selected_features_path, "r") as fh:
    selected_features = json.load(fh)
logging.info(f"Loaded selected_features.json ({len(selected_features['features'])} features)")



with open("model_files/selected_features.json") as f:
    data = json.load(f)

# Extract the list inside "features"
selected_features = data["features"]

top_15_features = selected_features[:15]  # now slicing works
logger.info("Using top_15_features (first 15 of selected_features) for ANFIS.")

# -----------------------------
# Load preprocessing objects
# -----------------------------
imputer = try_joblib_load(os.path.join(MODEL_DIR, "imputer.joblib"))
if imputer is None:
    logger.error("imputer.joblib not found/loaded. This is required.")
    raise FileNotFoundError("imputer.joblib not found")

scaler = try_joblib_load(os.path.join(MODEL_DIR, "scaler.joblib"))
if scaler is None:
    logger.error("scaler.joblib not found/loaded. This is required.")
    raise FileNotFoundError("scaler.joblib not found")

# ANFIS-specific scaler (optional)
anfis_scaler = try_joblib_load(os.path.join(MODEL_DIR, "anfis_scaler.joblib"))
if anfis_scaler is None:
    logger.info("No anfis_scaler.joblib found; will attempt to use general scaler sliced to top-15.")

# -----------------------------
# Load XGBoost model (joblib preferred, else JSON)
# -----------------------------
xgb_joblib_path = os.path.join(MODEL_DIR, "xgb_model.joblib")
xgb_json_path = os.path.join(MODEL_DIR, "xgb_model_1.json")

if os.path.exists(xgb_joblib_path):
    xgb_model = try_joblib_load(xgb_joblib_path)
    logger.info("Loaded XGBoost from joblib.")
elif os.path.exists(xgb_json_path):
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(xgb_json_path)
    logger.info("Loaded XGBoost from JSON.")
else:
    logger.error("No XGBoost model found (xgb_model.joblib or xgb_model_1.json).")
    raise FileNotFoundError("XGBoost model not found")

# -----------------------------
# Load LightGBM model (joblib preferred, else JSON/Booster)
# -----------------------------
lgb_joblib_path = os.path.join(MODEL_DIR, "lgb_model.joblib")
lgb_json_path = os.path.join(MODEL_DIR, "lgb_model.json")

lgb_model = None
if os.path.exists(lgb_joblib_path):
    lgb_model = try_joblib_load(lgb_joblib_path)
    logger.info("Loaded LightGBM from joblib.")
elif os.path.exists(lgb_json_path):
    try:
        # Try Booster load and wrap
        booster = lgb.Booster(model_file=lgb_json_path)
        # Build a minimal wrapper object with predict method that matches interface
        class _BoosterWrapper:
            def __init__(self, booster):
                self._booster = booster
            def predict_proba(self, X):
                # booster.predict returns probabilities for positive class for binary
                probs = self._booster.predict(X)
                # return shape (n,2) like sklearn: [1-p, p]
                probs = np.asarray(probs).reshape(-1)
                return np.vstack([1 - probs, probs]).T
        lgb_model = _BoosterWrapper(booster)
        logger.info("Loaded LightGBM Booster from JSON and wrapped for predict_proba.")
    except Exception as e:
        logger.exception("Failed to load LightGBM from JSON: %s", e)
        lgb_model = None
else:
    logger.warning("No LightGBM model found (lgb_model.joblib or lgb_model.json). LightGBM will be skipped.")

# -----------------------------
# Load ANFIS model (prefer joblib, else rebuild and load state_dict)
# -----------------------------
anfis_state_path = os.path.join(MODEL_DIR, "anfis_state.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
anfis_model = None

# Force skipping joblib load
logger.info("Skipping ANFIS .joblib load — rebuilding architecture instead.")

# Make sure top_15_features is valid
if not top_15_features or not isinstance(top_15_features, list):
    raise ValueError("top_15_features is missing or invalid. Cannot initialize ANFIS.")

n_features = len(top_15_features)
logger.info(f"Initializing ANFIS with {n_features} input features...")

# Build architecture with training hyperparameters
anfis_model = GdAnfisClassifier(
    num_rules=20,
    mf_class='Gaussian',
    epochs=0,           # no training
    batch_size=1,
    optim='Adam',
    optim_params={'lr': 1e-3},
    early_stopping=False,
    verbose=False
)

try:
    dummy_X = np.random.rand(20, n_features)  # Enough samples
    dummy_y = np.array([0, 1] * 10)
    anfis_model.fit(dummy_X, dummy_y)
except Exception as e:
    logger.exception("Dummy initialization failed: %s", e)
    if anfis_model.network is None:
        # Force-create a network if fit fails
        anfis_model._build_network(n_features, 2)


# Load state dict if available
if os.path.exists(anfis_state_path):
    try:
        state_dict = torch.load(anfis_state_path, map_location=device)
        anfis_model.network.load_state_dict(state_dict)
        anfis_model.network.to(device).eval()
        logger.info(f"Loaded ANFIS state dict from {anfis_state_path}")
    except Exception as e:
        logger.exception("Failed to load ANFIS state dict: %s", e)
else:
    logger.warning("No ANFIS state file found; model will remain untrained.")

# -----------------------------
# Load meta-learner (required for your stacked model)
# -----------------------------
meta_path = os.path.join(MODEL_DIR, "meta_learner.joblib")
meta_learner = None
meta_present = False
if os.path.exists(meta_path):
    meta_learner = try_joblib_load(meta_path)
    if meta_learner is not None:
        meta_present = True
        logger.info("Loaded meta_learner.joblib.")
    else:
        logger.warning("meta_learner.joblib exists but failed to load; fallback will be used.")
else:
    logger.warning("meta_learner.joblib not found; fallback averaging will be used.")

# -----------------------------
# Load SHAP explainer if present, else build TreeExplainer
# -----------------------------
shap_path = os.path.join(MODEL_DIR, "shap_explainer.joblib")
if os.path.exists(shap_path):
    shap_explainer = try_joblib_load(shap_path)
    if shap_explainer is not None:
        logger.info("Loaded SHAP explainer from joblib.")
    else:
        logger.warning("shap_explainer.joblib exists but failed to load; rebuilding.")
        shap_explainer = shap.TreeExplainer(xgb_model)
else:
    shap_explainer = shap.TreeExplainer(xgb_model)
    logger.info("Built SHAP TreeExplainer from XGBoost model.")

# -----------------------------
# GCS uploader helper (optional)
# -----------------------------
async def upload_log_to_gcs_async(log_data: dict):
    if not GCS_BUCKET_NAME or not GCS_AVAILABLE:
        return None
    def _upload():
        try:
            client = storage.Client()
            bucket = client.bucket(GCS_BUCKET_NAME)
            key = f"logs/{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}.json"
            blob = bucket.blob(key)
            blob.upload_from_string(json.dumps(log_data), content_type="application/json")
            # Make public (bucket policy may already make objects public)
            try:
                blob.make_public()
            except Exception:
                pass
            return blob.public_url
        except Exception as e:
            logger.exception("GCS upload failed: %s", e)
            return None
    return await asyncio.to_thread(_upload)

# -----------------------------
# Compute meta prediction helper
# -----------------------------
def compute_meta_prediction(stacked_row: np.ndarray):
    """
    stacked_row: shape (n_models,) or (1, n_models)
    Returns: (pred_label:int, prob:float)
    Expected meta stacking order: [XGBoost, ANFIS, LightGBM] (matches your training)
    """
    stacked_row = np.asarray(stacked_row).reshape(1, -1)
    if meta_present and meta_learner is not None:
        try:
            prob = float(meta_learner.predict_proba(stacked_row)[:, 1][0])
            lab = int(meta_learner.predict(stacked_row)[0])
            return lab, prob
        except Exception as e:
            logger.exception("meta_learner.predict failed, falling back to average: %s", e)
    # fallback: average probabilities
    avg = float(np.mean(stacked_row))
    label = int(avg > 0.5)
    return label, avg

# -----------------------------
# Blocking POST helper (run in thread)
# -----------------------------
def post_to_supabase_sync(url, headers, payload, timeout=10):
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        return r.status_code, r.text
    except Exception as e:
        logger.exception("Supabase POST failed: %s", e)
        return None, str(e)

# -----------------------------
# WebSocket handler
# -----------------------------
async def handle_connection(websocket):
    async for message in websocket:
        try:
            logger.info("Received message")
            feature_dict = json.loads(message)

            # Build DataFrame in the exact selected_features order
            df_features = pd.DataFrame([{feat: feature_dict.get(feat, 0) for feat in selected_features}])

            # Imputer & scaler: run in thread to avoid blocking event loop
            X_imputed = await asyncio.to_thread(imputer.transform, df_features)
            X_scaled = await asyncio.to_thread(scaler.transform, X_imputed)  # used by XGB & LGB

            # Prepare ANFIS input: select top 15 features and apply ANFIS scaler if present
            X_top15 = pd.DataFrame(X_imputed, columns=selected_features)[top_15_features].values
            if anfis_scaler is not None:
                X_anfis_scaled = await asyncio.to_thread(anfis_scaler.transform, X_top15)
            else:
                try:
                    X_anfis_scaled = X_scaled[:, :len(top_15_features)]
                except Exception:
                    X_anfis_scaled = X_top15  # fallback

            # ANFIS prediction
            try:
                X_tensor = torch.tensor(np.asarray(X_anfis_scaled, dtype=np.float32), dtype=torch.float32).to(device)
                with torch.no_grad():
                    net_out = anfis_model.network(X_tensor)
                    if isinstance(net_out, torch.Tensor):
                        raw = net_out.cpu().numpy().reshape(-1)
                        anfis_probs = 1.0 / (1.0 + np.exp(-raw))  # apply sigmoid if needed
                    else:
                        anfis_probs = np.asarray(net_out).reshape(-1)
            except Exception as e:
                logger.exception("ANFIS inference failed: %s", e)
                anfis_probs = np.zeros((X_anfis_scaled.shape[0],), dtype=float)

            # XGBoost prediction (uses X_scaled)
            try:
                xgb_probs = xgb_model.predict_proba(X_scaled)[:, 1]
            except Exception as e:
                logger.exception("XGBoost inference failed: %s", e)
                xgb_probs = np.zeros((X_scaled.shape[0],), dtype=float)

            # LightGBM prediction (if available)
            if lgb_model is not None:
                try:
                    # If LGB model loaded as LGBMClassifier object, use predict_proba
                    if hasattr(lgb_model, "predict_proba"):
                        lgb_probs = lgb_model.predict_proba(X_scaled)[:, 1]
                    else:
                        # If loaded as Booster wrapper, predict_proba is provided by wrapper above
                        lgb_probs = lgb_model.predict_proba(X_scaled)[:, 1]
                except Exception as e:
                    logger.exception("LightGBM inference failed: %s", e)
                    lgb_probs = np.zeros((X_scaled.shape[0],), dtype=float)
            else:
                # If no LGB model available, use zeros so stacking still works
                lgb_probs = np.zeros((X_scaled.shape[0],), dtype=float)

            # Order of stacked predictions must match meta training:
            # stacked_train = np.column_stack((xgb_train_preds, anfis_train_preds, lgb_train_preds))
            stacked_row = np.column_stack((xgb_probs, anfis_probs, lgb_probs))[0, :]  # single-row

            # Compute meta prediction and probability
            meta_label, meta_prob = compute_meta_prediction(stacked_row)

            # SHAP values (non-blocking)
            try:
                explainer_output = shap_explainer(df_features)  # returns Explanation
                shap_vals = explainer_output.values[0]
                shap_dict = {feat: float(shap_vals[i]) for i, feat in enumerate(selected_features)}
            except Exception as e:
                logger.exception("SHAP computation failed: %s", e)
                shap_dict = {feat: 0.0 for feat in selected_features}

            # Prepare payload/record
            record = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "data": feature_dict,
                "prediction": int(meta_label),
                "probability": float(meta_prob),
                "message": "Attack detected" if int(meta_label) == 1 else "Benign",
                "shap_values": shap_dict
            }

            # Post to Supabase in a thread (non-blocking)
            status, resp_text = await asyncio.to_thread(post_to_supabase_sync, SUPABASE_URL, SUPABASE_HEADERS, record)
            if status is None or status not in (200, 201):
                logger.warning("Supabase insert may have failed: status=%s resp=%s", status, resp_text)
            else:
                logger.info("Inserted into Supabase (status %s)", status)

            # Optionally upload to GCS for public logs
            if GCS_BUCKET_NAME and GCS_AVAILABLE:
                public_url = await upload_log_to_gcs_async(record)
                if public_url:
                    logger.info("Uploaded log to GCS: %s", public_url)
                    record["gcs_url"] = public_url

            # Send result back to the websocket client
            out = {"prediction": int(meta_label), "probability": float(meta_prob), "message": record["message"]}
            await websocket.send(json.dumps(out))

        except Exception as e:
            logger.exception("Error handling message: %s", e)
            try:
                await websocket.send(json.dumps({"error": str(e)}))
            except Exception:
                pass

# -----------------------------
# Main
# -----------------------------
async def main():
    host = os.environ.get("WS_HOST", "localhost")
    port = int(os.environ.get("WS_PORT", 8765))
    logger.info("Starting WebSocket server on %s:%d", host, port)
    async with websockets.serve(handle_connection, host, port):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
