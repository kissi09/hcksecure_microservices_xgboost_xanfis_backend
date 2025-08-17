# capture_replay.py (batch mode)
import os
import json
import time
import asyncio
import websockets
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("csv_replayer")

# -----------------------------
# Load selected features list
# -----------------------------
BASE_DIR = os.path.dirname(__file__)  # folder where capture_replay.py is located
MODEL_DIR = os.path.join(BASE_DIR, "..", "model_files")

with open(os.path.join(MODEL_DIR, "selected_features.json"), "r") as f:
    sel = json.load(f)
SELECTED_FEATURES = sel["features"] if isinstance(sel, dict) and "features" in sel else sel

# -----------------------------
# WebSocket sender
# -----------------------------
async def send_to_websocket(batch_records):
    uri = "ws://localhost:8765"  # must match web_socket.py
    try:
        async with websockets.connect(uri, max_size=2**25) as websocket:  # allow big payloads
            await websocket.send(json.dumps(batch_records))
            logger.info(f"Sent batch of {len(batch_records)} rows")
    except Exception as e:
        logger.error(f"WebSocket send failed: {e}")

# -----------------------------
# Replay CSVs as batches
# -----------------------------
def replay_csvs(dataset_dir):
    files = [f for f in os.listdir(dataset_dir) if f.endswith(".csv")]
    files.sort()  # ensure replay order

    for csv_file in files:
        path = os.path.join(dataset_dir, csv_file)
        logger.info(f"[+] Replaying file: {path}")

        try:
            df = pd.read_csv(path)
        except Exception as e:
            logger.error(f"Failed to read {path}: {e}")
            continue

        # Convert dataframe into list of dicts (rows with only selected features)
        batch = df[SELECTED_FEATURES].fillna(0).astype(float).to_dict(orient="records")

        # Send entire batch
        asyncio.run(send_to_websocket(batch))

        logger.info(f"[✓] Finished {csv_file}, sleeping 20s before next...")
        time.sleep(20)

if __name__ == "__main__":
    dataset_dir = os.path.join(BASE_DIR, "BCCC-CICIDS")
    replay_csvs(dataset_dir)
