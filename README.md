# XANFIS_XGBOOST Intrusion Detection System (IDS)

## Overview
A real-time IDS using a stacking ensemble model (XGBoost + XANFIS with logistic regression meta-learner).

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Ensure model files are in `model_files/`.
3. Configure Supabase and WebSocket settings in `config/config.json`.

## Running the System
1. Start the web app: `python src/webapp/dummy_webapp.py`
2. Start the WebSocket server: `python src/backend/websocket_server.py`
3. Capture traffic: `python src/backend/capture_traffic.py`

##Alternative 
1. Use the bash file run.sh to automate the process :)

## Directory Structure
- `data/`: Raw and processed datasets.
- `model_files/`: Trained models and preprocessing objects.
- `backend/`: Source code for backend and web app written in python.
- `tests/`: Unit and integration tests.
- `scripts/`: Training and simulation scripts.
- `config/`: Configuration files.
- `logs/`: Log files.
