#!/bin/bash

# Ensure the script is executable
chmod +x "$0"

# Activate virtual environment (if used)
# source venv/bin/activate  # Uncomment and adjust if using a virtual environment

echo "Starting dummy_webapp.py..."
python backend/dummy_webapp.py &
WEBAPP_PID=$!
sleep 2  # Wait for the web app to start

echo "Starting websocket_server.py..."
python backend/websocket_server.py &
WEBSOCKET_PID=$!
sleep 2  # Wait for the WebSocket server to start

echo "Starting capture_traffic.py..."
python backend/capture_traffic.py &
CAPTURE_PID=$!

# Wait for all background processes to finish
wait $WEBAPP_PID $WEBSOCKET_PID $CAPTURE_PID

echo "All services stopped."
