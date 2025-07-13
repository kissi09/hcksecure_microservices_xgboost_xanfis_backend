#!/bin/bash

# Ensure the script is executable
chmod +x "$0"

# Activate virtual environment (adjust path as needed)
# source backend/venv/bin/activate  # Uncomment if using a virtual environment

# Function to check if a process is running
check_process() {
    local pid=$1
    local name=$2
    if ! ps -p $pid > /dev/null 2>&1; then
        echo "Error: $name failed to start or crashed."
        exit 1
    fi
}

echo "Starting dummy_webapp.py..."
python backend/dummy_webapp.py &
WEBAPP_PID=$!
sleep 2
check_process $WEBAPP_PID "dummy_webapp.py"

echo "Starting websocket_server.py..."
python backend/websocket_server.py &
WEBSOCKET_PID=$!
sleep 2
check_process $WEBSOCKET_PID "websocket_server.py"

echo "Starting capture_traffic.py..."
python backend/capture_traffic.py &
CAPTURE_PID=$!
sleep 2
check_process $CAPTURE_PID "capture_traffic.py"

# Wait for all background processes to finish
wait $WEBAPP_PID $WEBSOCKET_PID $CAPTURE_PID

echo "All services stopped."