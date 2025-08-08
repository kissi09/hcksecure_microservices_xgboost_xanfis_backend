import requests
import time

url = "http://127.0.0.1:5000/login"

# Simulate login brute-force
for i in range(1000):
    response = requests.post(url, data={
        "student_id": "20250001",
        "password": f"wrongpassword{i}"
    })
    print(f"Attempt {i} - Status: {response.status_code}")
    time.sleep(0.05)  # Tune rate to simulate DoS
