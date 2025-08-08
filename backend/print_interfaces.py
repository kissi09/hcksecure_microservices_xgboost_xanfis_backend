import requests
import concurrent.futures
import time

URL = "http://127.0.0.1:5000/login"

# Simulated login data
data = {
    "student_id": "20250001",
    "password": "password123"
}

def send_post(i):
    try:
        response = requests.post(URL, data=data)
        print(f"Request {i} → Status: {response.status_code}")
    except Exception as e:
        print(f"Error on request {i}: {e}")

if __name__ == "__main__":
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        executor.map(send_post, range(1000))  # send 1000 POST requests
    print(f"Completed in {time.time() - start:.2f}s")
