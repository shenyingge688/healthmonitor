import requests
r = requests.post("http://127.0.0.1:8000/api/predict", json={"ecg": [0.0]*180000}, timeout=30)
print("status:", r.status_code)
print("text:", r.text[-500:] if len(r.text)>500 else r.text)
