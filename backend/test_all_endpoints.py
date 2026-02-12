import sys
import os

# Ensure the backend directory is in the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_endpoints():
    print("Testing /health ...")
    resp = client.get("/health")
    print(f"Status: {resp.status_code}, Output: {resp.json()}\n")

    print("Testing /model/info ...")
    resp = client.get("/model/info")
    print(f"Status: {resp.status_code}, Output: {resp.json()}\n")

    print("Testing /model/features ...")
    resp = client.get("/model/features")
    print(f"Status: {resp.status_code}, Output: {resp.json()}\n")

    payload = {
        "patient_id": "P001",
        "horizon": "6h",
        "age": 65,
        "gender": "M",
        "lactate_mean": 2.8,
        "sbp_mean": 90
    }

    print("Testing /predict ...")
    resp = client.post("/predict", json=payload)
    print(f"Status: {resp.status_code}, Output: {resp.json()}\n")

    print("Testing /predict/batch ...")
    batch_payload = {
        "patients": [
            payload,
            {"patient_id": "P002", "horizon": "12h", "age": 45, "gender": "F"}
        ]
    }
    resp = client.post("/predict/batch", json=batch_payload)
    print(f"Status: {resp.status_code}, Output: {resp.json()}\n")

    print("Testing /predict/explain ...")
    resp = client.post("/predict/explain", json=payload)
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"Prediction: {data['prediction']}, Probability: {data['risk_probability']}")
        print(f"Top explanation: {data['explanation'][0] if data.get('explanation') else None}\n")
    else:
        print(f"Output: {resp.json()}\n")
        
    print("Testing /predict/window ...")
    window_payload = {
        "patient_id": "P_WIN",
        "horizon": "12h",
        "age in years": 60,
        "sbp array in mmHg": [120, 115, 110, 105, 100],
        "lactate array in mmol/L": [1.2, 1.5, 2.0, 3.1, 4.5]
    }
    resp = client.post("/predict/window", json=window_payload)
    print(f"Status: {resp.status_code}, Output: {resp.json()}\n")

    print("Testing /predict/trajectory ...")
    trajectory_payload = {
        "patient_id": "P_TRAJ",
        "horizon": "6h",
        "history": [
            {"patient_id": "P_TRAJ", "sbp_mean in mmHg": 120, "lactate_mean in mmol/L": 1.2},
            {"patient_id": "P_TRAJ", "sbp_mean in mmHg": 110, "lactate_mean in mmol/L": 2.5},
            {"patient_id": "P_TRAJ", "sbp_mean in mmHg": 95,  "lactate_mean in mmol/L": 4.5}
        ]
    }
    resp = client.post("/predict/trajectory", json=trajectory_payload)
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        traj = resp.json().get('trajectory', [])
        for i, t in enumerate(traj):
            print(f"  Hour {i+1} Risk: {t['risk_probability']} ({t['risk_level']})")
        print()
    else:
        print(f"Output: {resp.json()}\n")

if __name__ == "__main__":
    # In TestClient, startup events are run automatically when using a context manager.
    # To run the startup events, we need to do this:
    with TestClient(app) as client:
        test_endpoints()
