from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    # Depending on test environment, model might not be loaded.
    assert "model_loaded" in data

def test_model_info():
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_type"] == "LightGBM"
    assert data["num_features"] == 17
    assert "6h" in data["horizons"]

def test_features_info():
    response = client.get("/model/features")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 17
    names = [f["name"] for f in data]
    assert "age" in names
    assert "lactate_mean" in names
