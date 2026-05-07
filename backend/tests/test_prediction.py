from fastapi.testclient import TestClient
from app.main import app
import pytest

client = TestClient(app)

# Dummy payload with all fields missing except patient_id
valid_payload = {
    "patient_id": "P001",
    "horizon": "6h"
}

def test_invalid_input_type():
    payload = {
        "patient_id": "P002",
        "age": "this_is_not_a_number"
    }
    response = client.post("/predict", json=payload)
    # Should fail pydantic validation
    assert response.status_code == 422
    data = response.json()
    assert data["error"] is True

def test_prediction_without_model_mocking():
    """
    If models aren't loaded in the test environment, this should return 503.
    If they are loaded, it should return 200.
    """
    response = client.post("/predict", json=valid_payload)
    
    if response.status_code == 503:
        assert response.json()["detail"] == "Model is currently unavailable"
    else:
        assert response.status_code == 200
        data = response.json()
        assert data["patient_id"] == "P001"
        assert "risk_probability" in data
        assert "top_contributing_feature" in data
