from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "service" in response.json()

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

def test_predict_positive():
    response = client.post(
        "/predict",
        json={"text": "This movie was absolutely amazing! I loved it."}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "positive"
    assert data["confidence"] > 0.5

def test_predict_negative():
    response = client.post(
        "/predict",
        json={"text": "This was the worst movie I've ever seen. Terrible."}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "negative"
    assert data["confidence"] > 0.5

def test_predict_empty_text():
    response = client.post(
        "/predict",
        json={"text": ""}
    )
    assert response.status_code == 422

def test_batch_predict():
    response = client.post(
        "/batch-predict",
        json={
            "texts": [
                "Great movie!",
                "Terrible film.",
                "It was okay."
            ]
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 3

def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "test_accuracy" in data
    assert data["test_accuracy"] > 0.85