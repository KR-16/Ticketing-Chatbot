"""API tests with a stubbed predictor.

No trained bundle, no model download, no network - the lifespan loader is
monkeypatched so these run anywhere in seconds.
"""

import pytest
from fastapi.testclient import TestClient

import src.api.main as api_main


class StubPredictor:
    classes = ["Change", "Incident", "Problem", "Request"]

    def predict(self, subject="", body=""):
        if "RAISE_VALUE_ERROR" in subject:
            raise ValueError("Ticket is empty after cleaning")
        return {
            "predicted_type": "Incident",
            "confidence": 0.9,
            "probabilities": {
                "Change": 0.05, "Incident": 0.9,
                "Problem": 0.03, "Request": 0.02,
            },
        }


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api_main, "_load_predictor", lambda: StubPredictor())
    with TestClient(api_main.app) as test_client:
        yield test_client


@pytest.fixture
def client_without_model(monkeypatch):
    def failing_loader():
        raise FileNotFoundError("No inference bundle at models/bundle")

    monkeypatch.setattr(api_main, "_load_predictor", failing_loader)
    with TestClient(api_main.app) as test_client:
        yield test_client


def test_predict_happy_path(client):
    response = client.post(
        "/predict",
        json={"subject": "Server down", "body": "It crashed this morning"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_type"] == "Incident"
    assert payload["confidence"] == pytest.approx(0.9)
    assert set(payload["probabilities"]) == set(StubPredictor.classes)


def test_predict_rejects_empty_ticket(client):
    response = client.post("/predict", json={"subject": "", "body": "  "})
    assert response.status_code == 422


def test_predict_rejects_wrong_types(client):
    response = client.post("/predict", json={"subject": 123, "body": ["nope"]})
    assert response.status_code == 422


def test_predict_rejects_oversized_body(client):
    response = client.post(
        "/predict", json={"subject": "hi", "body": "x" * 20001}
    )
    assert response.status_code == 422


def test_predictor_valueerror_becomes_400(client):
    response = client.post(
        "/predict", json={"subject": "RAISE_VALUE_ERROR", "body": ""}
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "classes": StubPredictor.classes}


def test_health_503_without_model(client_without_model):
    response = client_without_model.get("/health")
    assert response.status_code == 503
    assert "No inference bundle" in response.json()["detail"]


def test_predict_503_without_model(client_without_model):
    response = client_without_model.post(
        "/predict", json={"subject": "hi", "body": "there"}
    )
    assert response.status_code == 503
