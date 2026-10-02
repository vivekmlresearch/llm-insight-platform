from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_and_demo_mode():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["offline_only"] is True


def test_demo_trace_is_explicitly_labeled():
    response = client.post("/api/run", json={"prompt": "A model predicts", "max_new_tokens": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["demo"] is True
    assert len(body["steps"]) == 3
    assert body["steps"][0]["candidates"]
    assert "illustrative" in body["limitations"][0].lower()


def test_demo_intervention_has_guardrail():
    response = client.post("/api/intervene", json={"prompt": "Test prompt", "layer": 1})
    assert response.status_code == 200
    body = response.json()
    assert body["demo"] is True
    assert "illustrative" in body["note"].lower()


def test_generation_bounds_are_validated():
    response = client.post("/api/run", json={"prompt": "Test", "max_new_tokens": 1000})
    assert response.status_code == 422
