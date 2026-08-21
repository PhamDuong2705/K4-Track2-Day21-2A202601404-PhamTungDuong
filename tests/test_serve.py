from fastapi.testclient import TestClient

from src import serve


class _FakeModel:
    def __init__(self, prediction=1):
        self.prediction = prediction

    def predict(self, rows):
        assert len(rows) == 1
        return [self.prediction]


def test_healthz_when_model_is_ready(monkeypatch):
    monkeypatch.setattr(serve, "model", _FakeModel())
    response = TestClient(serve.app).get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_score_returns_prediction_and_label(monkeypatch):
    monkeypatch.setattr(serve, "model", _FakeModel(prediction=1))
    response = TestClient(serve.app).post("/score", json={"features": [0] * 10})
    assert response.status_code == 200
    assert response.json() == {"prediction": 1, "label": "thu_nhap_cao"}


def test_score_rejects_wrong_feature_count(monkeypatch):
    monkeypatch.setattr(serve, "model", _FakeModel())
    response = TestClient(serve.app).post("/score", json={"features": [0] * 9})
    assert response.status_code == 400
    assert "Expected 10 features" in response.json()["detail"]


def test_healthz_fails_when_model_is_missing(monkeypatch):
    monkeypatch.setattr(serve, "model", None)
    response = TestClient(serve.app).get("/healthz")
    assert response.status_code == 503
