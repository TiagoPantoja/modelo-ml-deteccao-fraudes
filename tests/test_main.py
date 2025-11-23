from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/docs")
    assert response.status_code == 200


@patch("src.main.ml_models")
def test_predict_success(mock_ml_models):
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [[0.2, 0.8]]  # 80% fraude
    mock_ml_models.get.return_value = mock_model

    payload = {
        "Time": 0, "Amount": 100,
        "V_features": {f"V{i}": 0.0 for i in range(1, 29)}
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["is_fraud"] is True
    assert response.json()["fraud_probability"] == 0.8


@patch("src.main.ml_models")
def test_predict_model_not_loaded(mock_ml_models):
    mock_ml_models.get.return_value = None

    payload = {
        "Time": 0, "Amount": 100,
        "V_features": {f"V{i}": 0.0 for i in range(1, 29)}
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 503
    assert "Modelo não carregado" in response.json()["detail"]


@patch("src.main.ml_models")
def test_predict_missing_columns(mock_ml_models):
    mock_model = MagicMock()
    mock_ml_models.get.return_value = mock_model

    v_feats = {f"V{i}": 0.0 for i in range(1, 28)}
    payload = {
        "Time": 0, "Amount": 100,
        "V_features": v_feats
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 500
    assert "Erro no processamento" in response.json()["detail"]