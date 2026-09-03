import os
import sys

import pytest

# Ensure project root is on path for flask_app import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask_app.app import app, get_tracking_uri


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_home(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Welcome" in response.data


def test_predict(client):
    response = client.post(
        "/predict",
        json={
            "comments": [
                "This show is absolutely brilliant, loved every episode!",
                "Worst series ever, complete waste of time.",
            ]
        },
    )
    assert response.status_code == 200
    data = response.get_json()
    assert len(data) == 2
    assert "sentiment" in data[0]
    assert data[0]["sentiment"] in {"-1", "0", "1"}
    assert data[1]["sentiment"] in {"-1", "0", "1"}


def test_predict_with_timestamps(client):
    response = client.post(
        "/predict_with_timestamps",
        json={
            "comments": [
                {"text": "Squid Game is a masterpiece!", "timestamp": "2021-10-06T12:05:38Z"},
                {"text": "Not worth the hype at all.", "timestamp": "2021-10-06T12:06:38Z"},
            ]
        },
    )
    assert response.status_code == 200
    data = response.get_json()
    assert len(data) == 2
    assert "timestamp" in data[0]
    assert "sentiment" in data[0]


def test_generate_chart(client):
    response = client.post(
        "/generate_chart",
        json={"sentiment_counts": {"1": 10, "0": 5, "-1": 2}},
    )
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "image/png"


def test_generate_wordcloud(client):
    response = client.post(
        "/generate_wordcloud",
        json={"comments": ["Great show", "Amazing acting", "Terrible plot"]},
    )
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "image/png"


def test_generate_trend_graph(client):
    response = client.post(
        "/generate_trend_graph",
        json={
            "sentiment_data": [
                {"sentiment": 1, "timestamp": "2021-10-01"},
                {"sentiment": -1, "timestamp": "2021-10-02"},
                {"sentiment": 0, "timestamp": "2021-11-03"},
            ]
        },
    )
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "image/png"


def test_predict_rejects_non_list_comments(client):
    """'comments' must be a list; a string or object should be rejected with 400."""
    response = client.post("/predict", json={"comments": "not a list"})
    assert response.status_code == 400


def test_predict_with_timestamps_rejects_missing_text(client):
    """Each comment object must have a 'text' field."""
    response = client.post(
        "/predict_with_timestamps",
        json={"comments": [{"timestamp": "2021-10-01"}]},
    )
    assert response.status_code == 400


def test_predict_with_timestamps_rejects_non_object(client):
    """Each comment must be an object, not a bare string."""
    response = client.post(
        "/predict_with_timestamps",
        json={"comments": ["just a string"]},
    )
    assert response.status_code == 400


def test_get_tracking_uri_falls_back_when_env_unset(monkeypatch):
    """An empty MLFLOW_TRACKING_URI must NOT override the local default."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    uri = get_tracking_uri()
    assert uri.startswith("sqlite:///")
    assert uri.endswith("mlflow.db")


def test_get_tracking_uri_ignores_empty_env(monkeypatch):
    """An empty-string MLFLOW_TRACKING_URI (as GitHub Actions sets unset secrets)
    must fall back to the local default rather than silently breaking tracking."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "")
    uri = get_tracking_uri()
    assert uri.startswith("sqlite:///")
    assert uri.endswith("mlflow.db")


def test_get_tracking_uri_respects_explicit_env(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:////custom/path/mlflow.db")
    assert get_tracking_uri() == "sqlite:////custom/path/mlflow.db"
