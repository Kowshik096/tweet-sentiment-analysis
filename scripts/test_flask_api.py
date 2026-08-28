import pytest
import requests

BASE_URL = "http://localhost:5000"

def test_home():
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200
    assert "Welcome" in response.text

def test_predict():
    response = requests.post(
        f"{BASE_URL}/predict",
        json={"comments": ["This show is absolutely brilliant, loved every episode!",
                            "Worst series ever, complete waste of time."]},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert "sentiment" in data[0]
    assert data[0]["sentiment"] in {"-1", "0", "1"}
    assert data[1]["sentiment"] in {"-1", "0", "1"}

def test_predict_with_timestamps():
    response = requests.post(
        f"{BASE_URL}/predict_with_timestamps",
        json={"comments": [
            {"text": "Squid Game is a masterpiece!", "timestamp": "2021-10-06T12:05:38Z"},
            {"text": "Not worth the hype at all.", "timestamp": "2021-10-06T12:06:38Z"},
        ]},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert "timestamp" in data[0]
    assert "sentiment" in data[0]

def test_generate_chart():
    response = requests.post(
        f"{BASE_URL}/generate_chart",
        json={"sentiment_counts": {"1": 10, "0": 5, "-1": 2}},
    )
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "image/png"

def test_generate_wordcloud():
    response = requests.post(
        f"{BASE_URL}/generate_wordcloud",
        json={"comments": ["Great show", "Amazing acting", "Terrible plot"]},
    )
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "image/png"

def test_generate_trend_graph():
    response = requests.post(
        f"{BASE_URL}/generate_trend_graph",
        json={"sentiment_data": [
            {"sentiment": 1, "timestamp": "2021-10-01"},
            {"sentiment": -1, "timestamp": "2021-10-02"},
            {"sentiment": 0, "timestamp": "2021-11-03"},
        ]},
    )
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "image/png"
