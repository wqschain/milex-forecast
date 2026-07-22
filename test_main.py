from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_forecast_valid_country():
    response = client.post("/forecast", json=
        {"country": "Canada", 
         "years_ahead": 5
         })
    
    assert response.status_code == 200
    data = response.json()
    assert data["country"] == "Canada"
    assert len(data["forecast"]) == 5


def test_predict_invalid_input():
    response = client.post("/forecast", json=
        {"country": "Wakanda", 
         "years_ahead": 5
         })
    
    assert response.status_code == 404

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

