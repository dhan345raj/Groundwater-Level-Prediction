from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["version"] == "2.0.0"

def test_list_stations():
    response = client.get("/stations")
    assert response.status_code == 200
    stations = response.json()
    assert isinstance(stations, list)
    assert len(stations) > 0
    assert "MW_1" in stations

def test_get_station_metadata():
    response = client.get("/stations/MW_1")
    assert response.status_code == 200
    meta = response.json()
    assert meta["station_id"] == "MW_1"
    assert "latitude" in meta
    assert "longitude" in meta
    assert "elevation_m" in meta
    assert "well_depth_m" in meta

def test_predict_scenario():
    payload = {
        "station_id": "MW_1",
        "baseline_gwl_m": 5.0,
        "era5_sro": 0.5,
        "hyras_pr": 12.0,  # Heavy rainfall
        "hyras_tasmax": 14.0,
        "hyras_tas": 10.0,
        "dwd_soil_moist": 85.0,
        "dwd_soil_temp5cm": 9.0,
        "rain_4sum": 25.0,
        "hyras_hurs": 88.0,
        "dwd_evapo_fao": 1.2,
        "dwd_evapo_r": 1.0
    }
    response = client.post("/predict/scenario", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert "delta_m" in res
    assert "projected_gwl_m" in res
    assert "status" in res
    assert "risk_category" in res

def test_realtime_forecast_endpoint():
    response = client.get("/predict/realtime/MW_1")
    assert response.status_code == 200
    data = response.json()
    assert data["station_id"] == "MW_1"
    assert len(data["forecast_days"]) == 7
    assert "net_7day_change_cm" in data
    assert "trajectory_status" in data
