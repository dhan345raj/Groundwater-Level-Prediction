from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from api.schemas import (
    HealthResponse,
    StationMetadata,
    RealtimeForecastResponse,
    ScenarioRequest,
    ScenarioResponse
)
from api.services import (
    get_ml_artifacts,
    get_all_stations,
    execute_realtime_prediction
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up artifacts on startup
    get_ml_artifacts()
    get_all_stations()
    yield

app = FastAPI(
    title="HydroPulse MLOps Inference API",
    description="Physics-calibrated groundwater level prediction and 7-day hydrological forecasting service.",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Health check verifying ML model availability and API readiness."""
    model, scaler, _ = get_ml_artifacts()
    return HealthResponse(
        status="healthy" if (model and scaler) else "degraded",
        model_loaded=(model is not None and scaler is not None),
        version="2.0.0"
    )

@app.get("/stations", response_model=List[str], tags=["Metadata"])
def list_stations():
    """Returns all available monitoring well station IDs."""
    stations = get_all_stations()
    return sorted(list(stations.keys()))

@app.get("/stations/{station_id}", response_model=StationMetadata, tags=["Metadata"])
def get_station_details(station_id: str):
    """Returns GPS coordinates, elevation, depth, and latest water table for a station."""
    stations = get_all_stations()
    if station_id not in stations:
        raise HTTPException(status_code=404, detail=f"Station '{station_id}' not found")
    return stations[station_id]

@app.get("/predict/realtime/{station_id}", response_model=RealtimeForecastResponse, tags=["Inference"])
def get_realtime_forecast(station_id: str):
    """
    Ingests live weather observations and 7-day weather forecasts via Open-Meteo REST API
    and projects the forward 7-day groundwater level trajectory.
    """
    try:
        result = execute_realtime_prediction(station_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Forecast execution failed: {str(e)}")

@app.post("/predict/scenario", response_model=ScenarioResponse, tags=["Inference"])
def simulate_scenario(req: ScenarioRequest):
    """
    Simulates aquifer recharge/drawdown for custom meteorological and soil conditions.
    """
    model, scaler, top10 = get_ml_artifacts()
    if not model or not scaler:
        raise HTTPException(status_code=500, detail="ML model assets not initialized")

    feat_map = {
        "ERA5_sro": req.era5_sro,
        "HYRAS_pr": req.hyras_pr,
        "HYRAS_tasmax": req.hyras_tasmax,
        "HYRAS_tas": req.hyras_tas,
        "DWD_soil_moist": req.dwd_soil_moist,
        "DWD_soil_temp5cm": req.dwd_soil_temp5cm,
        "rain_4sum": req.rain_4sum,
        "HYRAS_hurs": req.hyras_hurs,
        "DWD_evapo_fao": req.dwd_evapo_fao,
        "DWD_evapo_r": req.dwd_evapo_r
    }
    
    input_df = pd.DataFrame([[feat_map[f] for f in top10]], columns=top10)
    scaled_vec = scaler.transform(input_df)
    pred_delta = float(model.predict(scaled_vec)[0])
    projected_gwl = req.baseline_gwl_m + pred_delta
    
    is_rise = pred_delta < 0
    status = "Aquifer Recharge (Rising)" if is_rise else "Drawdown (Falling)"
    
    if projected_gwl < 0:
        risk = "Critical Flood / Saturated Soil"
    elif projected_gwl <= 3.0:
        risk = "Very High Water Table"
    elif projected_gwl <= 15.0:
        risk = "Normal / Sustainable Level"
    elif projected_gwl <= 35.0:
        risk = "Depleted Water Table"
    else:
        risk = "Drought Alert"
        
    return ScenarioResponse(
        delta_m=round(pred_delta, 4),
        delta_cm=round(abs(pred_delta) * 100, 2),
        projected_gwl_m=round(projected_gwl, 2),
        status=status,
        risk_category=risk
    )
