from typing import List, Optional
from pydantic import BaseModel, Field

class StationMetadata(BaseModel):
    station_id: str = Field(..., description="Unique monitoring well identifier")
    latitude: float = Field(..., description="WGS84 latitude coordinate")
    longitude: float = Field(..., description="WGS84 longitude coordinate")
    elevation_m: float = Field(..., description="Surface elevation above sea level in meters")
    well_depth_m: float = Field(..., description="Total drilled well depth in meters")
    last_recorded_gwl_m: float = Field(..., description="Latest ground water table depth in meters")

class ForecastDay(BaseModel):
    date: str = Field(..., description="ISO 8601 calendar date")
    precipitation_mm: float = Field(..., description="Daily rainfall in mm")
    max_temperature_c: float = Field(..., description="Daily peak temperature in Celsius")
    daily_change_cm: float = Field(..., description="Daily water table shift in centimeters")
    delta_m: float = Field(..., description="Daily water table shift in meters")
    status: str = Field(..., description="Hydrological state: Recharging or Drawdown")
    projected_depth_m: float = Field(..., description="Projected water table depth below ground in meters")

class RealtimeForecastResponse(BaseModel):
    station_id: str
    latitude: float
    longitude: float
    baseline_gwl_m: float
    forecast_days: List[ForecastDay]
    net_7day_change_cm: float
    trajectory_status: str

class ScenarioRequest(BaseModel):
    station_id: Optional[str] = "MW_1"
    baseline_gwl_m: float = Field(5.0, ge=-10.0, le=150.0, description="Previous day groundwater level (m)")
    era5_sro: float = Field(0.0, ge=0.0, description="Surface runoff (mm/day)")
    hyras_pr: float = Field(0.0, ge=0.0, description="Daily precipitation (mm)")
    hyras_tasmax: float = Field(15.0, ge=-40.0, le=55.0, description="Max temperature (°C)")
    hyras_tas: float = Field(10.0, ge=-40.0, le=50.0, description="Mean temperature (°C)")
    dwd_soil_moist: float = Field(60.0, ge=0.0, description="Topsoil moisture depth (mm)")
    dwd_soil_temp5cm: float = Field(10.0, ge=-20.0, le=45.0, description="Soil temperature 5cm (°C)")
    rain_4sum: float = Field(0.0, ge=0.0, description="4-day cumulative rainfall (mm)")
    hyras_hurs: float = Field(75.0, ge=0.0, le=100.0, description="Relative humidity (%)")
    dwd_evapo_fao: float = Field(2.0, ge=0.0, description="Reference evapotranspiration (mm/day)")
    dwd_evapo_r: float = Field(1.5, ge=0.0, description="Actual evapotranspiration (mm/day)")

class ScenarioResponse(BaseModel):
    delta_m: float
    delta_cm: float
    projected_gwl_m: float
    status: str
    risk_category: str

class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool
    version: str = "2.0.0"
