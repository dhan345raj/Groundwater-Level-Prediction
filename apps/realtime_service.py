import os
import requests
import numpy as np
import pandas as pd
from pyproj import Transformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_FILE = os.path.join(BASE_DIR, "data", "static", "static_features_MW_1toMW_3207.csv")

# Coordinate transformer: EPSG:3035 (LAEA Europe) -> EPSG:4326 (WGS84 Lat/Lon)
_transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)

# Cache static coordinates
_coords_cache = None

def get_well_coordinates(mw_id: str):
    """
    Looks up the EPSG:3035 coordinates for a well and converts them to (lat, lon).
    Returns (latitude, longitude, elevation, depth).
    """
    global _coords_cache
    if _coords_cache is None:
        if os.path.exists(STATIC_FILE):
            try:
                df_static = pd.read_csv(
                    STATIC_FILE,
                    usecols=["MW_ID", "Easting (EPSG:3035)", "Northing (EPSG:3035)", "Elevation", "Depth"]
                )
                _coords_cache = df_static.set_index("MW_ID")
            except Exception:
                _coords_cache = {}
        else:
            _coords_cache = {}

    if _coords_cache is not None and mw_id in _coords_cache.index:
        row = _coords_cache.loc[mw_id]
        easting = row["Easting (EPSG:3035)"]
        northing = row["Northing (EPSG:3035)"]
        elevation = row["Elevation"] if "Elevation" in row and not pd.isna(row["Elevation"]) else 150.0
        depth = row["Depth"] if "Depth" in row and not pd.isna(row["Depth"]) else 20.0
        
        if not pd.isna(easting) and not pd.isna(northing):
            lon, lat = _transformer.transform(easting, northing)
            return round(lat, 4), round(lon, 4), float(elevation), float(depth)

    # Fallback to central Germany
    return 51.1657, 10.4515, 150.0, 20.0


def fetch_live_weather_and_forecast(latitude: float, longitude: float):
    """
    Fetches real-time weather observation and 7-day forecast from Open-Meteo REST API.
    Returns parsed daily forecast data dictionary or None on failure.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": [
            "precipitation_sum",
            "temperature_2m_max",
            "temperature_2m_mean",
            "et0_fao_evapotranspiration"
        ],
        "hourly": [
            "soil_moisture_0_to_7cm",
            "soil_temperature_0_to_7cm",
            "relative_humidity_2m"
        ],
        "timezone": "auto"
    }
    try:
        response = requests.get(url, params=params, timeout=8)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Failed to fetch live weather: {e}")
    return None


def run_realtime_forecast(model, scaler, top10_features, live_weather_json, baseline_gwl: float):
    """
    Feeds live weather observations and 7-day forecasts through the calibrated
    XGBoost model to compute the projected 7-day groundwater level trajectory.
    """
    daily = live_weather_json["daily"]
    hourly = live_weather_json["hourly"]
    dates = daily["time"]
    
    forecast_results = []
    current_gwl = float(baseline_gwl)
    
    # Track past 4-day rainfall sum dynamically
    past_rainfall_buffer = [float(daily["precipitation_sum"][0])] * 3

    for i in range(len(dates)):
        pr = float(daily["precipitation_sum"][i])
        tasmax = float(daily["temperature_2m_max"][i])
        tas = float(daily["temperature_2m_mean"][i])
        et_fao = float(daily["et0_fao_evapotranspiration"][i])
        
        # Midday hourly index
        h_idx = min(i * 24 + 12, len(hourly["soil_moisture_0_to_7cm"]) - 1)
        # Convert soil moisture volumetric m3/m3 to topsoil mm depth (~250mm root zone)
        sm_vol = float(hourly["soil_moisture_0_to_7cm"][h_idx])
        sm = sm_vol * 250.0
        stemp = float(hourly["soil_temperature_0_to_7cm"][h_idx])
        
        # Daily humidity mean
        h_slice = hourly["relative_humidity_2m"][i*24:(i+1)*24]
        hurs = float(np.mean(h_slice)) if len(h_slice) > 0 else 75.0
        
        # Hydrological approximations for physical features
        sro = max(0.0, 0.15 * (pr - 4.0)) if pr > 4.0 else 0.0
        past_rainfall_buffer.append(pr)
        rain_4sum = float(sum(past_rainfall_buffer[-4:]))
        et_r = max(0.0, et_fao * 0.85)

        feature_map = {
            "ERA5_sro": sro,
            "HYRAS_pr": pr,
            "HYRAS_tasmax": tasmax,
            "HYRAS_tas": tas,
            "DWD_soil_moist": sm,
            "DWD_soil_temp5cm": stemp,
            "rain_4sum": rain_4sum,
            "HYRAS_hurs": hurs,
            "DWD_evapo_fao": et_fao,
            "DWD_evapo_r": et_r
        }

        # Build DataFrame with explicit feature names for scikit-learn/XGBoost
        input_df = pd.DataFrame([[feature_map[f] for f in top10_features]], columns=top10_features)
        scaled_vec = scaler.transform(input_df)
        pred_delta = float(model.predict(scaled_vec)[0])
        
        # Update water table depth: Depth_today = Depth_yesterday + delta
        # Negative delta = rising (recharge); Positive delta = falling (drawdown)
        current_gwl += pred_delta
        
        is_rising = pred_delta < 0
        status_label = "Recharging" if is_rising else "Drawdown"
        
        forecast_results.append({
            "Date": dates[i],
            "Rain (mm)": round(pr, 1),
            "Max Temp (°C)": round(tasmax, 1),
            "Daily Change (cm)": round(-pred_delta * 100, 1) if is_rising else round(-pred_delta * 100, 1),
            "Delta (m)": round(pred_delta, 3),
            "Status": status_label,
            "Projected Depth (m)": round(current_gwl, 2)
        })
        
    return pd.DataFrame(forecast_results)
