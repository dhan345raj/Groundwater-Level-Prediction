import os
import time
from functools import lru_cache
import joblib
import numpy as np
import pandas as pd
import requests
from pyproj import Transformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "models", "xgb_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "models", "scaler.pkl")
TOP10_PATH = os.path.join(BASE_DIR, "models", "models", "top10_features.txt")
STATIC_FILE = os.path.join(BASE_DIR, "data", "static", "static_features_MW_1toMW_3207.csv")
DATA_FILE = os.path.join(BASE_DIR, "data_all", "processed_all.csv")

# Coordinate Transformer
_transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)

# Cache loaded model in memory
_model_cache = None
_scaler_cache = None
_top10_cache = None
_stations_cache = None

def get_ml_artifacts():
    global _model_cache, _scaler_cache, _top10_cache
    if _model_cache is None or _scaler_cache is None or _top10_cache is None:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            _model_cache = joblib.load(MODEL_PATH)
            _scaler_cache = joblib.load(SCALER_PATH)
            if os.path.exists(TOP10_PATH):
                _top10_cache = pd.read_csv(TOP10_PATH)["feature"].tolist()
            else:
                _top10_cache = [
                    "ERA5_sro", "HYRAS_pr", "HYRAS_tasmax", "HYRAS_tas",
                    "DWD_soil_moist", "DWD_soil_temp5cm", "rain_4sum",
                    "HYRAS_hurs", "DWD_evapo_fao", "DWD_evapo_r"
                ]
    return _model_cache, _scaler_cache, _top10_cache


def get_all_stations():
    global _stations_cache
    if _stations_cache is None:
        if os.path.exists(DATA_FILE):
            try:
                df = pd.read_csv(DATA_FILE, usecols=["MW_ID", "Date", "GWL"], parse_dates=["Date"])
                latest = df.sort_values("Date").groupby("MW_ID").last().reset_index()
                
                # Merge static coordinates
                if os.path.exists(STATIC_FILE):
                    sf = pd.read_csv(STATIC_FILE, usecols=["MW_ID", "Easting (EPSG:3035)", "Northing (EPSG:3035)", "Elevation", "Depth"])
                    merged = latest.merge(sf, on="MW_ID", how="left")
                else:
                    merged = latest
                    merged["Easting (EPSG:3035)"] = np.nan
                    merged["Northing (EPSG:3035)"] = np.nan
                    merged["Elevation"] = 150.0
                    merged["Depth"] = 20.0
                
                stations = {}
                for _, r in merged.iterrows():
                    mw = str(r["MW_ID"])
                    e, n = r["Easting (EPSG:3035)"], r["Northing (EPSG:3035)"]
                    if not pd.isna(e) and not pd.isna(n):
                        lon, lat = _transformer.transform(e, n)
                    else:
                        lat, lon = 51.1657, 10.4515
                    
                    elev = float(r["Elevation"]) if "Elevation" in r and not pd.isna(r["Elevation"]) else 150.0
                    depth = float(r["Depth"]) if "Depth" in r and not pd.isna(r["Depth"]) else 20.0
                    gwl = float(r["GWL"]) if "GWL" in r and not pd.isna(r["GWL"]) else 5.0
                    
                    stations[mw] = {
                        "station_id": mw,
                        "latitude": round(lat, 4),
                        "longitude": round(lon, 4),
                        "elevation_m": round(elev, 1),
                        "well_depth_m": round(depth, 1),
                        "last_recorded_gwl_m": round(gwl, 2)
                    }
                _stations_cache = stations
            except Exception as e:
                print(f"Error loading stations: {e}")
                _stations_cache = {}
        else:
            _stations_cache = {}
    return _stations_cache


# Weather cache with TTL (Time To Live in seconds)
_weather_cache = {}
CACHE_TTL_SECONDS = 600  # 10 minutes cache to prevent rate-limiting

def fetch_weather_cached(latitude: float, longitude: float):
    cache_key = (round(latitude, 3), round(longitude, 3))
    now = time.time()
    
    if cache_key in _weather_cache:
        data, ts = _weather_cache[cache_key]
        if now - ts < CACHE_TTL_SECONDS:
            return data
            
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
        res = requests.get(url, params=params, timeout=8)
        if res.status_code == 200:
            data = res.json()
            _weather_cache[cache_key] = (data, now)
            return data
    except Exception as e:
        print(f"Error calling Open-Meteo: {e}")
    return None


def execute_realtime_prediction(station_id: str):
    stations = get_all_stations()
    station = stations.get(station_id)
    if not station:
        lat, lon, elev, depth, gwl = 51.1657, 10.4515, 150.0, 20.0, 5.0
    else:
        lat = station["latitude"]
        lon = station["longitude"]
        gwl = station["last_recorded_gwl_m"]
        
    weather = fetch_weather_cached(lat, lon)
    if not weather:
        raise RuntimeError("External meteorological telemetry service unreachable")
        
    model, scaler, top10 = get_ml_artifacts()
    if model is None or scaler is None:
        raise RuntimeError("ML model assets not loaded")
        
    daily = weather["daily"]
    hourly = weather["hourly"]
    dates = daily["time"]
    
    forecast_days = []
    curr_gwl = gwl
    past_rainfall = [float(daily["precipitation_sum"][0])] * 3
    
    for i in range(len(dates)):
        pr = float(daily["precipitation_sum"][i])
        tasmax = float(daily["temperature_2m_max"][i])
        tas = float(daily["temperature_2m_mean"][i])
        et_fao = float(daily["et0_fao_evapotranspiration"][i])
        
        h_idx = min(i * 24 + 12, len(hourly["soil_moisture_0_to_7cm"]) - 1)
        sm = float(hourly["soil_moisture_0_to_7cm"][h_idx]) * 250.0
        stemp = float(hourly["soil_temperature_0_to_7cm"][h_idx])
        
        h_slice = hourly["relative_humidity_2m"][i*24:(i+1)*24]
        hurs = float(np.mean(h_slice)) if len(h_slice) > 0 else 75.0
        
        sro = max(0.0, 0.15 * (pr - 4.0)) if pr > 4.0 else 0.0
        past_rainfall.append(pr)
        rain_4sum = float(sum(past_rainfall[-4:]))
        et_r = max(0.0, et_fao * 0.85)
        
        feature_dict = {
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
        
        input_df = pd.DataFrame([[feature_dict[f] for f in top10]], columns=top10)
        scaled_vec = scaler.transform(input_df)
        pred_delta = float(model.predict(scaled_vec)[0])
        curr_gwl += pred_delta
        
        is_rising = pred_delta < 0
        status_str = "Recharging" if is_rising else "Drawdown"
        
        forecast_days.append({
            "date": dates[i],
            "precipitation_mm": round(pr, 1),
            "max_temperature_c": round(tasmax, 1),
            "daily_change_cm": round(-pred_delta * 100, 1),
            "delta_m": round(pred_delta, 3),
            "status": status_str,
            "projected_depth_m": round(curr_gwl, 2)
        })
        
    net_change_cm = round((curr_gwl - gwl) * 100, 1)
    trajectory_status = "Recharging (Rising)" if net_change_cm < 0 else "Drawdown (Falling)"
    
    return {
        "station_id": station_id,
        "latitude": lat,
        "longitude": lon,
        "baseline_gwl_m": gwl,
        "forecast_days": forecast_days,
        "net_7day_change_cm": net_change_cm,
        "trajectory_status": trajectory_status
    }
