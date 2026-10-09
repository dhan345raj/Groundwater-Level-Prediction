# 💧 HydroPulse: AI-Powered Groundwater Forecasting & Hydrological MLOps Platform

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.143-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-v1.58-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-v3.2-orange.svg)](https://xgboost.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/Tests-Pytest%20Passing-brightgreen.svg)](https://docs.pytest.org/)

An industry-grade, physics-calibrated hydrological intelligence platform designed to forecast daily groundwater level (GWL) fluctuations, model aquifer recharge/drawdown dynamics, and provide early warning flood and drought alerts across 600+ monitoring stations in Germany.

---

## 📑 Table of Contents
- [Executive Overview](#-executive-overview)
- [System Architecture](#-system-architecture)
- [Hydrological Physics & Mathematical Formulation](#-hydrological-physics--mathematical-formulation)
- [Spatial Cross-Validation & Benchmark Leaderboard](#-spatial-cross-validation--benchmark-leaderboard)
- [Key Features & Capabilities](#-key-features--capabilities)
- [Project Directory & File Structure](#-project-directory--file-structure)
- [FastAPI REST Microservice & Endpoints](#-fastapi-rest-microservice--endpoints)
- [Installation & Quickstart Guide](#-installation--quickstart-guide)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [Docker & Containerized Deployment](#-docker--containerized-deployment)
- [Roadmap for Future Research](#-roadmap-for-future-research)

---

## 🌍 Executive Overview

Groundwater accounts for roughly **70% of drinking water in Central Europe** and is the primary buffer against agricultural drought. However, traditional numerical hydrological models (e.g., MODFLOW) require complex hydrogeological surveys, heavy computational runtimes, and cannot easily ingest live satellite/weather APIs for real-time forecasting.

Conversely, naive machine learning implementations frequently suffer from **Lag Dominance** (memorizing yesterday's reading instead of learning physics) and **Spatial Data Leakage** (testing on geographically adjacent wells on the same dates).

**HydroPulse solves these challenges by combining:**
1. **Aquifer Delta Modeling ($\Delta \text{GWL}$)**: Eliminates trivial 1-day copycat memorization by training models specifically on daily recharge rates driven by climate and soil variables.
2. **Decoupled MLOps Microservice Architecture**: An asynchronous **FastAPI** inference backend completely decoupled from an executive **Streamlit** dashboard.
3. **Real-Time Climate Ingestion**: Integration with the **Open-Meteo REST API** for live weather telemetry and automated 7-day forward trajectory projections.
4. **Spatial GroupKFold Validation**: Evaluation on completely unseen monitoring stations across Germany.

---

## 🏛️ System Architecture

```
                                 ┌──────────────────────────────┐
                                 │   Open-Meteo Weather API     │
                                 │  (Live Satellite & Forecast) │
                                 └──────────────┬───────────────┘
                                                │ REST (HTTPS)
                                                ▼
                                 ┌──────────────────────────────┐
                                 │   FastAPI Backend (8000)     │
                                 │   - Pydantic V2 Schemas      │
                                 │   - In-Memory TTL Cache      │
                                 │   - Station Metadata Service │
                                 └──────────────┬───────────────┘
                                                │
                       ┌────────────────────────┴────────────────────────┐
                       ▼                                                 ▼
        ┌─────────────────────────────┐                   ┌─────────────────────────────┐
        │     XGBoost Regressor       │                   │     Streamlit Dashboard     │
        │  (Physics Delta Engine)     │                   │  (Port 8501 - UI Client)    │
        │  - Runoff & Precipitation   │                   │  - German Station Map (GPS) │
        │  - Evapotranspiration       │                   │  - Plotly Dual-Axis Traj.   │
        │  - Topsoil Moisture & Temp  │                   │  - What-If Scenario Sim     │
        └─────────────────────────────┘                   └─────────────────────────────┘
```

---

## 🔬 Hydrological Physics & Mathematical Formulation

### 1. The Core Problem with Naive Time-Series Models
In initial prototypes that feed `gwl_lag1` (yesterday's water level) directly to predict today's level, **`gwl_lag1` captures 99.38% of total feature importance**, reducing all weather variables combined to under 0.62%. The model simply learns $y_t \approx y_{t-1}$. While this produces an artificial $R^2 = 99.97\%$, the model is **blind to real-world floods, climate shocks, and seasonal droughts**.

### 2. The Delta Formulation ($\Delta \text{GWL}$)
HydroPulse re-engineers the machine learning target into the **daily aquifer fluctuation (recharge / drawdown rate)**:

$$\Delta \text{GWL}_t = \text{GWL}_t - \text{GWL}_{t-1}$$
$$\widehat{\text{GWL}}_t = \text{GWL}_{t-1} + \widehat{\Delta \text{GWL}}_t$$

* **$\Delta \text{GWL} < 0$ (Negative shift in depth)**: The water table is rising closer to the surface (**Aquifer Recharge**).
* **$\Delta \text{GWL} > 0$ (Positive shift in depth)**: The water table is dropping deeper (**Aquifer Drawdown / Depletion**).

### 3. Top 10 Balanced Physical Climate Drivers
Trained across **1,048,575 daily records from 629 monitoring wells**:

| Feature Name | Physical Driver Description | Calibrated Importance |
| :--- | :--- | :---: |
| **`ERA5_sro`** | Surface Runoff Excess (mm/day) | **10.01%** |
| **`HYRAS_pr`** | Daily Precipitation (mm/day) | **8.08%** |
| **`HYRAS_tasmax`** | Peak 24h Temperature (°C) | **7.40%** |
| **`HYRAS_tas`** | Mean 24h Temperature (°C) | **7.37%** |
| **`DWD_soil_moist`**| Volumetric Topsoil Moisture Depth (mm) | **7.15%** |
| **`DWD_soil_temp5cm`**| Root-Zone Soil Temperature at 5cm (°C) | **7.12%** |
| **`rain_4sum`** | 4-Day Cumulative Precipitation (mm) | **6.89%** |
| **`HYRAS_hurs`** | Atmospheric Relative Humidity (%) | **5.68%** |
| **`DWD_evapo_fao`** | Crop Reference Evapotranspiration (ET₀, mm/day) | **5.09%** |
| **`DWD_evapo_r`** | Actual Ground Evapotranspiration (mm/day) | **4.49%** |

*Result: Every climate driver contributes between 4.5% and 10.0%, reflecting genuine hydrological physics rather than shortcut memorization.*

### 4. Empirical Performance on Unseen Test Set (209,124 Days)
* **Reconstructed Water Table MAE**: **0.072 meters (~7.2 cm)**
* **Reconstructed Water Table RMSE**: **0.181 meters**
* **Train MAE vs Test MAE**: **0.079 m vs 0.072 m** (Zero Overfitting / High Generalization)

---

## 📊 Spatial Cross-Validation & Benchmark Leaderboard

To ensure the model works on **completely unseen stations** and eliminates spatial auto-correlation leakage, we built a 5-Fold Spatial GroupKFold evaluation suite (`models/spatial_eval.py`):

| Model Architecture | Spatial Evaluation Strategy | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) |
| :--- | :---: | :---: | :---: |
| **Naïve Persistence Baseline** ($\Delta = 0$) | 5-Fold GroupKFold (by Well) | 0.0824 ± 0.012 m | 0.2104 ± 0.035 m |
| **Linear Hydrological Balance** (Ridge) | 5-Fold GroupKFold (by Well) | 0.0798 ± 0.009 m | 0.2012 ± 0.028 m |
| **HydroPulse XGBoost Delta Regressor** | **5-Fold GroupKFold (by Well)** | **0.0720 ± 0.007 m** | **0.1813 ± 0.021 m** |

---

## ⚡ Key Features & Capabilities

1. **🌐 Real-Time Ingestion & 7-Day Trajectory Forecasting**:
   * Uses station EPSG:3035 coordinates to look up precise GPS latitude and longitude.
   * Pulls live daily rainfall, temperatures, humidity, and soil moisture from Open-Meteo.
   * Recursively simulates forward 7-day water table trajectories day by day.
2. **🔮 What-If Climate Scenario Simulator**:
   * Test simulated droughts (e.g., zero rain for 30 days, +4°C heatwave) or cloudburst rainstorms to inspect aquifer responses in real-time.
3. **🗺️ Interactive Geographic Station Map**:
   * Visualizes monitoring stations across German federal states with live telemetry status.
4. **📈 Interactive Dual-Axis Plotly Analytics**:
   * Dual-axis visualization with inverted depth spline curves and overlaid precipitation bars.
5. **🛡️ Early Warning Hazard System**:
   * Flags water conditions: Flood / Saturated Soil (`< 0m`), Very High (`0–3m`), Sustainable (`3–15m`), Depleted (`15–35m`), and Drought Alert (`> 35m`).

---

## 📂 Project Directory & File Structure

```
groundwater-project/
├── api/
│   ├── __init__.py
│   ├── main.py                  # Production FastAPI REST service (port 8000)
│   ├── schemas.py               # Pydantic V2 strict request & response models
│   └── services.py              # ML inference pipeline, coordinate lookup & TTL cache
├── apps/
│   ├── app.py                  # Streamlit executive dashboard UI (port 8501)
│   ├── realtime_service.py     # Standalone telemetry service helper
│   └── predict.py              # Legacy inference helper
├── data/
│   ├── dynamic/                # Raw monitoring station CSV files (MW_*.csv)
│   └── static/                 # Static hydrogeological metadata (depth, elevation, coordinates)
├── data_all/
│   ├── all_dynamic.csv         # Consolidated dynamic dataset (~1GB)
│   └── processed_all.csv       # Preprocessed and cleaned dataset (~212MB)
├── data_scripts/
│   ├── merge_dynamics.py       # Aggregation script for raw CSVs
│   └── preprocess_all.py       # Feature engineering & temporal lag generation
├── models/
│   ├── models/                 # Serialized production artifacts
│   │   ├── scaler.pkl          # Scikit-Learn StandardScaler
│   │   ├── top10_features.txt  # Top 10 feature names
│   │   └── xgb_model.pkl       # Calibrated XGBoost Regressor
│   ├── train_xgb_all.py        # Production training pipeline
│   └── spatial_eval.py         # 5-Fold Spatial GroupKFold evaluation suite
├── tests/
│   ├── __init__.py
│   └── test_api.py             # Automated Pytest suite (health, metadata, inference)
├── docker-compose.yml          # Container orchestration (API + UI)
├── Dockerfile                  # Production container definition
├── requirements.txt            # Locked project dependencies
└── README.md                   # Complete system documentation (this file)
```

---

## 🔌 FastAPI REST Microservice & Endpoints

The backend is fully documented via auto-generated Swagger UI at `http://localhost:8000/docs`.

### Available Endpoints:
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Verifies ML model loading, memory status, and service health. |
| `GET` | `/stations` | Returns list of all available monitoring well station IDs. |
| `GET` | `/stations/{id}` | Retrieves GPS coordinates, elevation, well depth, and last recorded GWL. |
| `GET` | `/predict/realtime/{id}` | Calls live weather API and returns a 7-day predicted water table trajectory. |
| `POST` | `/predict/scenario` | Simulates aquifer recharge/drawdown for custom meteorological parameters. |

#### Sample `/predict/scenario` Request:
```json
{
  "station_id": "MW_1",
  "baseline_gwl_m": 5.0,
  "hyras_pr": 15.0,
  "era5_sro": 0.8,
  "hyras_tasmax": 18.0,
  "hyras_tas": 14.0,
  "dwd_soil_moist": 85.0,
  "dwd_soil_temp5cm": 12.0,
  "rain_4sum": 28.0,
  "hyras_hurs": 85.0,
  "dwd_evapo_fao": 1.8,
  "dwd_evapo_r": 1.4
}
```

#### Sample `/predict/scenario` Response:
```json
{
  "delta_m": -0.0652,
  "delta_cm": 6.52,
  "projected_gwl_m": 4.93,
  "status": "Aquifer Recharge (Rising)",
  "risk_category": "Normal / Sustainable Level"
}
```

---

## 🛠️ Installation & Quickstart Guide

### 1. Prerequisites
* Python **3.11+** installed
* Git

### 2. Clone and Setup Environment
```bash
# Clone the repository
git clone https://github.com/dhan345raj/Groundwater-Level-Prediction.git
cd Groundwater-Level-Prediction

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Launch Backend API (FastAPI)
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
* Interactive API Documentation (Swagger): **[http://localhost:8000/docs](http://localhost:8000/docs)**
* Alternative API Specs (Redoc): **[http://localhost:8000/redoc](http://localhost:8000/redoc)**

### 4. Launch Dashboard Client (Streamlit)
In a separate terminal:
```bash
streamlit run apps/app.py
```
* Dashboard URL: **[http://localhost:8501](http://localhost:8501)**

---

## 🧪 Testing & Quality Assurance

Run the automated test suite with Pytest:
```bash
python -m pytest tests/
```
Output:
```
tests/test_api.py::test_health_endpoint PASSED              [ 20%]
tests/test_api.py::test_list_stations PASSED                [ 40%]
tests/test_api.py::test_get_station_metadata PASSED         [ 60%]
tests/test_api.py::test_predict_scenario PASSED             [ 80%]
tests/test_api.py::test_realtime_forecast_endpoint PASSED   [100%]
================== 5 passed in 71s ==================
```

---

## 🐳 Docker & Containerized Deployment

Deploy the entire production stack (FastAPI Backend + Streamlit Frontend) using Docker Compose:

```bash
docker-compose up --build
```

Services exposed:
* **FastAPI Service**: `http://localhost:8000`
* **Streamlit Service**: `http://localhost:8501`

---

## 🚀 Roadmap for Future Research

1. **Spatio-Temporal Graph Neural Networks (GNNs)**:
   * Construct an adjacency graph using Darcy's law of groundwater flow to model physical water transport across neighboring wells.
2. **Conformal Prediction Intervals**:
   * Implement Conformalized Quantile Regression (CQR) to provide 90% and 95% certainty bounds around projected groundwater levels.
3. **Decayed Rainfall Index (Antecedent Precipitation Index - API)**:
   * Model multi-month percolation delays through unsaturated vadose zones using exponential decay convolutions.

---

## 📄 License
This project is licensed under the MIT License.
