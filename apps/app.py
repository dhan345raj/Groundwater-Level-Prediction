import streamlit as st
import os
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Import real-time forecasting service
from realtime_service import (
    get_well_coordinates,
    fetch_live_weather_and_forecast,
    run_realtime_forecast
)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="HydroPulse | Groundwater AI Early Warning & Forecasting",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "models", "xgb_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "models", "scaler.pkl")
TOP10_PATH  = os.path.join(BASE_DIR, "models", "models", "top10_features.txt")
DATA_PATH   = os.path.join(BASE_DIR, "data_all", "processed_all.csv")

# ---------------- CACHED ASSET LOADERS ----------------
@st.cache_resource
def load_model_assets():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        if os.path.exists(TOP10_PATH):
            top10_df = pd.read_csv(TOP10_PATH)
            top10 = top10_df["feature"].tolist()
        else:
            top10 = [
                "ERA5_sro", "HYRAS_pr", "HYRAS_tasmax", "HYRAS_tas",
                "DWD_soil_moist", "DWD_soil_temp5cm", "rain_4sum",
                "HYRAS_hurs", "DWD_evapo_fao", "DWD_evapo_r"
            ]
        return model, scaler, top10
    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        return None, None, []

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH, parse_dates=["Date"], dayfirst=True)
        df = df.sort_values(["MW_ID", "Date"]).reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error loading historical dataset: {e}")
        return None

# Load model assets
model, scaler, top10 = load_model_assets()

# ---------------- METADATA & PHYSICAL UNITS ----------------
FEATURE_META = {
    "ERA5_sro": {
        "label": "Surface Runoff Excess",
        "unit": "mm/day",
        "desc": "Surface water runoff after ground infiltration"
    },
    "HYRAS_pr": {
        "label": "Daily Rainfall",
        "unit": "mm/day",
        "desc": "Total direct precipitation volume"
    },
    "HYRAS_tasmax": {
        "label": "Peak Daily Temperature",
        "unit": "°C",
        "desc": "Maximum 24-hour ambient temperature"
    },
    "HYRAS_tas": {
        "label": "Mean Daily Temperature",
        "unit": "°C",
        "desc": "Average 24-hour temperature"
    },
    "DWD_soil_moist": {
        "label": "Topsoil Moisture Depth",
        "unit": "mm",
        "desc": "Volumetric water retained in upper soil layer"
    },
    "DWD_soil_temp5cm": {
        "label": "Soil Temperature (5cm)",
        "unit": "°C",
        "desc": "Root-zone thermal ground condition"
    },
    "rain_4sum": {
        "label": "4-Day Cumulative Rain",
        "unit": "mm",
        "desc": "Multi-day rolling precipitation accumulation"
    },
    "HYRAS_hurs": {
        "label": "Relative Humidity",
        "unit": "%",
        "desc": "Atmospheric moisture concentration"
    },
    "DWD_evapo_fao": {
        "label": "Reference Evapotranspiration",
        "unit": "mm/day",
        "desc": "Crop reference evapotranspiration rate (FAO)"
    },
    "DWD_evapo_r": {
        "label": "Actual Evapotranspiration",
        "unit": "mm/day",
        "desc": "Actual surface water evaporated into air"
    }
}

# ---------------- PREMIUM MODERN STYLING ----------------
st.markdown("""
<style>
/* Global resets & typography */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Background */
.stApp {
    background: #F8FAFC;
}

/* Hero Header */
.hero-banner {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0369A1 100%);
    border-radius: 16px;
    padding: 28px 34px;
    color: white;
    margin-bottom: 22px;
    box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
    display: flex;
    justify-content: space-between;
    align-items: center;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.hero-title {
    font-size: 30px;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0;
    line-height: 1.2;
    background: linear-gradient(90deg, #FFFFFF, #E0F2FE);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    font-size: 14px;
    color: #94A3B8;
    margin-top: 6px;
    font-weight: 400;
}

.status-badge {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid #10B981;
    color: #34D399;
    padding: 6px 14px;
    border-radius: 30px;
    font-size: 12px;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    letter-spacing: 0.03em;
}

/* Modern Card */
.glass-card {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 22px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03), 0 2px 4px -2px rgba(0, 0, 0, 0.02);
    margin-bottom: 18px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.glass-card:hover {
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.06);
}

/* Metric Display Blocks */
.kpi-card {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 18px 20px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
    position: relative;
    overflow: hidden;
}

.kpi-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
}

.kpi-blue::before { background: #0284C7; }
.kpi-green::before { background: #10B981; }
.kpi-amber::before { background: #F59E0B; }
.kpi-purple::before { background: #8B5CF6; }

.kpi-title {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748B;
    margin-bottom: 4px;
}

.kpi-value {
    font-size: 26px;
    font-weight: 700;
    color: #0F172A;
    line-height: 1.2;
}

.kpi-sub {
    font-size: 12px;
    color: #94A3B8;
    margin-top: 4px;
}

/* Custom Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #E2E8F0;
    padding: 6px;
    border-radius: 12px;
    border-bottom: none;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 18px;
    font-weight: 600;
    font-size: 14px;
    color: #475569;
    border: none !important;
    background: transparent;
}

.stTabs [aria-selected="true"] {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.06);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%);
    color: white;
    font-weight: 600;
    font-size: 14px;
    padding: 10px 22px;
    border-radius: 10px;
    border: none;
    box-shadow: 0 4px 10px rgba(2, 132, 199, 0.3);
    transition: all 0.2s ease;
}

.stButton>button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 14px rgba(2, 132, 199, 0.4);
}
</style>
""", unsafe_allow_html=True)

# ---------------- HERO HEADER ----------------
st.markdown("""
<div class="hero-banner">
    <div>
        <h1 class="hero-title">💧 HydroPulse | AI Groundwater Intelligence</h1>
        <div class="hero-subtitle">Physics-Calibrated Telemetry Ingestion • Real-Time Aquifer Dynamics • 7-Day Forecasting</div>
    </div>
    <div>
        <span class="status-badge">
            <span style="font-size: 8px;">●</span> LIVE TELEMETRY CONNECTED
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# Load data
df = load_data()

# ---------------- SIDEBAR CONTROLS ----------------
with st.sidebar:
    st.markdown("### 🎛️ Station Selector & GPS")
    
    selected_mw = "MW_1"
    lat, lon, elev, depth = 53.42, 14.14, 43.5, 57.6
    last_recorded_gwl = 5.0
    
    if df is not None:
        wells = sorted(df["MW_ID"].unique())
        selected_mw = st.selectbox("Select Monitoring Station", wells)
        
        # Look up station coordinates and well specs
        lat, lon, elev, depth = get_well_coordinates(selected_mw)
        
        df_well = df[df["MW_ID"] == selected_mw].sort_values("Date")
        last_recorded_gwl = float(df_well["GWL"].iloc[-1])
        
        # Station profile card
        st.markdown(f"""
        <div style="background: white; border: 1px solid #E2E8F0; border-radius: 10px; padding: 14px; margin-top: 10px; font-size: 13px;">
            <div style="font-weight: 700; color: #0F172A; margin-bottom: 8px;">📍 Station {selected_mw} Profile</div>
            <div style="color: #475569; display: grid; grid-template-columns: 1fr 1fr; gap: 6px;">
                <div><b>Latitude:</b> {lat:.3f}° N</div>
                <div><b>Longitude:</b> {lon:.3f}° E</div>
                <div><b>Elevation:</b> {elev:.1f} m</div>
                <div><b>Well Depth:</b> {depth:.1f} m</div>
            </div>
            <div style="margin-top: 8px; padding-top: 8px; border-top: 1px dashed #E2E8F0; color: #0284C7; font-weight: 600;">
                Latest Water Table: {last_recorded_gwl:.2f} m
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Mini map showing well position in Germany
        st.markdown("###### 🗺️ Station Location")
        map_df = pd.DataFrame([{"lat": lat, "lon": lon}])
        st.map(map_df, zoom=7, height=180)
    else:
        st.error("Historical dataset not loaded.")

# ---------------- TABS ----------------
tab_realtime, tab_scenario, tab_analytics = st.tabs([
    "🌐 Real-Time Telemetry & 7-Day Forecast",
    "🔮 Historical & Scenario Simulation",
    "🔬 Hydrological Physics & Model Verification"
])

# ================= TAB 1: REAL-TIME INGESTION & 7-DAY FORECAST =================
with tab_realtime:
    # Action bar
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.markdown(f"#### 📡 Live Climate Ingestion: Station `{selected_mw}`")
        st.caption(f"Real-time observation query at coordinates: **{lat}° N, {lon}° E** (Source: Open-Meteo REST API)")
    with header_col2:
        fetch_now = st.button("🔄 Refresh Live Telemetry", use_container_width=True)

    # Fetch live weather data
    weather_data = fetch_live_weather_and_forecast(lat, lon)
    
    if weather_data is not None:
        daily_weather = weather_data["daily"]
        today_rain = daily_weather["precipitation_sum"][0]
        today_temp_max = daily_weather["temperature_2m_max"][0]
        today_temp_mean = daily_weather["temperature_2m_mean"][0]
        today_et = daily_weather["et0_fao_evapotranspiration"][0]
        
        # Top KPI Cards
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="kpi-card kpi-blue">
                <div class="kpi-title">Today's Rainfall</div>
                <div class="kpi-value">{today_rain:.1f} <span style="font-size: 16px; font-weight: 500;">mm</span></div>
                <div class="kpi-sub">Direct precipitation volume</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="kpi-card kpi-amber">
                <div class="kpi-title">Peak Temperature</div>
                <div class="kpi-value">{today_temp_max:.1f} <span style="font-size: 16px; font-weight: 500;">°C</span></div>
                <div class="kpi-sub">Mean: {today_temp_mean:.1f} °C</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="kpi-card kpi-purple">
                <div class="kpi-title">Evapotranspiration (ET₀)</div>
                <div class="kpi-value">{today_et:.1f} <span style="font-size: 16px; font-weight: 500;">mm/day</span></div>
                <div class="kpi-sub">Atmospheric vapor demand</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="kpi-card kpi-green">
                <div class="kpi-title">Station Water Table</div>
                <div class="kpi-value">{last_recorded_gwl:.2f} <span style="font-size: 16px; font-weight: 500;">m</span></div>
                <div class="kpi-sub">Depth below ground level</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # Run model simulation across the 7-day forecast
        df_forecast = run_realtime_forecast(model, scaler, top10, weather_data, baseline_gwl=last_recorded_gwl)
        
        # Chart + Table Layout
        col_chart_rt, col_table_rt = st.columns([3, 2])
        
        with col_chart_rt:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("##### 📈 7-Day Water Table Trajectory & Rainfall Correlation")
            
            # Interactive Dual-Axis Plotly Chart
            fig = go.Figure()
            
            # Precipitation Bars (Secondary Y Axis)
            fig.add_trace(go.Bar(
                x=df_forecast["Date"],
                y=df_forecast["Rain (mm)"],
                name="Rainfall (mm)",
                yaxis="y2",
                marker_color="rgba(14, 165, 233, 0.4)",
                hovertemplate="<b>Date:</b> %{x}<br><b>Rainfall:</b> %{y:.1f} mm<extra></extra>"
            ))
            
            # Water Table Spline Curve (Primary Y Axis)
            fig.add_trace(go.Scatter(
                x=df_forecast["Date"],
                y=df_forecast["Projected Depth (m)"],
                mode="lines+markers",
                name="Groundwater Depth (m)",
                line=dict(color="#0284C7", width=3, shape="spline"),
                marker=dict(size=8, color="#0369A1", symbol="circle"),
                fill="tozeroy",
                fillcolor="rgba(2, 132, 199, 0.08)",
                hovertemplate="<b>Date:</b> %{x}<br><b>Water Table:</b> %{y:.2f} m<extra></extra>"
            ))
            
            fig.update_layout(
                margin=dict(l=10, r=10, t=20, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(showgrid=False, title=None),
                yaxis=dict(
                    title="Depth below ground (m)",
                    autorange="reversed",  # Standard hydrology convention: shallower = higher up
                    gridcolor="#F1F5F9",
                    zeroline=False
                ),
                yaxis2=dict(
                    title="Rainfall (mm)",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                    range=[0, max(df_forecast["Rain (mm)"].max() * 2.5, 10)]
                ),
                plot_bgcolor="white",
                paper_bgcolor="white",
                height=340
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Outlook badge
            net_change = df_forecast["Projected Depth (m)"].iloc[-1] - last_recorded_gwl
            net_cm = abs(net_change) * 100
            trend_str = "Aquifer Recharge (Rising)" if net_change < 0 else "Drawdown / Depletion (Falling)"
            pill_color = "#10B981" if net_change < 0 else "#F59E0B"
            
            st.markdown(f"""
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 14px; display: flex; align-items: center; justify-content: space-between;">
                <span style="font-size: 13px; color: #475569;"><b>7-Day Outlook:</b> Expected net shift of <b>{net_cm:.1f} cm</b> ({"+" if net_change > 0 else "-"}{abs(net_change):.2f}m)</span>
                <span style="background: {pill_color}20; color: {pill_color}; font-weight: 700; font-size: 12px; padding: 4px 10px; border-radius: 20px;">{trend_str}</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_table_rt:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("##### 📋 Daily Simulation Breakdown")
            
            # Format dataframe for display
            styled_df = df_forecast[["Date", "Rain (mm)", "Max Temp (°C)", "Status", "Projected Depth (m)"]].copy()
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True,
                height=380
            )
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.error("Could not fetch real-time telemetry. Please check internet connection.")

# ================= TAB 2: HISTORICAL & SCENARIO SIMULATION =================
with tab_scenario:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown(f"#### 🔮 Historical Benchmark & What-If Scenario: Station `{selected_mw}`")
    st.caption("Select any recorded historical day for this station to inspect the baseline, or adjust climate variables to test aquifer response.")
    
    dates = df_well["Date"].dt.strftime("%Y-%m-%d").tolist() if df is not None else []
    selected_date_str = st.selectbox("Select Historical Date", dates, index=len(dates)-1 if dates else 0)
    
    row = df_well[df_well["Date"] == pd.to_datetime(selected_date_str)].iloc[0] if dates else {}
    base_gwl = float(row.get("gwl_lag1", last_recorded_gwl))
    
    col_inputs, col_preview = st.columns([3, 2])
    
    with col_inputs:
        st.markdown("##### 🌦️ Environmental Variables")
        col_a, col_b = st.columns(2)
        scenario_inputs = {}
        for i, feat in enumerate(top10):
            meta = FEATURE_META.get(feat, {"label": feat, "unit": "", "desc": ""})
            lbl = f"{meta['label']} ({meta['unit']})" if meta['unit'] else meta['label']
            val = float(row.get(feat, 0.0))
            with (col_a if i % 2 == 0 else col_b):
                scenario_inputs[feat] = st.number_input(
                    lbl,
                    value=round(val, 2),
                    step=0.1,
                    key=f"scen_{feat}",
                    help=meta.get("desc", "")
                )
                
    with col_preview:
        st.markdown("##### 🎯 Model Response")
        
        # Run inference
        vec_df = pd.DataFrame([[scenario_inputs[f] for f in top10]], columns=top10)
        scaled_s = scaler.transform(vec_df)
        pred_delta_s = float(model.predict(scaled_s)[0])
        pred_gwl_s = base_gwl + pred_delta_s
        
        is_rise = pred_delta_s < 0
        delta_cm = abs(pred_delta_s) * 100
        pill_text = "Aquifer Recharge (Rising)" if is_rise else "Drawdown (Falling)"
        card_border = "#10B981" if is_rise else "#F59E0B"
        
        st.markdown(f"""
        <div style="background: white; border: 2px solid {card_border}; border-radius: 12px; padding: 20px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 13px; font-weight: 600; color: #64748B;">Predicted Daily Fluctuation (ΔGWL)</span>
                <span style="background: {card_border}20; color: {card_border}; font-size: 11px; font-weight: 700; padding: 3px 10px; border-radius: 20px;">{pill_text}</span>
            </div>
            <div style="font-size: 32px; font-weight: 800; color: #0F172A;">
                {"+" if not is_rise else "-"}{abs(pred_delta_s):.3f} m
                <span style="font-size: 16px; color: #64748B; font-weight: 500;">({delta_cm:.1f} cm/day)</span>
            </div>
        </div>
        
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 20px;">
            <div style="font-size: 13px; font-weight: 600; color: #64748B; margin-bottom: 4px;">Resulting Water Table Depth</div>
            <div style="font-size: 36px; font-weight: 800; color: #0284C7;">{pred_gwl_s:.2f} m</div>
            <div style="font-size: 12px; color: #64748B; margin-top: 6px; padding-top: 6px; border-top: 1px dashed #CBD5E1;">
                Baseline (t-1): <b>{base_gwl:.2f} m</b> | Historical Measured: <b>{float(row.get('GWL', base_gwl)):.2f} m</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True)

# ================= TAB 3: INSIGHTS & ANALYTICS =================
with tab_analytics:
    col_long_chart, col_feature_chart = st.columns([3, 2])
    
    with col_long_chart:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown(f"##### 📈 Multi-Year Hydrological Verification: Station `{selected_mw}`")
        
        if df is not None and selected_mw is not None:
            df_w = df[df["MW_ID"] == selected_mw].sort_values("Date").dropna(subset=top10 + ["gwl_lag1", "GWL"]).reset_index(drop=True)
            
            if len(df_w) > 0 and model is not None and scaler is not None:
                X_w = scaler.transform(df_w[top10])
                pred_deltas = model.predict(X_w)
                pred_gwls = df_w["gwl_lag1"].values + pred_deltas
                
                # Plotly time series
                fig_ts = go.Figure()
                fig_ts.add_trace(go.Scatter(
                    x=df_w["Date"],
                    y=df_w["GWL"],
                    mode="lines",
                    name="Actual Measured GWL",
                    line=dict(color="#0284C7", width=2)
                ))
                fig_ts.add_trace(go.Scatter(
                    x=df_w["Date"],
                    y=pred_gwls,
                    mode="lines",
                    name="Physics-Predicted GWL",
                    line=dict(color="#10B981", width=2, dash="dot")
                ))
                
                fig_ts.update_layout(
                    margin=dict(l=10, r=10, t=10, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(title="Groundwater Depth (m)", autorange="reversed", gridcolor="#F1F5F9"),
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    height=340
                )
                
                st.plotly_chart(fig_ts, use_container_width=True)
            else:
                st.info("Insufficient records for long-term plot.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_feature_chart:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("##### 🔬 Calibrated Feature Importances")
        st.caption("Distribution of model attention across physical weather drivers (no shortcut memorization).")
        
        if os.path.exists(TOP10_PATH):
            feat_df = pd.read_csv(TOP10_PATH)
            feat_df["feature_clean"] = feat_df["feature"].apply(lambda x: FEATURE_META.get(x, {}).get("label", x))
            feat_df["importance_pct"] = feat_df["importance"] * 100
            feat_df = feat_df.sort_values("importance_pct", ascending=True)
            
            fig_bar = px.bar(
                feat_df,
                x="importance_pct",
                y="feature_clean",
                orientation="h",
                color="importance_pct",
                color_continuous_scale="Blues",
                labels={"importance_pct": "Attention (%)", "feature_clean": "Hydrological Driver"}
            )
            fig_bar.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                coloraxis_showscale=False,
                plot_bgcolor="white",
                paper_bgcolor="white",
                height=340
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Export capabilities
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("##### 📥 Export Station Predictions")
    if df is not None and selected_mw is not None and len(df_w) > 0:
        export_df = pd.DataFrame({
            "Date": df_w["Date"],
            "Well_ID": df_w["MW_ID"],
            "Actual_GWL": df_w["GWL"],
            "Predicted_GWL": pred_gwls,
            "Predicted_Delta": pred_deltas,
            "Absolute_Error_m": np.abs(df_w["GWL"].values - pred_gwls)
        })
        
        st.download_button(
            label=f"📥 Download Time-Series Dataset for Station {selected_mw} (CSV)",
            data=export_df.to_csv(index=False),
            file_name=f"hydro_predictions_{selected_mw}.csv",
            mime="text/csv",
            use_container_width=True
        )
    st.markdown("</div>", unsafe_allow_html=True)
