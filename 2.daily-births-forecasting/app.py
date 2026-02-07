import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import warnings

warnings.filterwarnings("ignore")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Daily Births Forecasting",
    page_icon="👶",
    layout="centered"
)

# ---------------- HEADER ----------------
st.title("👶 Daily Births Forecasting Dashboard")
st.markdown("""
This application forecasts **daily female births** using **Facebook Prophet**.
Upload data or use the sample dataset and generate **real-time forecasts**.
""")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙ Forecast Settings")

forecast_days = st.sidebar.slider(
    "Select number of future days to predict",
    min_value=7,
    max_value=365,
    value=50
)

seasonality_mode = st.sidebar.selectbox(
    "Seasonality Mode",
    ["additive", "multiplicative"]
)

changepoint_scale = st.sidebar.slider(
    "Changepoint Prior Scale",
    0.01, 1.0, 0.5
)

# ---------------- DATA UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📂 Upload Daily Births CSV File",
    type=["csv"]
)

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"])
    return df

# ---------------- LOAD DATA ----------------
if uploaded_file:
    df = load_data(uploaded_file)
    st.success("Custom dataset loaded successfully!")
else:
    df = pd.read_csv("data/daily_births.csv")
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"])
    st.info("Using sample dataset")

# ---------------- DATA PREVIEW ----------------
st.subheader("📊 Data Preview")
st.dataframe(df.head())

# ---------------- HISTORICAL PLOT ----------------
st.subheader("📈 Historical Birth Data")

fig1, ax1 = plt.subplots()
ax1.plot(df["ds"], df["y"])
ax1.set_xlabel("Date")
ax1.set_ylabel("Number of Births")
ax1.set_title("Daily Female Births")
st.pyplot(fig1)

# ---------------- MODEL TRAINING ----------------
st.subheader("🤖 Model Training")

with st.spinner("Training Prophet model..."):
    model = Prophet(
        yearly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_scale,
        seasonality_mode=seasonality_mode
    )
    model.fit(df)

st.success("Model trained successfully!")

# ---------------- FORECASTING ----------------
future = model.make_future_dataframe(periods=forecast_days, freq="D")
forecast = model.predict(future)

# ---------------- FORECAST PLOT ----------------
st.subheader("🔮 Forecast Visualization")

fig2 = model.plot(forecast)
st.pyplot(fig2)

# ---------------- COMPONENTS ----------------
st.subheader("📉 Trend & Seasonality Components")
fig3 = model.plot_components(forecast)
st.pyplot(fig3)

# ---------------- REAL-TIME TABLE ----------------
st.subheader("📋 Forecast Data (Last 10 Days)")
st.dataframe(
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(10)
)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "📌 **Tech Stack:** Python, Streamlit, Prophet | "
    "📈 **Use Case:** Time Series Forecasting"
)
