import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="ARIMA Anomaly Detection",
    page_icon="🚨",
    layout="centered"
)

# ---------------- HEADER ----------------
st.title("🚨 Anomaly Detection using ARIMA")
st.markdown("""
This application detects **anomalies in time-series data** using the
**ARIMA forecasting model** with real-time visualization.
""")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙ Model Settings")

p = st.sidebar.slider("AR (p)", 0, 5, 1)
d = st.sidebar.slider("Differencing (d)", 0, 2, 1)
q = st.sidebar.slider("MA (q)", 0, 5, 1)

threshold = st.sidebar.slider(
    "Anomaly Threshold (Std Dev)",
    min_value=1.0,
    max_value=5.0,
    value=2.5
)

# ---------------- DATA UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📂 Upload Time Series CSV",
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
    st.success("Custom dataset loaded!")
else:
    df = pd.read_csv("data/anomaly_data.csv")
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"])
    st.info("Using sample dataset")

# ---------------- DATA PREVIEW ----------------
st.subheader("📊 Data Preview")
st.dataframe(df.head())

# ---------------- HISTORICAL PLOT ----------------
st.subheader("📈 Time Series Data")

fig1, ax1 = plt.subplots()
ax1.plot(df["ds"], df["y"], label="Actual")
ax1.set_xlabel("Date")
ax1.set_ylabel("Value")
ax1.legend()
st.pyplot(fig1)

# ---------------- ARIMA MODEL ----------------
st.subheader("🤖 ARIMA Model Training")

with st.spinner("Fitting ARIMA model..."):
    model = ARIMA(df["y"], order=(p, d, q))
    model_fit = model.fit()

st.success("Model trained successfully!")

# ---------------- PREDICTION ----------------
pred = model_fit.predict(start=0, end=len(df)-1)

residuals = df["y"] - pred
std_dev = np.std(residuals)

# ---------------- ANOMALY DETECTION ----------------
df["anomaly"] = np.abs(residuals) > threshold * std_dev

# ---------------- ANOMALY PLOT ----------------
st.subheader("🚩 Detected Anomalies")

fig2, ax2 = plt.subplots()
ax2.plot(df["ds"], df["y"], label="Actual")
ax2.scatter(
    df[df["anomaly"]]["ds"],
    df[df["anomaly"]]["y"],
    color="red",
    label="Anomaly"
)
ax2.legend()
ax2.set_title("ARIMA-Based Anomaly Detection")
st.pyplot(fig2)

# ---------------- RESULT TABLE ----------------
st.subheader("📋 Anomaly Records")

st.dataframe(
    df[df["anomaly"]]
)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "📌 **Model:** ARIMA | **Use Case:** Anomaly Detection in Time-Series"
)
