import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from prophet import Prophet
import warnings

warnings.filterwarnings("ignore")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="centered"
)

# ---------------- HEADER ----------------
st.title("📈 Real-Time Stock Price Prediction")
st.markdown("""
This application predicts **future stock prices** using **Facebook Prophet**
with **live market data** fetched from Yahoo Finance.
""")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙ Prediction Settings")

ticker = st.sidebar.text_input(
    "Stock Ticker Symbol",
    value="AAPL"
)

forecast_days = st.sidebar.slider(
    "Forecast Duration (Days)",
    min_value=7,
    max_value=365,
    value=30
)

seasonality_mode = st.sidebar.selectbox(
    "Seasonality Mode",
    ["additive", "multiplicative"]
)

changepoint_scale = st.sidebar.slider(
    "Changepoint Prior Scale",
    0.01, 1.0, 0.5
)

# ---------------- LOAD STOCK DATA ----------------
@st.cache_data
def load_stock_data(symbol):
    stock = yf.download(symbol, period="5y")
    stock.reset_index(inplace=True)
    return stock

data = load_stock_data(ticker)

# ---------------- VALIDATION ----------------
if data.empty:
    st.error("❌ Invalid stock symbol. Please enter a valid ticker.")
    st.stop()

# ---------------- PREPARE DATA ----------------
df = data[["Date", "Close"]]
df.columns = ["ds", "y"]

# ---------------- DATA PREVIEW ----------------
st.subheader("📊 Recent Stock Data")
st.dataframe(df.tail())

# ---------------- HISTORICAL PLOT ----------------
st.subheader("📉 Historical Stock Price")

fig1, ax1 = plt.subplots()
ax1.plot(df["ds"], df["y"])
ax1.set_xlabel("Date")
ax1.set_ylabel("Closing Price")
ax1.set_title(f"{ticker} Stock Price History")
st.pyplot(fig1)

# ---------------- MODEL TRAINING ----------------
st.subheader("🤖 Training Prophet Model")

with st.spinner("Training model..."):
    model = Prophet(
        daily_seasonality=False,
        yearly_seasonality=True,
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_scale
    )
    model.fit(df)

st.success("Model trained successfully!")

# ---------------- FORECASTING ----------------
future = model.make_future_dataframe(periods=forecast_days)
forecast = model.predict(future)

# ---------------- FORECAST PLOT ----------------
st.subheader("🔮 Stock Price Forecast")

fig2 = model.plot(forecast)
st.pyplot(fig2)

# ---------------- COMPONENTS ----------------
st.subheader("📊 Trend & Seasonality Components")

fig3 = model.plot_components(forecast)
st.pyplot(fig3)

# ---------------- FORECAST TABLE ----------------
st.subheader("📋 Forecasted Prices (Last 10 Days)")

st.dataframe(
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(10)
)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "📌 **Built with:** Prophet, Streamlit, Yahoo Finance API | "
    "⚠ Educational purpose only"
)
