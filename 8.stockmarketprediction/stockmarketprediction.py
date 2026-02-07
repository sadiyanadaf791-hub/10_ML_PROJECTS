import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Page configuration
st.set_page_config(
    page_title="Stock Price Forecasting App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App header
st.markdown(
    """
    <div style='text-align:center;'>
        <h1>📈 Real-Time Stock Price Forecasting</h1>
        <p style='font-size:18px;'>Predict future stock prices using Machine Learning (Linear Regression)</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")
stock_symbol = st.sidebar.selectbox(
    "Select Stock",
    ["GOOG", "AAPL", "MSFT", "AMZN", "TSLA"]
)
forecast_days = st.sidebar.slider("Forecast Days", 1, 10, 5)

# Load data
@st.cache_data
def load_data(symbol):
    data = yf.download(symbol, period="6mo")
    data.reset_index(inplace=True)
    return data

df = load_data(stock_symbol)

# Display historical data
st.subheader(f"📊 Historical Data for {stock_symbol}")
st.dataframe(df.tail())

# Prepare data for ML
df["Close_Shifted"] = df["Close"].shift(-forecast_days)
df.dropna(inplace=True)

X = df[["Close"]].values
y = df["Close_Shifted"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

# Forecast
last_prices = scaler.transform(df[["Close"]].values[-forecast_days:])
forecast = model.predict(last_prices)

# Display model performance
st.subheader("🔍 Model Performance")
st.metric(label="Accuracy", value=f"{accuracy*100:.2f}%")

# Display forecasted prices
st.subheader("🔮 Forecasted Prices")
forecast_df = pd.DataFrame({"Forecasted Price": forecast})
st.dataframe(forecast_df)

# Plot actual vs forecast
st.subheader("📉 Stock Price Chart")
df_plot = df.copy()
df_plot["Forecast"] = np.nan
for i in range(forecast_days):
    df_plot.loc[len(df_plot) - forecast_days + i, "Forecast"] = forecast[i]

plt.figure(figsize=(12, 6))
plt.plot(df_plot["Date"], df_plot["Close"], label="Actual Price", color="#1f77b4", linewidth=2)
plt.plot(df_plot["Date"], df_plot["Forecast"], label="Forecasted Price", linestyle="--", color="#ff7f0e", linewidth=2)
plt.xlabel("Date")
plt.ylabel("Price")
plt.title(f"{stock_symbol} Stock Price vs Forecast")
plt.legend()
plt.grid(alpha=0.3)

st.pyplot(plt)

# Footer
st.markdown(
    "<div style='text-align:center; margin-top:30px;'>Made with ❤️ by Sadiya Nadaf</div>",
    unsafe_allow_html=True
)
