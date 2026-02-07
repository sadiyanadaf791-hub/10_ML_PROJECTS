import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Rainfall Prediction",
    page_icon="🌧️",
    layout="centered"
)

# ---------------- HEADER ----------------
st.title("🌧️ Rainfall Prediction Dashboard")
st.markdown("""
Predict **daily rainfall** using Machine Learning models.
Upload your dataset or use the sample dataset to forecast rainfall.
""")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙ Settings")

model_option = st.sidebar.selectbox(
    "Select ML Model",
    ["Random Forest", "Linear Regression"]
)

forecast_days = st.sidebar.slider(
    "Prediction Days",
    min_value=1,
    max_value=30,
    value=7
)

# ---------------- DATA UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📂 Upload Rainfall CSV",
    type=["csv"]
)

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = ["Date", "Rainfall"]
    df["Date"] = pd.to_datetime(df["Date"])
    return df

# ---------------- LOAD DATA ----------------
if uploaded_file:
    df = load_data(uploaded_file)
    st.success("Custom dataset loaded!")
else:
    df = pd.read_csv("data/rainfall.csv")
    df.columns = ["Date", "Rainfall"]
    df["Date"] = pd.to_datetime(df["Date"])
    st.info("Using sample dataset")

# ---------------- DATA PREVIEW ----------------
st.subheader("📊 Data Preview")
st.dataframe(df.head())

# ---------------- HISTORICAL PLOT ----------------
st.subheader("📈 Historical Rainfall Data")

fig1, ax1 = plt.subplots()
ax1.plot(df["Date"], df["Rainfall"], marker='o')
ax1.set_xlabel("Date")
ax1.set_ylabel("Rainfall (mm)")
ax1.set_title("Historical Rainfall")
st.pyplot(fig1)

# ---------------- FEATURE ENGINEERING ----------------
df["DayOfYear"] = df["Date"].dt.dayofyear

X = df[["DayOfYear"]]
y = df["Rainfall"]

# ---------------- SPLIT DATA ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL TRAINING ----------------
st.subheader("🤖 Model Training")

if model_option == "Random Forest":
    model = RandomForestRegressor(n_estimators=100, random_state=42)
else:
    model = LinearRegression()

with st.spinner("Training model..."):
    model.fit(X_train, y_train)

st.success(f"{model_option} trained successfully!")

# ---------------- FORECAST ----------------
last_day = df["DayOfYear"].max()
future_days = np.array([last_day + i for i in range(1, forecast_days+1)]).reshape(-1,1)
predictions = model.predict(future_days)

future_dates = pd.date_range(df["Date"].max() + pd.Timedelta(days=1), periods=forecast_days)
forecast_df = pd.DataFrame({"Date": future_dates, "Predicted Rainfall": predictions})

# ---------------- FORECAST PLOT ----------------
st.subheader("🔮 Rainfall Forecast")

fig2, ax2 = plt.subplots()
ax2.plot(df["Date"], df["Rainfall"], label="Historical")
ax2.plot(forecast_df["Date"], forecast_df["Predicted Rainfall"], marker='o', linestyle='--', color="red", label="Predicted")
ax2.set_xlabel("Date")
ax2.set_ylabel("Rainfall (mm)")
ax2.legend()
ax2.set_title("Rainfall Prediction")
st.pyplot(fig2)

# ---------------- FORECAST TABLE ----------------
st.subheader("📋 Forecasted Rainfall")
st.dataframe(forecast_df)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("📌 **Tech Stack:** Python, scikit-learn, Streamlit | 🌧️ Rainfall Prediction App")
