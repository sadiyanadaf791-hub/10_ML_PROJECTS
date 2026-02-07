# =========================================
# Global Temperature Prediction - Streamlit
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -----------------------------------------
# Page Config
# -----------------------------------------
st.set_page_config(
    page_title="🌍 Global Temperature Prediction",
    layout="centered"
)

st.title("🌍 Global Temperature Prediction")
st.write("Machine Learning based Weather Prediction using Random Forest")

# -----------------------------------------
# Load Dataset
# -----------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("../data/GlobalTemperatures.csv")

try:
    data = load_data()
except Exception:
    st.error("❌ CSV file not found or empty. Please check GlobalTemperatures.csv")
    st.stop()

# -----------------------------------------
# Data Preprocessing
# -----------------------------------------
def convert_to_fahrenheit(temp):
    return (temp * 1.8) + 32


def wrangle(df):
    df = df.copy()

    df.drop(columns=[
        "LandAverageTemperatureUncertainty",
        "LandMaxTemperatureUncertainty",
        "LandMinTemperatureUncertainty",
        "LandAndOceanAverageTemperatureUncertainty"
    ], inplace=True)

    temp_cols = [
        "LandAverageTemperature",
        "LandMaxTemperature",
        "LandMinTemperature",
        "LandAndOceanAverageTemperature"
    ]

    for col in temp_cols:
        df[col] = df[col].apply(convert_to_fahrenheit)

    df["dt"] = pd.to_datetime(df["dt"])
    df["Year"] = df["dt"].dt.year
    df = df[df["Year"] >= 1850]

    df.drop(columns=["dt"], inplace=True)
    df.dropna(inplace=True)

    return df


data = wrangle(data)

# -----------------------------------------
# Sidebar
# -----------------------------------------
st.sidebar.header("⚙️ Options")

show_data = st.sidebar.checkbox("Show Dataset")
show_corr = st.sidebar.checkbox("Show Correlation Heatmap")

# -----------------------------------------
# Dataset Preview
# -----------------------------------------
if show_data:
    st.subheader("📄 Dataset Preview")
    st.dataframe(data.head(20))

# -----------------------------------------
# Correlation Heatmap
# -----------------------------------------
if show_corr:
    st.subheader("📊 Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# -----------------------------------------
# Model Training
# -----------------------------------------
TARGET = "LandAndOceanAverageTemperature"

X = data[
    ["LandAverageTemperature", "LandMaxTemperature", "LandMinTemperature"]
]
y = data[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = Pipeline([
    ("features", SelectKBest(k="all")),
    ("scaler", StandardScaler()),
    ("rf", RandomForestRegressor(
        n_estimators=100,
        max_depth=50,
        random_state=77,
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)

# -----------------------------------------
# Model Evaluation (FIXED)
# -----------------------------------------
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

st.success("✅ Model Trained Successfully")
st.metric("RMSE", f"{rmse:.2f} °F")
st.metric("MAE", f"{mae:.2f} °F")

# -----------------------------------------
# Prediction Section
# -----------------------------------------
st.subheader("🌡️ Predict Temperature")

land_avg = st.number_input("Land Average Temperature (°F)", value=50.0)
land_max = st.number_input("Land Max Temperature (°F)", value=60.0)
land_min = st.number_input("Land Min Temperature (°F)", value=40.0)

if st.button("Predict"):
    input_data = np.array([[land_avg, land_max, land_min]])
    prediction = model.predict(input_data)
    st.success(
        f"🌍 Predicted Land & Ocean Average Temperature: **{prediction[0]:.2f} °F**"
    )

st.markdown("---")
st.caption("📌 Built with Streamlit & Random Forest | ML Project")
