import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.metrics import mean_squared_error
from utils.preprocessing import create_dataset, scale_data
from model.lstm_model import build_lstm_model

st.set_page_config(
    page_title="📈 LSTM Time Series Forecasting",
    layout="centered"
)
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fa;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 Time Series Forecasting with LSTM")
st.markdown("Predict future values using Deep Learning (LSTM)")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

look_back = st.slider("Look Back Window", 1, 12, 1)
epochs = st.slider("Training Epochs", 10, 200, 100)

if uploaded_file:
    df = pd.read_csv(uploaded_file, usecols=[1])
    dataset = df.values.astype("float32")

    dataset, scaler = scale_data(dataset)

    train_size = int(len(dataset) * 0.67)
    train, test = dataset[:train_size], dataset[train_size:]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    trainX = trainX.reshape(trainX.shape[0], 1, trainX.shape[1])
    testX = testX.reshape(testX.shape[0], 1, testX.shape[1])

    model = build_lstm_model((1, look_back))

    with st.spinner("Training LSTM Model..."):
        model.fit(trainX, trainY, epochs=epochs, batch_size=1, verbose=0)

    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    trainPredict = scaler.inverse_transform(trainPredict)
    testPredict = scaler.inverse_transform(testPredict)

    trainY = scaler.inverse_transform([trainY])
    testY = scaler.inverse_transform([testY])

    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))

    st.success(f"Train RMSE: {trainScore:.2f}")
    st.success(f"Test RMSE: {testScore:.2f}")

    # Plot
    st.subheader("📉 Prediction Visualization")

    trainPlot = np.empty_like(dataset)
    trainPlot[:, :] = np.nan
    trainPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

    testPlot = np.empty_like(dataset)
    testPlot[:, :] = np.nan
    testPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict

    fig = plt.figure(figsize=(10, 5))
    plt.plot(scaler.inverse_transform(dataset), label="Actual Data")
    plt.plot(trainPlot, label="Train Prediction")
    plt.plot(testPlot, label="Test Prediction")
    plt.legend()
    st.pyplot(fig)
