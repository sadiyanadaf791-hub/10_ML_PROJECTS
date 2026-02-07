import streamlit as st
import pandas as pd
from src.data_preprocessing import load_and_clean_data

st.set_page_config(page_title="EarthQuake Prediction", layout="centered")

st.title("🌍 AI-Based Weather & Earthquake Prediction")

uploaded_file = st.file_uploader("Upload CSV File", type="csv")

if uploaded_file:
    data = load_and_clean_data(uploaded_file)
    st.success("Data processed successfully!")

    st.write("📊 Sample Data")
    st.dataframe(data.head())
