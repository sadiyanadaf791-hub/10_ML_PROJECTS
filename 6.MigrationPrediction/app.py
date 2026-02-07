import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.set_page_config(
    page_title="Migration Prediction NZ",
    page_icon="🌍",
    layout="centered"
)

st.title("🌍 Migration Prediction to New Zealand")
st.markdown("Predict migration values using Machine Learning")

@st.cache_data
def load_data():
    data = pd.read_csv("data/migration_nz.csv")

    data['Measure'] = data['Measure'].map({
        'Arrivals': 0,
        'Departures': 1,
        'Net': 2
    })

    data['CountryID'] = pd.factorize(data['Country'])[0]
    data['CitID'] = pd.factorize(data['Citizenship'])[0]
    data['Value'].fillna(data['Value'].median(), inplace=True)

    return data

data = load_data()

X = data[['CountryID', 'Measure', 'Year', 'CitID']]
y = data['Value']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

st.success("Model trained successfully!")

# 🎛 User Input
st.sidebar.header("Enter Prediction Inputs")

country_id = st.sidebar.slider("Country ID", 0, int(data['CountryID'].max()))
cit_id = st.sidebar.slider("Citizenship ID", 0, int(data['CitID'].max()))
measure = st.sidebar.selectbox(
    "Migration Type",
    options=[("Arrivals", 0), ("Departures", 1), ("Net", 2)]
)
year = st.sidebar.slider("Year", int(data['Year'].min()), int(data['Year'].max()))

input_data = np.array([[country_id, measure[1], year, cit_id]])

if st.sidebar.button("Predict"):
    prediction = model.predict(input_data)
    st.metric("Predicted Migration Value", int(prediction[0]))

# 📈 Visualization
st.subheader("📈 Migration Trend by Year")
yearly_data = data.groupby("Year")['Value'].sum()
st.line_chart(yearly_data)

st.caption("Made with ❤️ by Sadiya Nadaf")
