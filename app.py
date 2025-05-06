import streamlit as st
from model import load_and_train_model
from utils import load_model, preprocess_input
import pandas as pd

st.title("🚦 AI-Driven Traffic Accident Severity Prediction")

st.sidebar.header("Input Parameters")
temp = st.sidebar.slider("Temperature (F)", -10, 120, 70)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 50)
visibility = st.sidebar.slider("Visibility (mi)", 0, 10, 5)
wind_speed = st.sidebar.slider("Wind Speed (mph)", 0, 50, 10)
weather_cond = st.sidebar.selectbox("Weather Condition", ['Clear', 'Rain', 'Snow', 'Fog', 'Cloudy'])

if st.button("Train Model"):
    model = load_and_train_model()
    st.success("Model trained and saved!")

if st.button("Predict Severity"):
    model, encoder = load_model()
    input_data = preprocess_input(temp, humidity, visibility, wind_speed, weather_cond, encoder)
    prediction = model.predict(input_data)
    st.subheader(f"Predicted Severity: {prediction[0]}")

from model import load_and_train_model

# This will train the model and save model.joblib and encoder.joblib
load_and_train_model()

