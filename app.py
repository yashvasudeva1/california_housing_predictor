import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('linear_model.pkl')

st.title("🏠 California House Value Predictor")

st.markdown("Enter the values below:")

MedInc = st.number_input("Median Income", value=3.0)
HouseAge = st.number_input("House Age", value=25.0)
AveRooms = st.number_input("Average Rooms", value=5.0)
AveBedrms = st.number_input("Average Bedrooms", value=1.0)
Latitude = st.number_input("Latitude", value=34.0)
Longitude = st.number_input("Longitude", value=-118.0)

if st.button("Predict"):
    features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Latitude, Longitude]])
    prediction = model.predict(features)[0][0]
    st.success(f"Estimated Median House Value: ${prediction*100000:.2f}")
