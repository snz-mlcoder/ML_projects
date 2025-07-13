import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load("model/car_price_model.pkl")

# You can replace these with your actual data statistics if needed
average_values = {
    'Year': 2015,
    'HP': 180,
    'Cylinders': 4,
    'MPG-H': 30,
    'MPG-C': 24
}

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

st.title(" 🚗 Car Price Prediction App")
st.markdown("Enter car details below. If you're not sure about a value, check the '**I don't know**' box and we’ll use the average.")

st.markdown("---")

# Input: Year
year_unknown = st.checkbox("I don't know the year")
if year_unknown:
    year = average_values['Year']
    st.text(f"Using average Year: {year}")
else:
    year = st.slider("Year", 1990, 2025, value=2015, help="Newer cars usually cost more")

# Input: Horsepower (HP)
hp_unknown = st.checkbox("I don't know the horsepower (HP)")
if hp_unknown:
    hp = average_values['HP']
    st.text(f"Using average HP: {hp}")
else:
    hp = st.slider("Horsepower (HP)", 50, 700, value=150,step=25, help="Most cars range between 100–400 HP")

# Input: Cylinders
cyl_unknown = st.checkbox("I don't know the number of cylinders")
if cyl_unknown:
    cylinders = average_values['Cylinders']
    st.text(f"Using average Cylinders: {cylinders}")
else:
    cylinders = st.selectbox("Number of Cylinders", [2, 3, 4, 5, 6, 8, 10, 12], index=2)

# Input: MPG (Highway)
mpgh_unknown = st.checkbox("I don't know highway MPG")
if mpgh_unknown:
    mpg_h = average_values['MPG-H']
    st.text(f"Using average MPG-H: {mpg_h}")
else:
    mpg_h = st.number_input("MPG (Highway)", min_value=5, max_value=60, value=30, step=2)

# Input: MPG (City)
mpgc_unknown = st.checkbox("I don't know city MPG")
if mpgc_unknown:
    mpg_c = average_values['MPG-C']
    st.text(f"Using average MPG-C: {mpg_c}")
else:
    mpg_c = st.number_input("MPG (City)", min_value=5, max_value=60, value=24, step=2)

st.markdown("---")

# Predict button
if st.button("💰 Predict Price"):
    features = np.array([[year, hp, cylinders, mpg_h, mpg_c]])
    predicted_price = model.predict(features)[0]
    st.success(f"Estimated Price: ${predicted_price:,.2f}")
