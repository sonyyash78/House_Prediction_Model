import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib

# ================= LOAD MODELS =================

rf_model = joblib.load("models/rf_model.pkl")
xgb_model = joblib.load("models/xgb_model.pkl")
columns = pickle.load(open("models/columns.pkl", "rb"))
meta = pickle.load(open("models/model_meta.pkl", "rb"))

# ================= UI =================
st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏠 House Price Prediction")
st.write("Predict house prices using ML Ensemble Model (RF + XGBoost)")

# ================= INPUT =================
sqft = st.number_input(
    "Total Sqft",
    int(meta["min_total_sqft"]),
    int(meta["max_total_sqft"]),
    1200
)

bhk = st.number_input("BHK", 1, 10, 2)
bath = st.number_input("Bathrooms", 1, 10, 2)
balcony = st.number_input("Balcony", 0, 5, 1)

location = st.selectbox("Location", meta["locations"])
area_type = st.selectbox("Area Type", meta["area_types"])

ready = st.checkbox("Ready To Move", value=True)

# ================= PREDICT =================
if st.button("Predict Price"):

    # create empty dataframe
    X = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

    # fill values
    X["total_sqft"] = sqft
    X["bath"] = bath
    X["balcony"] = balcony
    X["bhk"] = bhk
    X["is_ready"] = int(ready) 
    # encode location
    loc_col = "location_" + location
    if loc_col in X.columns:
        X[loc_col] = 1

    # encode area type
    area_col = "area_type_" + area_type
    if area_col in X.columns:
        X[area_col] = 1

    # ================= MODEL PREDICTION =================
    rf_pred = rf_model.predict(X)[0]
    xgb_pred = xgb_model.predict(X)[0]

    # ensemble
    final_pred = (0.7 * rf_pred) + (0.3 * xgb_pred)

    # convert (if log used)
    try:
        price = np(final_pred)
    except:
        price = final_pred

    st.success(f"💰 Estimated Price: ₹ {round(price, 2)} Lakhs")

    # ================= EXTRA INFO =================
    st.info("Model used: Weighted Ensemble (Random Forest + XGBoost)")