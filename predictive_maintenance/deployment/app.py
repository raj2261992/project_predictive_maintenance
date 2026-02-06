import streamlit as st
import numpy as np
import joblib

# ----------------------------------------------------
# Load trained model + scaler
# ----------------------------------------------------
MODEL_PATH = "model.joblib"        # change if different
SCALER_PATH = "scaler.joblib"      # optional

model = joblib.load(MODEL_PATH)

try:
    scaler = joblib.load(SCALER_PATH)
    use_scaler = True
except:
    use_scaler = False


# ----------------------------------------------------
# Page Config
# ----------------------------------------------------
st.set_page_config(
    page_title="Engine Health Prediction",
    layout="centered"
)

st.title("🛠️ Predictive Maintenance – Engine Health")
st.markdown("Enter live sensor values to predict engine condition.")

st.divider()

# ----------------------------------------------------
# Sidebar Inputs
# ----------------------------------------------------
st.sidebar.header("Sensor Inputs")

engine_rpm = st.sidebar.number_input("Engine RPM", min_value=0.0, value=1500.0)

lub_oil_pressure = st.sidebar.number_input(
    "Lub Oil Pressure (bar/kPa)", min_value=0.0, value=3.5
)

fuel_pressure = st.sidebar.number_input(
    "Fuel Pressure (bar/kPa)", min_value=0.0, value=4.0
)

coolant_pressure = st.sidebar.number_input(
    "Coolant Pressure (bar/kPa)", min_value=0.0, value=2.0
)

lub_oil_temp = st.sidebar.number_input(
    "Lub Oil Temperature (°C)", min_value=0.0, value=80.0
)

coolant_temp = st.sidebar.number_input(
    "Coolant Temperature (°C)", min_value=0.0, value=75.0
)

# ----------------------------------------------------
# Prediction
# ----------------------------------------------------
if st.button("Predict Engine Condition"):

    input_data = np.array([[
        engine_rpm,
        lub_oil_pressure,
        fuel_pressure,
        coolant_pressure,
        lub_oil_temp,
        coolant_temp

    ]])

    if use_scaler:
        input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 0:
        st.success("Engine Condition: NORMAL")
    else:
        st.error("Engine Condition: FAULTY / AT RISK")

    st.divider()

    st.write("### Input Summary")
    st.table({
        "Engine RPM": [engine_rpm],
        "Lub Oil Pressure": [lub_oil_pressure],
        "Fuel Pressure": [fuel_pressure],
        "Coolant Pressure": [coolant_pressure],
        "Lub Oil Temp": [lub_oil_temp],
        "Coolant Temp": [coolant_temp],
    })

# ----------------------------------------------------
# Footer
# ----------------------------------------------------
st.markdown("---")
st.caption("Predictive Maintenance Dashboard | Built with Streamlit")
