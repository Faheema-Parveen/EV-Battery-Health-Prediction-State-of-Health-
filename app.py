import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("battery_soh_model.pkl")

st.set_page_config(page_title="EV Battery Health Prediction", page_icon="🔋")

st.title("🔋 EV Battery Health Prediction System")
st.write("Predict battery State of Health (SOH) using Machine Learning")

st.markdown("---")

# ==============================
# Manual Prediction Section
# ==============================

st.subheader("⚙ Manual Battery Prediction")

col1, col2 = st.columns(2)

with col1:
    voltage = st.number_input("⚡ Voltage (V)", min_value=0.0)
    current = st.number_input("🔌 Current (A)", min_value=0.0)
    temperature = st.number_input("🌡 Temperature (°C)", min_value=0.0)

with col2:
    capacity = st.number_input("🔋 Capacity (Ah)", min_value=0.0)
    cycle = st.number_input("🔄 Charge Cycle", min_value=0)

if st.button("🔍 Predict Battery SOH"):

    input_data = np.array([[voltage, current, temperature, capacity, cycle]])
    prediction = model.predict(input_data)[0]

    st.success(f"🔋 Predicted Battery SOH: {prediction:.2f}%")

    # Battery condition
    if prediction > 80:
        st.success("Battery Condition: GOOD ✅")
    elif prediction > 60:
        st.warning("Battery Condition: MODERATE ⚠")
    else:
        st.error("Battery Condition: POOR ❌")

    # Manual prediction graph
    st.subheader("📊 Manual Prediction Visualization")

    fig, ax = plt.subplots()

    ax.bar(["Predicted SOH"], [prediction], color="green")

    ax.set_ylabel("SOH (%)")
    ax.set_ylim(0, 100)

    st.pyplot(fig)

st.markdown("---")

# ==============================
# Automatic Dataset Testing
# ==============================

st.subheader("📂 Automatic Dataset Testing")

if st.button("Test Entire Dataset"):

    df = pd.read_csv("battery_dataset.csv")

    # Keep only required columns
    df = df[['Voltage','Current','Temperature','Capacity','Cycle','SOH']]

    X = df[['Voltage','Current','Temperature','Capacity','Cycle']]

    predictions = model.predict(X)

    df['Predicted_SOH'] = predictions

    st.write("Dataset with Predictions")
    st.dataframe(df)

    # Graph
    st.subheader("📊 Battery SOH Prediction Graph")

    fig, ax = plt.subplots()

    ax.plot(df['Predicted_SOH'], label="Predicted SOH", color="blue")

    ax.set_xlabel("Battery Samples")
    ax.set_ylabel("SOH (%)")

    ax.legend()

    st.pyplot(fig)

st.markdown("---")
