import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("mlr_predictor.joblib")

# State encoding
state_mapping = {"California": 0, "Florida": 1, "New York": 2}

st.title("🏢 Startup Profit Predictor")
st.markdown("Predict startup profits based on spending and location using Multiple Linear Regression.")

# Input fields
rd_spend = st.number_input("R&D Spend", min_value=0.0, step=1000.0)
admin_spend = st.number_input("Administration Spend", min_value=0.0, step=1000.0)
marketing_spend = st.number_input("Marketing Spend", min_value=0.0, step=1000.0)
state = st.selectbox("Select State", list(state_mapping.keys()))

if st.button("Predict Profit 💰"):
    try:
        state_encoded = state_mapping[state]
        input_data = np.array([[rd_spend, admin_spend, marketing_spend, state_encoded]])
        predicted_profit = model.predict(input_data)[0]
        st.success(f"**Predicted Profit:** ₹{predicted_profit:,.2f}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Optional: show a comparison chart
import matplotlib.pyplot as plt
features = ["R&D", "Admin", "Marketing"]
values = [rd_spend, admin_spend, marketing_spend]

fig, ax = plt.subplots()
ax.bar(features, values, color=["#4CAF50", "#FF9800", "#2196F3"])
ax.set_ylabel("Amount (₹)")
st.pyplot(fig)



