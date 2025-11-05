import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("mlr_predictor.joblib")

# State encoding
state_mapping = {
    "Maharashtra": 0,
    "Karnataka": 1,
    "Delhi NCR": 2,
    "Gujarat": 3,
    "Tamil Nadu": 4,
    "Telangana": 5,
    "West Bengal": 6,
    "Uttar Pradesh": 7,
    "Kerala": 8,
    "Rajasthan": 9
}

st.title("🇮🇳 Indian Startup Profit Predictor")
st.markdown("Predict your startup's profit based on spending and Indian state location using Multiple Linear Regression.")

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




