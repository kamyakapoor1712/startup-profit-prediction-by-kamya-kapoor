import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

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

# --- Page Title ---
st.title("🚀 Indian Startup Profit Predictor")
st.markdown("Predict your startup’s profit and test *What-If* business scenarios using sliders.")

# --- Base Input Fields ---
st.subheader("📊 Base Business Inputs")
col1, col2 = st.columns(2)
with col1:
    rd_spend = st.number_input("R&D Spend (₹)", min_value=0.0, step=1000.0, value=100000.0)
    admin_spend = st.number_input("Administration Spend (₹)", min_value=0.0, step=1000.0, value=120000.0)
with col2:
    marketing_spend = st.number_input("Marketing Spend (₹)", min_value=0.0, step=1000.0, value=150000.0)
    state = st.selectbox("State", list(state_mapping.keys()))

state_encoded = state_mapping[state]
base_input = np.array([[rd_spend, admin_spend, marketing_spend, state_encoded]])
base_profit = model.predict(base_input)[0]

st.success(f"💰 **Base Predicted Profit:** ₹{base_profit:,.2f}")

# --- What-If Analysis Section ---
st.subheader("🤔 What-If Analysis — Adjust Key Factors")

colA, colB, colC = st.columns(3)
with colA:
    rd_change = st.slider("R&D Change (%)", -50, 50, 0)
with colB:
    admin_change = st.slider("Admin Change (%)", -50, 50, 0)
with colC:
    marketing_change = st.slider("Marketing Change (%)", -50, 50, 0)

# Compute adjusted values
rd_new = rd_spend * (1 + rd_change / 100)
admin_new = admin_spend * (1 + admin_change / 100)
marketing_new = marketing_spend * (1 + marketing_change / 100)

adjusted_input = np.array([[rd_new, admin_new, marketing_new, state_encoded]])
adjusted_profit = model.predict(adjusted_input)[0]

st.info(f"📈 **Adjusted Profit:** ₹{adjusted_profit:,.2f}")

# --- Scenario Comparison (Pessimistic / Realistic / Optimistic) ---
st.subheader("📉 Compare Scenarios")

scenarios = {
    "Pessimistic": [rd_spend * 0.9, admin_spend * 0.9, marketing_spend * 0.9],
    "Realistic": [rd_spend, admin_spend, marketing_spend],
    "Optimistic": [rd_spend * 1.1, admin_spend * 1.1, marketing_spend * 1.1],
}

profits = {}
for scenario, vals in scenarios.items():
    x = np.array([[vals[0], vals[1], vals[2], state_encoded]])
    profits[scenario] = model.predict(x)[0]

# --- Chart Visualization ---
fig, ax = plt.subplots(figsize=(6,4))
ax.bar(profits.keys(), profits.values(), color=["#FF6F61", "#FFB74D", "#81C784"])
ax.set_title("Profit Comparison Across Scenarios")
ax.set_ylabel("Predicted Profit (₹)")
for i, val in enumerate(profits.values()):
    ax.text(i, val, f"₹{val:,.0f}", ha='center', va='bottom')

st.pyplot(fig)

# --- Footer ---
st.markdown("---")
st.caption("💡 Created by Kamya Kapoor | Enhanced with What-If Analysis and Scenario Comparison")





