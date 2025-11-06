import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load model
model = joblib.load("mlr_predictor.joblib")

# --- State label encoding ---
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

# --- Page setup ---
st.set_page_config(page_title="Startup Profit Predictor", layout="centered")
st.title("🚀 Startup Profit Predictor")
st.markdown("Predict your startup’s profit based on your **expenses and location**. Now includes _What-If_ and _Scenario Comparison_ features!")

# ---------- INPUT SECTION ----------
st.subheader("💼 Business Details")

col1, col2 = st.columns(2)
with col1:
    rd = st.number_input("R&D Spend (₹)", min_value=0.0, value=100000.0, step=1000.0)
    admin = st.number_input("Administration (₹)", min_value=0.0, value=120000.0, step=1000.0)
with col2:
    marketing = st.number_input("Marketing Spend (₹)", min_value=0.0, value=150000.0, step=1000.0)
    state = st.selectbox("State", list(state_mapping.keys()))

# ---------- "WHAT-IF" SLIDERS ----------
st.subheader("🎯 What-If Analysis")
st.markdown("Adjust the sliders to test different growth and cost change scenarios.")

revenue_change = st.slider("Revenue Growth (%)", -20, 50, 10)
expense_change = st.slider("Expense Change (%)", -10, 30, 0)
funding_boost = st.slider("Additional Funding (₹)", 0, 100000, 20000)

# Adjusted values
rd_adj = rd * (1 + revenue_change / 100)
admin_adj = admin * (1 + expense_change / 100)
marketing_adj = marketing + funding_boost

# Predict new profit
state_encoded = state_mapping[state]
input_data = np.array([[rd_adj, admin_adj, marketing_adj, state_encoded]])
predicted_profit = model.predict(input_data)[0]

st.success(f"💰 Predicted Profit (Adjusted): ₹{predicted_profit:,.2f}")

# ---------- SCENARIO COMPARISON ----------
st.subheader("📈 Compare Scenarios")

# Define three different scenarios
def predict_profit(rd, admin, marketing, state_encoded):
    return model.predict(np.array([[rd, admin, marketing, state_encoded]]))[0]

pess = predict_profit(rd * 0.9, admin * 1.1, marketing * 0.8, state_encoded)
real = predict_profit(rd, admin, marketing, state_encoded)
opt = predict_profit(rd * 1.2, admin * 0.9, marketing * 1.3, state_encoded)

col1, col2, col3 = st.columns(3)
col1.metric("Pessimistic", f"₹{pess:,.2f}")
col2.metric("Realistic", f"₹{real:,.2f}")
col3.metric("Optimistic", f"₹{opt:,.2f}")

# ---------- VISUALIZATION ----------
st.subheader("📊 Scenario Profit Comparison")

scenarios = ["Pessimistic", "Realistic", "Optimistic"]
profits = [pess, real, opt]

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(scenarios, profits, color=["#FFB347", "#FFD966", "#FF8000"])
ax.set_ylabel("Predicted Profit (₹)")
ax.set_title("Profit Comparison Across Scenarios")

for i, v in enumerate(profits):
    ax.text(i, v + 5000, f"₹{v:,.0f}", ha='center', fontsize=10, color='black')

st.pyplot(fig)

# ---------- FOOTER ----------
st.markdown("---")
st.caption("💡 Made with ❤️ by Kamya Kapoor using Streamlit and Machine Learning")







