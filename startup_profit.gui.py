import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Load trained model ----------------
model = joblib.load("mlr_predictor.joblib")

# ---------------- State encoding ----------------
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

# ---------------- Page setup ----------------
st.set_page_config(page_title="Startup Profit Predictor", layout="centered")
st.title("🚀 Startup Profit Predictor")
st.markdown(
    "Predict your startup’s profit based on **expenses and location** — now with _What-If_ and _Scenario Comparison_ features plus interactive graphs!"
)

# ---------------- Input Section ----------------
st.subheader("💼 Business Details")

col1, col2 = st.columns(2)
with col1:
    rd = st.number_input("R&D Spend (₹)", min_value=0.0, value=100000.0, step=1000.0)
    admin = st.number_input("Administration (₹)", min_value=0.0, value=120000.0, step=1000.0)
with col2:
    marketing = st.number_input("Marketing Spend (₹)", min_value=0.0, value=150000.0, step=1000.0)
    state = st.selectbox("State", list(state_mapping.keys()))

chart_type = st.selectbox("Select Graph Type", ["Bar", "Line", "Scatter"])

# ---------------- Base Prediction ----------------
state_encoded = state_mapping[state]
base_input = np.array([[rd, admin, marketing, state_encoded]])
base_profit = model.predict(base_input)[0]
st.success(f"💰 Predicted Profit: ₹{base_profit:,.2f}")

# ---------------- What-If Sliders ----------------
st.subheader("🎯 What-If Analysis")
st.markdown("Adjust sliders to test growth, expense, and funding impact.")

revenue_change = st.slider("Revenue Growth (%)", -20, 50, 10)
expense_change = st.slider("Expense Change (%)", -10, 30, 0)
funding_boost = st.slider("Additional Funding (₹)", 0, 100000, 20000)

# Adjusted values
rd_adj = rd * (1 + revenue_change / 100)
admin_adj = admin * (1 + expense_change / 100)
marketing_adj = marketing + funding_boost

# Predict new profit
input_data = np.array([[rd_adj, admin_adj, marketing_adj, state_encoded]])
predicted_profit = model.predict(input_data)[0]
st.info(f"📈 Adjusted Profit (What-If Result): ₹{predicted_profit:,.2f}")

# ---------------- Input-Impact Graph (Your Old Graphs) ----------------
st.subheader("📊 How Inputs Affect Predicted Profit")

features = ["R&D Spend", "Administration", "Marketing Spend", "Predicted Profit"]
values = [rd, admin, marketing, base_profit]

fig1, ax1 = plt.subplots(figsize=(7, 4))
ax1.set_title("Inputs vs Predicted Profit")
ax1.set_ylabel("Value (₹)")

if chart_type == "Bar":
    colors = ["#FFB347", "#FFD966", "#FFC300", "#FF8000"]
    ax1.bar(features, values, color=colors)
elif chart_type == "Line":
    ax1.plot(features, values, marker='o', color="#FF8000", linewidth=2)
elif chart_type == "Scatter":
    ax1.scatter(features, values, color="#E65100", s=100)

ax1.text(3, base_profit, f"₹{base_profit:,.2f}", ha='center', va='bottom', fontsize=10, color='red')
st.pyplot(fig1)

# ---------------- Scenario Comparison ----------------
st.subheader("📈 Compare Business Scenarios")

def predict_profit(rd, admin, marketing, state_encoded):
    return model.predict(np.array([[rd, admin, marketing, state_encoded]]))[0]

pess = predict_profit(rd * 0.9, admin * 1.1, marketing * 0.8, state_encoded)
real = predict_profit(rd, admin, marketing, state_encoded)
opt = predict_profit(rd * 1.2, admin * 0.9, marketing * 1.3, state_encoded)

colA, colB, colC = st.columns(3)
colA.metric("Pessimistic", f"₹{pess:,.2f}")
colB.metric("Realistic", f"₹{real:,.2f}")
colC.metric("Optimistic", f"₹{opt:,.2f}")

# Scenario bar chart
scenarios = ["Pessimistic", "Realistic", "Optimistic"]
profits = [pess, real, opt]

fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.bar(scenarios, profits, color=["#FFB347", "#FFD966", "#FF8000"])
ax2.set_ylabel("Predicted Profit (₹)")
ax2.set_title("Profit Comparison Across Scenarios")
for i, v in enumerate(profits):
    ax2.text(i, v + 5000, f"₹{v:,.0f}", ha='center', fontsize=10, color='black')

st.pyplot(fig2)

# ---------------- Footer ----------------
st.markdown("---")
st.caption("💡 Made with ❤️ by Kamya Kapoor | Enhanced with What-If Analysis, Scenario Comparison & Interactive Graphs")









