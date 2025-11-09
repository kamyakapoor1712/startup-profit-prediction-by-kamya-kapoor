import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---------------- Load trained ML model ----------------
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
st.set_page_config(page_title="Startup Profit Predictor + AI Assistant", layout="centered")
st.title("🚀 Indian Startup Profit Predictor + 🤖 Business Assistant")
st.markdown("Predict your startup’s profit and get smart AI-powered business advice!")

# ---------------- Input Section ----------------
st.subheader("📊 Enter Business Details")

col1, col2 = st.columns(2)
with col1:
    rd_spend = st.number_input("R&D Spend (₹)", min_value=0.0, value=100000.0, step=1000.0)
    admin_spend = st.number_input("Administration Spend (₹)", min_value=0.0, value=120000.0, step=1000.0)
with col2:
    marketing_spend = st.number_input("Marketing Spend (₹)", min_value=0.0, value=150000.0, step=1000.0)
    state = st.selectbox("Select State", list(state_mapping.keys()))

chart_type = st.selectbox("Select Graph Type", ["Bar", "Line", "Scatter"])

# ---------------- Base Prediction ----------------
state_encoded = state_mapping[state]
base_input = np.array([[rd_spend, admin_spend, marketing_spend, state_encoded]])
base_profit = model.predict(base_input)[0]
st.success(f"💰 Predicted Profit: ₹{base_profit:,.2f}")

# ---------------- Input Impact Graph ----------------
st.subheader("📈 How Inputs Affect Predicted Profit")
features = ["R&D Spend", "Administration", "Marketing Spend", "Predicted Profit"]
values = [rd_spend, admin_spend, marketing_spend, base_profit]

fig1, ax1 = plt.subplots(figsize=(7, 4))
ax1.set_title("Input Impact on Predicted Profit")
ax1.set_ylabel("Value (₹)")

if chart_type == "Bar":
    ax1.bar(features, values, color=["#FFB74D", "#4FC3F7", "#81C784", "#E57373"])
elif chart_type == "Line":
    ax1.plot(features, values, marker='o', color="#673AB7")
elif chart_type == "Scatter":
    ax1.scatter(features, values, color="#388E3C", s=100)

ax1.text(3, base_profit, f"₹{base_profit:,.2f}", ha='center', va='bottom', fontsize=10, color='red')
st.pyplot(fig1)

# ---------------- What-If Sliders ----------------
st.subheader("🤔 What-If Analysis (Adjust Key Factors)")
colA, colB, colC = st.columns(3)
with colA:
    rd_change = st.slider("R&D Change (%)", -50, 50, 0)
with colB:
    admin_change = st.slider("Admin Change (%)", -50, 50, 0)
with colC:
    marketing_change = st.slider("Marketing Change (%)", -50, 50, 0)

# Calculate new adjusted values
rd_new = rd_spend * (1 + rd_change / 100)
admin_new = admin_spend * (1 + admin_change / 100)
marketing_new = marketing_spend * (1 + marketing_change / 100)

adjusted_input = np.array([[rd_new, admin_new, marketing_new, state_encoded]])
adjusted_profit = model.predict(adjusted_input)[0]
st.info(f"📈 Adjusted Profit: ₹{adjusted_profit:,.2f}")

# ---------------- Scenario Comparison ----------------
st.subheader("📊 Compare Business Scenarios")

scenarios = {
    "Pessimistic": [rd_spend * 0.9, admin_spend * 0.9, marketing_spend * 0.9],
    "Realistic": [rd_spend, admin_spend, marketing_spend],
    "Optimistic": [rd_spend * 1.1, admin_spend * 1.1, marketing_spend * 1.1],
}

profits = {}
for s, vals in scenarios.items():
    x = np.array([[vals[0], vals[1], vals[2], state_encoded]])
    profits[s] = model.predict(x)[0]

fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.bar(profits.keys(), profits.values(), color=["#E57373", "#FFB74D", "#81C784"])
ax2.set_title("Profit Comparison Across Scenarios")
ax2.set_ylabel("Predicted Profit (₹)")
for i, val in enumerate(profits.values()):
    ax2.text(i, val, f"₹{val:,.0f}", ha='center', va='bottom')
st.pyplot(fig2)

# ---------------- Smart Advice Engine ----------------
st.subheader("🧠 Smart Advice Engine")
advice = []

category = st.selectbox("Select your startup category:", ["Food", "Tech", "Healthcare", "Education"])

if category == "Food":
    advice.append("🍴 Focus on supply chain optimization and delivery partnerships.")
    advice.append("Track seasonal demand patterns for better inventory planning.")
elif category == "Tech":
    advice.append("💻 Keep innovating — invest consistently in R&D.")
    advice.append("Apply for government startup grants in tech hubs like Karnataka or Telangana.")
elif category == "Education":
    advice.append("📚 Build an online learning platform with multilingual options.")
    advice.append("Focus on SEO and digital marketing for better reach.")
elif category == "Healthcare":
    advice.append("🩺 Ensure compliance with medical data and safety regulations.")
    advice.append("Invest in certifications to gain customer trust.")

if state in ["Maharashtra", "Delhi NCR"]:
    advice.append("🏙️ Expect higher operational costs — manage rent and admin expenses tightly.")
elif state in ["Karnataka", "Telangana"]:
    advice.append("🚀 Leverage government startup incentives and tech ecosystem.")
elif state in ["Kerala", "Tamil Nadu"]:
    advice.append("🌴 Build strong local brand loyalty through community engagement.")

if marketing_spend > rd_spend and marketing_spend > admin_spend:
    advice.append("📢 Heavy marketing — ensure campaigns are ROI-positive.")
elif rd_spend > marketing_spend:
    advice.append("🧪 Strong R&D — balance innovation with brand visibility.")
elif admin_spend > rd_spend:
    advice.append("🏢 High admin costs — optimize management structure.")

st.markdown("#### 💡 Personalized Advice:")
for tip in advice:
    st.markdown(f"- {tip}")

# ---------------- Financial Health Metrics ----------------
st.subheader("💼 Financial Health Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    monthly_expense = st.number_input("💰 Total Monthly Expenses (₹)", min_value=0.0, value=200000.0, step=10000.0)
    monthly_revenue = st.number_input("📈 Monthly Revenue (₹)", min_value=0.0, value=250000.0, step=10000.0)
with col2:
    current_funding = st.number_input("🏦 Total Available Funds (₹)", min_value=0.0, value=1000000.0, step=10000.0)
    customers = st.number_input("👥 Active Customers", min_value=1, value=100)
with col3:
    acquisition_cost = st.number_input("🎯 Cost to Acquire One Customer (₹)", min_value=0.0, value=5000.0, step=100.0)
    customer_lifetime = st.number_input("⏱️ Average Customer Lifetime (months)", min_value=1, value=12)

burn_rate = monthly_expense
runway = current_funding / burn_rate if burn_rate > 0 else 0
mrr = monthly_revenue
ltv = (monthly_revenue / customers) * customer_lifetime if customers > 0 else 0
break_even_point = burn_rate / (monthly_revenue / customers) if monthly_revenue > 0 else 0

st.markdown("### 📊 Key Metrics")
st.metric("🔥 Burn Rate", f"₹{burn_rate:,.0f} / month")
st.metric("⏳ Runway", f"{runway:.1f} months")
st.metric("💸 MRR", f"₹{mrr:,.0f}")
st.metric("🎯 CAC", f"₹{acquisition_cost:,.0f}")
st.metric("💎 LTV", f"₹{ltv:,.0f}")
st.metric("⚖️ Break-even Point", f"{break_even_point:.1f} customers")

# ---------------- Footer ----------------
st.markdown("---")
st.caption("💡 Made with ❤️ by Kamya Kapoor | Streamlit + ML + Gemini AI Business Assistant")








































