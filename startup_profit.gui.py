import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Page Setup ----------------
st.set_page_config(page_title="Startup Profit Predictor", layout="centered")

# ---------------- Load Model ----------------
model = joblib.load("mlr_predictor.joblib")

# ---------------- State Encoding ----------------
state_mapping = {
    "Maharashtra": 0, "Karnataka": 1, "Delhi NCR": 2, "Gujarat": 3,
    "Tamil Nadu": 4, "Telangana": 5, "West Bengal": 6,
    "Uttar Pradesh": 7, "Kerala": 8, "Rajasthan": 9
}

# ---------------- Title ----------------
st.title("🚀 Indian Startup Profit Predictor")
st.caption("AI-based profit forecasting & business decision support")
st.markdown("---")

# ================= BUSINESS DETAILS =================
st.subheader("📊 Business Details")

col1, col2 = st.columns(2)

with col1:
    rd_spend = st.number_input("R&D Spend (₹)", 0.0, 1e9, 100000.0, step=5000.0)
    admin_spend = st.number_input("Administration Spend (₹)", 0.0, 1e9, 120000.0, step=5000.0)
    marketing_spend = st.number_input("Marketing Spend (₹)", 0.0, 1e9, 150000.0, step=5000.0)

with col2:
    state = st.selectbox("State", list(state_mapping.keys()))
    category = st.selectbox("Startup Category", ["Tech", "Food", "Healthcare", "Education"])
    chart_type = st.selectbox("Chart Type", ["Bar", "Line", "Scatter"])

# ---------------- Prediction ----------------
state_encoded = state_mapping[state]
X = np.array([[rd_spend, admin_spend, marketing_spend, state_encoded]])
base_profit = model.predict(X)[0]

st.success(f"💰 Predicted Profit: ₹{base_profit:,.2f}")

st.markdown("---")

# ================= WHAT-IF & SCENARIOS =================
st.subheader("🔁 Business Impact Analysis")

left, right = st.columns(2)

# -------- What If --------
with left:
    st.markdown("### 🤔 What-If Analysis")

    rd_change = st.slider("R&D Change (%)", -50, 50, 0)
    admin_change = st.slider("Admin Change (%)", -50, 50, 0)
    marketing_change = st.slider("Marketing Change (%)", -50, 50, 0)

    X_adj = np.array([[
        rd_spend * (1 + rd_change / 100),
        admin_spend * (1 + admin_change / 100),
        marketing_spend * (1 + marketing_change / 100),
        state_encoded
    ]])

    adjusted_profit = model.predict(X_adj)[0]
    st.info(f"📈 Adjusted Profit: ₹{adjusted_profit:,.2f}")

# -------- Scenario Comparison --------
with right:
    st.markdown("### 📊 Scenario Comparison")

    scenarios = {
        "Pessimistic": [0.9, 0.9, 0.9],
        "Realistic": [1.0, 1.0, 1.0],
        "Optimistic": [1.1, 1.1, 1.1]
    }

    profits = []
    for s in scenarios.values():
        x = np.array([[rd_spend*s[0], admin_spend*s[1], marketing_spend*s[2], state_encoded]])
        profits.append(model.predict(x)[0])

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(scenarios.keys(), profits)
    ax.set_ylabel("₹ Profit")
    st.pyplot(fig)

st.markdown("---")

# ================= FINANCIAL HEALTH =================
st.subheader("💼 Financial Health Matrix")

c1, c2, c3 = st.columns(3)

monthly_expense = c1.number_input("Monthly Expenses (₹)", 0.0, 1e9, 200000.0)
monthly_revenue = c1.number_input("Monthly Revenue (₹)", 0.0, 1e9, 250000.0)

current_funding = c2.number_input("Available Funds (₹)", 0.0, 1e9, 1000000.0)
customers = c2.number_input("Active Customers", 1, 1_000_000, 100)

cac = c3.number_input("Customer Acquisition Cost (₹)", 0.0, 1e6, 5000.0)
lifetime = c3.number_input("Customer Lifetime (months)", 1, 120, 12)

burn_rate = monthly_expense
runway = current_funding / burn_rate if burn_rate else 0
ltv = (monthly_revenue / customers) * lifetime if customers else 0

st.markdown("---")

# ================= ADVICE ENGINE =================
st.subheader("🧠 Smart Advice Engine")

advice = []

if category == "Tech":
    advice += ["Invest consistently in R&D.", "Leverage incubators and startup grants."]
elif category == "Food":
    advice += ["Reduce wastage.", "Optimize supply chain & delivery."]
elif category == "Healthcare":
    advice += ["Ensure compliance.", "Build trust with certifications."]
elif category == "Education":
    advice += ["Focus on digital platforms.", "Use SEO and performance marketing."]

if marketing_spend > rd_spend:
    advice.append("Marketing-heavy strategy — monitor ROI.")
elif rd_spend > marketing_spend:
    advice.append("Innovation-led strategy — improve brand visibility.")

for tip in advice:
    st.markdown(f"- {tip}")

st.markdown("---")

# ================= KEY METRICS =================
st.subheader("📌 Key Metrics")

m1, m2, m3, m4 = st.columns(4)
m1.metric("🔥 Burn Rate", f"₹{burn_rate:,.0f}")
m2.metric("⏳ Runway", f"{runway:.1f} months")
m3.metric("💎 LTV", f"₹{ltv:,.0f}")
m4.metric("🎯 CAC", f"₹{cac:,.0f}")

# ---------------- Footer ----------------
st.markdown("---")
st.caption("Made with ❤️ by Kamya Kapoor | Streamlit + Machine Learning")

















































