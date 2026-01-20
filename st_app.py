import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Startup Profit Predictor",
    page_icon="🚀",
    layout="wide"
)

# ---------------- Load Model ----------------
model = joblib.load("mlr_predictor.joblib")

# ---------------- State Encoding ----------------
state_mapping = {
    "Maharashtra": 0, "Karnataka": 1, "Delhi NCR": 2, "Gujarat": 3,
    "Tamil Nadu": 4, "Telangana": 5, "West Bengal": 6,
    "Uttar Pradesh": 7, "Kerala": 8, "Rajasthan": 9
}

# ================= HERO =================
st.title("🚀 Indian Startup Profit Predictor")
st.caption("AI-powered profit forecasting & startup decision dashboard")
st.markdown("---")

# ================= SIDEBAR =================
st.sidebar.header("📊 Business Inputs")

rd_spend = st.sidebar.number_input("R&D Spend (₹)", 0.0, 1e9, 100000.0, step=5000.0)
admin_spend = st.sidebar.number_input("Administration Spend (₹)", 0.0, 1e9, 120000.0, step=5000.0)
marketing_spend = st.sidebar.number_input("Marketing Spend (₹)", 0.0, 1e9, 150000.0, step=5000.0)

state = st.sidebar.selectbox("State", list(state_mapping.keys()))
category = st.sidebar.selectbox(
    "Startup Category",
    ["Tech", "Food", "Healthcare", "Education"]
)

chart_type = st.sidebar.selectbox("Chart Type", ["Bar", "Line", "Scatter"])

# ================= PROFIT PREDICTION =================
state_encoded = state_mapping[state]
X = np.array([[rd_spend, admin_spend, marketing_spend, state_encoded]])
predicted_profit = model.predict(X)[0]

st.success(f"💰 **Predicted Annual Profit:** ₹{predicted_profit:,.2f}")

# ================= FINANCIAL HEALTH (MOVED UP) =================
st.markdown("## 💼 Financial Health Dashboard")

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

m1, m2, m3, m4 = st.columns(4)
m1.metric("🔥 Burn Rate", f"₹{burn_rate:,.0f}")
m2.metric("⏳ Runway", f"{runway:.1f} months")
m3.metric("💎 LTV", f"₹{ltv:,.0f}")
m4.metric("🎯 CAC", f"₹{cac:,.0f}")

st.markdown("---")

# ================= SMART ADVICE =================
st.markdown("## 🧠 Smart Advice Engine")

advice = []

if category == "Tech":
    advice += [
        "💻 Maintain consistent R&D investment.",
        "Leverage India’s startup grants & incubators."
    ]
elif category == "Food":
    advice += [
        "🍴 Optimize inventory & delivery logistics.",
        "Reduce wastage to improve margins."
    ]
elif category == "Healthcare":
    advice += [
        "🩺 Prioritize compliance & certifications.",
        "Build patient trust through transparency."
    ]
elif category == "Education":
    advice += [
        "📚 Focus on scalable digital platforms.",
        "Invest in performance marketing."
    ]

if marketing_spend > rd_spend:
    advice.append("📢 Marketing-heavy strategy — ensure ROI tracking.")
elif rd_spend > marketing_spend:
    advice.append("🧪 Innovation-driven — improve brand visibility.")

if state in ["Maharashtra", "Delhi NCR"]:
    advice.append("🏙️ Control high operational costs.")
elif state in ["Karnataka", "Telangana"]:
    advice.append("🚀 Leverage strong tech ecosystems.")

for tip in advice:
    st.markdown(f"- {tip}")

# ================= WHAT-IF =================
st.markdown("## 🤔 What-If Analysis")

c1, c2, c3 = st.columns(3)
rd_change = c1.slider("R&D Change (%)", -50, 50, 0)
admin_change = c2.slider("Admin Change (%)", -50, 50, 0)
marketing_change = c3.slider("Marketing Change (%)", -50, 50, 0)

X_adj = np.array([[
    rd_spend * (1 + rd_change / 100),
    admin_spend * (1 + admin_change / 100),
    marketing_spend * (1 + marketing_change / 100),
    state_encoded
]])

adjusted_profit = model.predict(X_adj)[0]
st.info(f"📈 **Adjusted Profit:** ₹{adjusted_profit:,.2f}")

# ================= VISUALS =================
st.markdown("## 📈 Input Impact Analysis")

features = ["R&D", "Admin", "Marketing", "Profit"]
values = [rd_spend, admin_spend, marketing_spend, predicted_profit]

fig, ax = plt.subplots(figsize=(6, 4))
if chart_type == "Bar":
    ax.bar(features, values)
elif chart_type == "Line":
    ax.plot(features, values, marker="o")
else:
    ax.scatter(features, values, s=100)

ax.set_ylabel("₹ Value")
ax.set_title("Impact of Inputs on Profit")
st.pyplot(fig)

# ================= FOOTER =================
st.markdown("---")
st.caption("Made  by **Kamya Kapoor** | Streamlit + Machine Learning")















