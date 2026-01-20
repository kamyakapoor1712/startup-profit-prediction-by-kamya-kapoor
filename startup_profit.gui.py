import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Startup Profit Predictor",
    page_icon="📊",
    layout="centered"
)

# ---------------- CLEAN UI STYLE ----------------
st.markdown("""
<style>
.stApp {
    background-color: #f4f6fb;
}

.block-container {
    padding-top: 2rem;
}

h1 {
    font-weight: 700;
}

.section {
    margin-top: 2.2rem;
}

.result-box {
    background-color: white;
    border-left: 6px solid #2563eb;
    padding: 1.3rem;
    border-radius: 8px;
    font-size: 24px;
    font-weight: 600;
    margin-top: 1rem;
    margin-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("mlr_predictor.joblib")

state_mapping = {
    "Maharashtra": 0,
    "Karnataka": 1,
    "Delhi NCR": 2,
    "Gujarat": 3,
    "Tamil Nadu": 4
}

# ---------------- HEADER ----------------
st.title("Startup Profit Prediction System")
st.caption("AI-powered business forecasting & decision support tool")
st.divider()

# ================= BUSINESS DETAILS =================
st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("Business Details")

col1, col2 = st.columns(2)

with col1:
    rd_spend = st.number_input("R&D Spend (₹)", 0.0, 1e9, 100000.0, step=5000.0)
    admin_spend = st.number_input("Administration Spend (₹)", 0.0, 1e9, 120000.0, step=5000.0)
    marketing_spend = st.number_input("Marketing Spend (₹)", 0.0, 1e9, 150000.0, step=5000.0)

with col2:
    state = st.selectbox("State", list(state_mapping.keys()))
    startup_category = st.selectbox(
        "Startup Category",
        ["Tech", "Food", "Healthcare", "Education"]
    )

st.markdown('</div>', unsafe_allow_html=True)

# ================= PROFIT RESULT =================
state_encoded = state_mapping[state]
X = np.array([[rd_spend, admin_spend, marketing_spend, state_encoded]])
profit = model.predict(X)[0]

st.markdown(
    f'<div class="result-box">Predicted Annual Profit: ₹{profit:,.2f}</div>',
    unsafe_allow_html=True
)

# ================= WHAT-IF + SCENARIO =================
st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("What-If Analysis & Scenario Comparison")

c1, c2 = st.columns(2)

with c1:
    rd_change = st.slider("R&D Change (%)", -30, 30, 0)
    admin_change = st.slider("Admin Change (%)", -30, 30, 0)
    marketing_change = st.slider("Marketing Change (%)", -30, 30, 0)

adjusted_input = np.array([[
    rd_spend * (1 + rd_change / 100),
    admin_spend * (1 + admin_change / 100),
    marketing_spend * (1 + marketing_change / 100),
    state_encoded
]])

adjusted_profit = model.predict(adjusted_input)[0]

with c2:
    scenarios = {
        "Pessimistic": 0.9,
        "Normal": 1.0,
        "Optimistic": 1.1
    }

    profits = []
    for factor in scenarios.values():
        x = np.array([[rd_spend*factor, admin_spend*factor, marketing_spend*factor, state_encoded]])
        profits.append(model.predict(x)[0])

    fig, ax = plt.subplots()
    ax.bar(scenarios.keys(), profits, color="#2563eb")
    ax.set_ylabel("Profit (₹)")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig)

st.info(f"Adjusted Profit: ₹{adjusted_profit:,.2f}")
st.markdown('</div>', unsafe_allow_html=True)

# ================= FINANCIAL HEALTH =================
st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("Financial Health Overview")

col1, col2, col3 = st.columns(3)

monthly_expense = col1.number_input("Monthly Expense (₹)", 0.0, 1e9, 200000.0)
monthly_revenue = col2.number_input("Monthly Revenue (₹)", 0.0, 1e9, 250000.0)
available_funds = col3.number_input("Available Funds (₹)", 0.0, 1e9, 1000000.0)

burn_rate = monthly_expense
runway = available_funds / burn_rate if burn_rate else 0

m1, m2 = st.columns(2)
m1.metric("Burn Rate", f"₹{burn_rate:,.0f} / month")
m2.metric("Runway", f"{runway:.1f} months")

st.markdown('</div>', unsafe_allow_html=True)

# ================= ADVICE ENGINE =================
st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("AI-Driven Business Insights")

advice = {
    "Tech": [
        "Prioritize scalable architecture",
        "Maintain consistent R&D investment"
    ],
    "Food": [
        "Control supply chain costs",
        "Focus on demand forecasting"
    ],
    "Healthcare": [
        "Ensure regulatory compliance",
        "Invest in quality assurance"
    ],
    "Education": [
        "Adopt digital learning platforms",
        "Use performance-based marketing"
    ]
}

for tip in advice[startup_category]:
    st.write("•", tip)

st.markdown('</div>', unsafe_allow_html=True)

# ================= KEY METRICS =================
st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("Key Summary Metrics")

k1, k2, k3 = st.columns(3)
k1.metric("R&D Spend", f"₹{rd_spend:,.0f}")
k2.metric("Marketing Spend", f"₹{marketing_spend:,.0f}")
k3.metric("Predicted Profit", f"₹{profit:,.0f}")

st.markdown('</div>', unsafe_allow_html=True)

st.divider()
st.caption("Developed by Kamya Kapoor • Streamlit & Machine Learning")























































