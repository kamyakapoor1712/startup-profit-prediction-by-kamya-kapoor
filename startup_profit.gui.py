import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Startup Profit Predictor",
    page_icon="🚀",
    layout="centered"
)

# ---------------- Styling ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to bottom right, #f8fafc, #eef2ff);
}
.card {
    background-color: #ffffff;
    border-radius: 14px;
    padding: 1.5rem;
    border: 1px solid #e5e7eb;
    box-shadow: 0 10px 25px rgba(0,0,0,0.05);
    margin-bottom: 1.6rem;
}
.section-title {
    font-size: 22px;
    font-weight: 600;
    color: #111827;
    margin-bottom: 1rem;
}
.profit-card {
    background: linear-gradient(to right, #ecfdf5, #d1fae5);
    border-left: 6px solid #10b981;
    padding: 1.5rem;
    border-radius: 12px;
    font-size: 24px;
    font-weight: 700;
    color: #065f46;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Load Model ----------------
model = joblib.load("mlr_predictor.joblib")

# ---------------- State Encoding ----------------
state_mapping = {
    "Maharashtra": 0, "Karnataka": 1, "Delhi NCR": 2, "Gujarat": 3,
    "Tamil Nadu": 4, "Telangana": 5, "West Bengal": 6,
    "Uttar Pradesh": 7, "Kerala": 8, "Rajasthan": 9
}

# ================= HEADER =================
st.title("🚀 Indian Startup Profit Predictor")
st.caption("AI-powered business forecasting & decision support system")
st.markdown("---")

# ================= BUSINESS DETAILS =================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📊 Business Details</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    rd_spend = st.number_input("R&D Spend (₹)", 0.0, 1e9, 100000.0, step=5000.0)
    admin_spend = st.number_input("Administration Spend (₹)", 0.0, 1e9, 120000.0, step=5000.0)
    marketing_spend = st.number_input("Marketing Spend (₹)", 0.0, 1e9, 150000.0, step=5000.0)

with c2:
    state = st.selectbox("State", list(state_mapping.keys()))
    category = st.selectbox("Startup Category", ["Tech", "Food", "Healthcare", "Education"])
    chart_type = st.selectbox("Chart Type", ["Bar", "Line", "Scatter"])

st.markdown('</div>', unsafe_allow_html=True)

# ================= PROFIT PREDICTION =================
state_encoded = state_mapping[state]
X = np.array([[rd_spend, admin_spend, marketing_spend, state_encoded]])
base_profit = model.predict(X)[0]

st.markdown(
    f'<div class="profit-card">💰 Predicted Profit: ₹{base_profit:,.2f}</div>',
    unsafe_allow_html=True
)

# ================= WHAT-IF & SCENARIOS =================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🔁 What-If & Scenario Analysis</div>', unsafe_allow_html=True)

left, right = st.columns(2)

with left:
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

with right:
    scenarios = {"Pessimistic": 0.9, "Realistic": 1.0, "Optimistic": 1.1}
    profits = []

    for f in scenarios.values():
        x = np.array([[rd_spend*f, admin_spend*f, marketing_spend*f, state_encoded]])
        profits.append(model.predict(x)[0])

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(scenarios.keys(), profits)
    ax.set_ylabel("Profit (₹)")
    ax.grid(alpha=0.2)
    st.pyplot(fig)

st.markdown('</div>', unsafe_allow_html=True)

# ================= FINANCIAL HEALTH =================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">💼 Financial Health Matrix</div>', unsafe_allow_html=True)

f1, f2, f3 = st.columns(3)

monthly_expense = f1.number_input("Monthly Expenses (₹)", 0.0, 1e9, 200000.0)
monthly_revenue = f1.number_input("Monthly Revenue (₹)", 0.0, 1e9, 250000.0)

current_funding = f2.number_input("Available Funds (₹)", 0.0, 1e9, 1000000.0)
customers = f2.number_input("Active Customers", 1, 1_000_000, 100)

cac = f3.number_input("Customer Acquisition Cost (₹)", 0.0, 1e6, 5000.0)
lifetime = f3.number_input("Customer Lifetime (months)", 1, 120, 12)

burn_rate = monthly_expense
runway = current_funding / burn_rate if burn_rate else 0
ltv = (monthly_revenue / customers) * lifetime if customers else 0

st.markdown('</div>', unsafe_allow_html=True)

# ================= AI ADVICE ENGINE (MIXED) =================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🧠 AI Smart Advice Engine</div>', unsafe_allow_html=True)

advice = []

if base_profit < 0:
    advice.append("❌ The startup is loss-making. Immediate cost control is required.")
elif base_profit < 100000:
    advice.append("⚠️ Low profit margins. Focus on efficiency and revenue growth.")
else:
    advice.append("✅ Profitable business. Reinvest profits for scaling.")

if runway < 6:
    advice.append("🔥 Cash runway below 6 months. Fundraising or cost reduction is urgent.")
elif runway < 12:
    advice.append("⚠️ Moderate runway. Plan funding within 6–9 months.")
else:
    advice.append("🛡️ Strong financial runway.")

if ltv < cac:
    advice.append("📉 LTV is lower than CAC. Marketing strategy needs optimization.")
else:
    advice.append("📈 Healthy unit economics.")

if marketing_spend > rd_spend:
    advice.append("📣 Marketing-heavy strategy. Ensure ROI tracking.")
elif rd_spend > marketing_spend:
    advice.append("🔬 Innovation-focused strategy. Improve market visibility.")

if category == "Tech":
    advice.append("💻 Focus on scalability, cloud optimization, and IP protection.")
elif category == "Food":
    advice.append("🍽️ Optimize supply chain and reduce wastage.")
elif category == "Healthcare":
    advice.append("🏥 Prioritize compliance and data security.")
elif category == "Education":
    advice.append("🎓 Strengthen digital platforms and learner engagement.")

if state in ["Maharashtra", "Delhi NCR"]:
    advice.append("🏙️ Metro region: manage high operational costs carefully.")
elif state in ["Karnataka", "Telangana"]:
    advice.append("🚀 Leverage startup ecosystem and government incentives.")

for tip in advice:
    st.markdown(f"- {tip}")

st.markdown('</div>', unsafe_allow_html=True)

# ================= KEY METRICS =================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📌 Key Metrics</div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
k1.metric("🔥 Burn Rate", f"₹{burn_rate:,.0f}")
k2.metric("⏳ Runway", f"{runway:.1f} months")
k3.metric("💎 LTV", f"₹{ltv:,.0f}")
k4.metric("🎯 CAC", f"₹{cac:,.0f}")

st.markdown('</div>', unsafe_allow_html=True)

# ================= FOOTER =================
st.caption("💡 Made with ❤️ by Kamya Kapoor | Streamlit + Machine Learning")
























































