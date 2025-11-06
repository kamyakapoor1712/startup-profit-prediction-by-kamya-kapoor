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
st.markdown("""
Predict your startup’s profit based on **expenses and location** — now with 
_What-If Analysis_, _Scenario Comparison_, and a 💡 **Smart Advice Engine!**
""")

# ---------------- Input Section ----------------
st.subheader("💼 Business Details")

col1, col2 = st.columns(2)
with col1:
    category = st.selectbox(
        "Startup Category",
        ["Food", "Tech", "Education", "Healthcare", "Retail", "Finance", "Other"],
        key="category"
    )
    rd_spend = st.number_input("R&D Spend (₹)", min_value=0.0, value=100000.0,
                               step=1000.0, key="rd_spend")
    admin_spend = st.number_input("Administration Spend (₹)", min_value=0.0,
                                  value=120000.0, step=1000.0, key="admin_spend")

with col2:
    marketing_spend = st.number_input("Marketing Spend (₹)", min_value=0.0,
                                      value=150000.0, step=1000.0, key="marketing_spend")
    state = st.selectbox("Select State", list(state_mapping.keys()), key="state")
    chart_type = st.selectbox("Select Graph Type", ["Bar", "Line", "Scatter"], key="chart_type")

# ---------------- Base Prediction ----------------
state_encoded = state_mapping[state]
base_input = np.array([[rd_spend, admin_spend, marketing_spend, state_encoded]])
base_profit = model.predict(base_input)[0]
st.success(f"💰 Predicted Profit: ₹{base_profit:,.2f}")

# ---------------- Graph: Input vs Profit ----------------
st.subheader("📊 How Inputs Affect Predicted Profit")

features = ["R&D Spend", "Administration", "Marketing Spend", "Predicted Profit"]
values = [rd_spend, admin_spend, marketing_spend, base_profit]

fig1, ax1 = plt.subplots(figsize=(7, 4))
ax1.set_title("Input Impact on Predicted Profit")
ax1.set_ylabel("Value (₹)")

if chart_type == "Bar":
    colors = ["#FFB74D", "#4FC3F7", "#81C784", "#E57373"]
    ax1.bar(features, values, color=colors)
elif chart_type == "Line":
    ax1.plot(features, values, marker='o', color="#673AB7", linewidth=2)
elif chart_type == "Scatter":
    ax1.scatter(features, values, color="#388E3C", s=100)

ax1.text(3, base_profit, f"₹{base_profit:,.2f}",
         ha='center', va='bottom', fontsize=10, color='red')
st.pyplot(fig1)

# ---------------- What-If Sliders ----------------
st.subheader("🎯 What-If Analysis (Adjust Key Factors)")

colA, colB, colC = st.columns(3)
with colA:
    rd_change = st.slider("R&D Change (%)", -50, 50, 0, key="rd_slider")
with colB:
    admin_change = st.slider("Admin Change (%)", -50, 50, 0, key="admin_slider")
with colC:
    marketing_change = st.slider("Marketing Change (%)", -50, 50, 0, key="marketing_slider")

# Calculate new adjusted values
rd_new = rd_spend * (1 + rd_change / 100)
admin_new = admin_spend * (1 + admin_change / 100)
marketing_new = marketing_spend * (1 + marketing_change / 100)

adjusted_input = np.array([[rd_new, admin_new, marketing_new, state_encoded]])
adjusted_profit = model.predict(adjusted_input)[0]
st.info(f"📈 Adjusted Profit (What-If Result): ₹{adjusted_profit:,.2f}")

# ---------------- Scenario Comparison ----------------
st.subheader("📈 Compare Business Scenarios")

def predict_profit(rd, admin, marketing, state_encoded):
    return model.predict(np.array([[rd, admin, marketing, state_encoded]]))[0]

scenarios = {
    "Pessimistic": predict_profit(rd_spend * 0.9, admin_spend * 1.1, marketing_spend * 0.8, state_encoded),
    "Realistic": base_profit,
    "Optimistic": predict_profit(rd_spend * 1.2, admin_spend * 0.9, marketing_spend * 1.3, state_encoded)
}

colX, colY, colZ = st.columns(3)
colX.metric("Pessimistic", f"₹{scenarios['Pessimistic']:,.2f}")
colY.metric("Realistic", f"₹{scenarios['Realistic']:,.2f}")
colZ.metric("Optimistic", f"₹{scenarios['Optimistic']:,.2f}")

fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.bar(scenarios.keys(), scenarios.values(), color=["#FF8A65", "#FFD54F", "#81C784"])
ax2.set_title("Profit Comparison Across Scenarios")
ax2.set_ylabel("Predicted Profit (₹)")
for i, val in enumerate(scenarios.values()):
    ax2.text(i, val, f"₹{val:,.0f}", ha='center', va='bottom')
st.pyplot(fig2)

# ---------------- Smart Advice Engine ----------------
st.subheader("💡 Smart Advice Engine")

advice = ""
if category == "Food" and state == "Maharashtra":
    advice = "🍴 Food startups in Maharashtra face higher rent — allocate 5–10% extra to operations."
elif category == "Tech" and state == "Karnataka":
    advice = "💻 Tech startups in Karnataka often benefit from R&D tax rebates — reinvest savings into marketing."
elif category == "Healthcare" and state == "Delhi NCR":
    advice = "🏥 Healthcare firms in Delhi NCR have strict compliance costs — budget 8–12% for licensing."
elif category == "Retail" and state == "Gujarat":
    advice = "🛍️ Retail startups in Gujarat benefit from strong logistics — focus more on marketing expansion."
elif category == "Education" and state == "West Bengal":
    advice = "📚 Education startups in West Bengal can boost reach through hybrid (offline + online) models."
else:
    advice = "🚀 Optimize your spend — increase R&D slightly and reduce admin costs for better margins."

st.info(advice)

# ---------------- Footer ----------------
st.markdown("---")
st.caption("✨ Made with ❤️ by Kamya Kapoor | Enhanced with What-If Analysis, Scenario Comparison & Smart Advice Engine")












