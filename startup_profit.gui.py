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
st.set_page_config(page_title="Startup Profit Predictor + AI Assistant", layout="centered")
st.title("🚀 Indian Startup Profit Predictor + 🤖 Business Assistant")
st.markdown(
    "Predict your startup’s profit and get smart AI-powered business advice!"
)

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
st.subheader("🧠 Smart Advice Engine")
advice = []  # ✅ Initialize list here
# Business category input
category = st.selectbox("Select your startup category:", ["Food", "Tech", "Healthcare", "Education"])
# Category-based advice
if category == "Food":
    advice.append("🍴 Food businesses in [State] often face higher rent — allocate 5–10% extra for premises.")
    advice.append("Focus on local supply chains and online delivery platforms.")
elif category == "Tech":
    advice.append("💻 Tech startups thrive on R&D — keep investing in product innovation.")
    advice.append("Consider government grants for IT-based innovations.")
elif category == "Education":
    advice.append("📚 Education startups grow through digital outreach — invest in online presence.")
    advice.append("In Karnataka or Delhi NCR, bilingual content helps expand reach.")
elif category == "Healthcare":
    advice.append("🩺 Healthcare startups face higher compliance costs — set aside funds for certifications.")
elif category == "Retail":
    advice.append("🛍️ Marketing is key — allocate at least 30% of spend to brand promotion.")
elif category == "Finance":
    advice.append("💰 Keep higher admin reserves for licensing and audits, especially in Delhi NCR.")

# State-based advice
if state in ["Maharashtra", "Delhi NCR"]:
    advice.append("🏙️ High operational costs — focus on rent and administrative efficiency.")
elif state in ["Karnataka", "Telangana"]:
    advice.append("🚀 Great for tech startups — leverage government startup incentives.")
elif state in ["Kerala", "Tamil Nadu"]:
    advice.append("🌴 Local customer trust is vital — use community-centric marketing.")

# Expense pattern advice
if marketing_spend > rd_spend and marketing_spend > admin_spend:
    advice.append("📢 Heavy marketing spend — track performance to ensure high ROI.")
elif rd_spend > marketing_spend:
    advice.append("🧪 Strong R&D focus — balance with visibility for faster product adoption.")
elif admin_spend > rd_spend:
    advice.append("🏢 High administrative costs — optimize management overheads.")

# Display advice
if advice:
    for tip in advice:
        st.markdown(f"- {tip}")
else:
    st.markdown("✅ Your spending looks balanced — maintain efficiency for steady growth.")
    # ---------------- Financial Health Metrics ----------------
st.subheader("💼 Financial Health Metrics")

st.markdown("Understand your startup’s sustainability and profitability metrics:")

col1, col2, col3 = st.columns(3)

with col1:
    monthly_expense = st.number_input("💰 Total Monthly Expenses (₹)", min_value=0.0, value=200000.0, step=10000.0)
    monthly_revenue = st.number_input("📈 Monthly Revenue (₹)", min_value=0.0, value=250000.0, step=10000.0)
with col2:
    current_funding = st.number_input("🏦 Total Available Funds (₹)", min_value=0.0, value=1000000.0, step=10000.0)
    customers = st.number_input("👥 Active Customers", min_value=1, value=100)
with col3:
    acquisition_cost = st.number_input("🎯 Cost to Acquire One Customer (CAC, ₹)", min_value=0.0, value=5000.0, step=100.0)
    customer_lifetime = st.number_input("⏱️ Average Customer Lifetime (months)", min_value=1, value=12)

# ---- Calculations ----
burn_rate = monthly_expense  # spending per month
runway = current_funding / burn_rate if burn_rate > 0 else 0
mrr = monthly_revenue
ltv = (monthly_revenue / customers) * customer_lifetime if customers > 0 else 0
break_even_point = burn_rate / (monthly_revenue / customers) if monthly_revenue > 0 else 0

# ---- Display Results ----
st.markdown("### 📊 Key Metrics")
st.metric("🔥 Burn Rate", f"₹{burn_rate:,.0f} / month")
st.metric("⏳ Runway", f"{runway:.1f} months", delta=None)
st.metric("💸 Monthly Recurring Revenue (MRR)", f"₹{mrr:,.0f}")
st.metric("🎯 Customer Acquisition Cost (CAC)", f"₹{acquisition_cost:,.0f}")
st.metric("💎 Customer Lifetime Value (LTV)", f"₹{ltv:,.0f}")
st.metric("⚖️ Break-even Point", f"{break_even_point:.1f} customers")

# ---- Insights ----
st.markdown("### 🧩 Financial Insights")

insights = []

if runway < 6:
    insights.append("⚠️ Your runway is short — consider reducing expenses or raising more funds.")
elif runway < 12:
    insights.append("🟡 You have a moderate runway. Plan fundraising within the next 6 months.")
else:
    insights.append("✅ Strong runway — you’re financially stable for now.")

if ltv < acquisition_cost:
    insights.append("🚨 Your LTV is lower than CAC — you’re losing money on each customer!")
elif ltv < acquisition_cost * 3:
    insights.append("🟠 LTV:CAC ratio is average — aim for 3x or higher for sustainable growth.")
else:
    insights.append("💚 Excellent LTV:CAC ratio — your growth is efficient and profitable.")

if break_even_point > customers:
    insights.append("📉 You haven’t reached break-even yet — need more customers or higher MRR.")
else:
    insights.append("💪 You’re operating at or beyond break-even — great work!")

for msg in insights:
    st.markdown(f"- {msg}")
    # ---------------- Smart Alerts ----------------
st.markdown("### 🚨 Smart Spending Alerts")

st.write("Get automatic alerts when your spending exceeds safe limits.")

# Define categories and spend
expense_categories = {
    "Marketing": st.number_input("📢 Marketing Spend (₹)", min_value=0.0, value=50000.0, step=5000.0),
    "Salaries": st.number_input("👩‍💼 Salaries & Team (₹)", min_value=0.0, value=100000.0, step=5000.0),
    "Operations": st.number_input("🏭 Operations & Logistics (₹)", min_value=0.0, value=30000.0, step=5000.0),
    "Technology": st.number_input("💻 Tech / Cloud Services (₹)", min_value=0.0, value=20000.0, step=5000.0),
    "Miscellaneous": st.number_input("🧾 Miscellaneous (₹)", min_value=0.0, value=10000.0, step=5000.0)
}

total_expense = sum(expense_categories.values())
alert_messages = []

# Alert rules
for category, amount in expense_categories.items():
    percent = (amount / total_expense) * 100 if total_expense > 0 else 0

    if percent > 40:
        alert_messages.append(f"🚨 {category} spending is **{percent:.1f}%** of total — too high! Consider rebalancing.")
    elif percent > 25:
        alert_messages.append(f"⚠️ {category} is taking {percent:.1f}% of your total spend — review if necessary.")
    else:
        alert_messages.append(f"✅ {category} spend ({percent:.1f}%) is within a healthy range.")

st.markdown("#### 💬 Spending Analysis")
for msg in alert_messages:
    st.markdown(f"- {msg}")

# --------------- Financial Forecast ----------------
st.markdown("---")
st.subheader("📈 Financial Forecast (Next 3–12 Months)")

st.write("Estimate your startup’s financial growth over time based on your current performance.")

forecast_months = st.slider("Select forecast duration (months)", 3, 12, 6)
growth_rate = st.slider("Expected monthly revenue growth (%)", 0.0, 50.0, 10.0)
expense_growth = st.slider("Expected monthly expense growth (%)", 0.0, 30.0, 5.0)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

months = np.arange(1, forecast_months + 1)
forecast_data = pd.DataFrame({
    "Month": months,
    "Revenue": monthly_revenue * ((1 + growth_rate/100) ** months),
    "Expenses": monthly_expense * ((1 + expense_growth/100) ** months)
})
forecast_data["Profit"] = forecast_data["Revenue"] - forecast_data["Expenses"]

# Plot
st.markdown("### 📊 Revenue vs Expense Forecast")
fig, ax = plt.subplots()
ax.plot(forecast_data["Month"], forecast_data["Revenue"], label="Revenue", linewidth=2)
ax.plot(forecast_data["Month"], forecast_data["Expenses"], label="Expenses", linewidth=2)
ax.plot(forecast_data["Month"], forecast_data["Profit"], label="Profit", linewidth=2)
ax.set_xlabel("Month")
ax.set_ylabel("Amount (₹)")
ax.legend()
st.pyplot(fig)

# Quick Summary
final_profit = forecast_data["Profit"].iloc[-1]
if final_profit < 0:
    st.error(f"🚨 In {forecast_months} months, you’ll be operating at a **loss of ₹{abs(final_profit):,.0f}**.")
else:
    st.success(f"✅ Projected **profit after {forecast_months} months**: ₹{final_profit:,.0f}.")
    # ---------------- Smart Alerts ----------------
st.markdown("### 🚨 Smart Spending Alerts")

st.write("Get automatic alerts when your spending exceeds safe limits.")

# Define categories and spend
expense_categories = {
    "Marketing": st.number_input("📢 Marketing Spend (₹)", min_value=0.0, value=50000.0, step=5000.0),
    "Salaries": st.number_input("👩‍💼 Salaries & Team (₹)", min_value=0.0, value=100000.0, step=5000.0),
    "Operations": st.number_input("🏭 Operations & Logistics (₹)", min_value=0.0, value=30000.0, step=5000.0),
    "Technology": st.number_input("💻 Tech / Cloud Services (₹)", min_value=0.0, value=20000.0, step=5000.0),
    "Miscellaneous": st.number_input("🧾 Miscellaneous (₹)", min_value=0.0, value=10000.0, step=5000.0)
}

total_expense = sum(expense_categories.values())
alert_messages = []
# ---------------- Footer ----------------
st.markdown("---")
st.caption("💡 Made with ❤️ by Kamya Kapoor | Streamlit + ML + AI Business Assistant")






























