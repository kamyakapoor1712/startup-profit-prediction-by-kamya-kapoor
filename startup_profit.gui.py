import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI # 👈 for AI assistant

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

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

# ---------------- AI Business Assistant Section ----------------
st.markdown("---")
st.header("💬 Talk to Your AI Business Assistant (ProfitBuddy)")

st.markdown(
    "Ask me anything — I can suggest strategies, funding ideas, expense optimization, and market-specific advice!"
)

# --- API Key ---
openai.api_key = st.secrets.get("OPENAI_API_KEY", "your-api-key-here")

user_input = st.text_input("Ask your business-related question:")

if user_input:
    try:
        with st.spinner("ProfitBuddy is thinking... 💭"):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a friendly startup business assistant for Indian founders."},
                    {"role": "user", "content": user_input}
                ]
            )
            answer = response["choices"][0]["message"]["content"]
            st.success(answer)
    except Exception as e:
        st.error(f"⚠️ Unable to connect to AI assistant: {e}")

# ---------------- Footer ----------------
st.markdown("---")
st.caption("💡 Made with ❤️ by Kamya Kapoor | Streamlit + ML + AI Business Assistant")















