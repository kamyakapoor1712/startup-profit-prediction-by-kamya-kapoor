import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("mlr_predictor (1).joblib")

# State label encoding
state_mapping = {"California": 0, "Florida": 1, "New York": 2}

# --- Page setup ---
st.set_page_config(page_title="Startup Profit Predictor", layout="centered")
st.title("💼 Startup Profit Predictor")
st.markdown("Predict your startup's profit based on expenses and location.")

# ---------- INPUT SECTION ----------
st.subheader("Enter Business Details")

col1, col2 = st.columns(2)
with col1:
    rd = st.number_input("R&D Spend ($)", min_value=0.0, value=100000.0, step=1000.0)
    admin = st.number_input("Administration ($)", min_value=0.0, value=120000.0, step=1000.0)
with col2:
    marketing = st.number_input("Marketing Spend ($)", min_value=0.0, value=150000.0, step=1000.0)
    state = st.selectbox("State", list(state_mapping.keys()))

# ---------- GRAPH TYPE SELECTION ----------
chart_type = st.selectbox("Select Graph Type", ["Bar", "Line", "Scatter"])

# ---------- PREDICTION ----------
state_encoded = state_mapping[state]
input_data = np.array([[rd, admin, marketing, state_encoded]])
predicted_profit = model.predict(input_data)[0]

st.success(f"### 💰 Predicted Profit: ${predicted_profit:,.2f}")

# ---------- VISUALIZATION ----------
st.subheader("📊 How Inputs Affect Predicted Profit")

features = ["R&D Spend", "Administration", "Marketing Spend", "Predicted Profit"]
values = [rd, admin, marketing, predicted_profit]

fig, ax = plt.subplots(figsize=(7, 4))
ax.set_title("How Inputs Affect Predicted Profit")
ax.set_ylabel("Value ($)")

if chart_type == "Bar":
    colors = ["skyblue", "orange", "lightgreen", "red"]
    ax.bar(features, values, color=colors)
elif chart_type == "Line":
    ax.plot(features, values, marker='o', color='purple')
elif chart_type == "Scatter":
    ax.scatter(features, values, color='darkgreen', s=100)

# Annotate predicted profit
ax.text(3, predicted_profit, f"${predicted_profit:,.2f}", ha='center', va='bottom', fontsize=10, color='red')

st.pyplot(fig)

# ---------- FOOTER ----------
st.markdown("---")
st.caption("Made with ❤️ using Streamlit and Machine Learning")

