import tkinter as tk
from tkinter import messagebox, ttk
import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load trained model
model = joblib.load("mlr_predictor (1).joblib")

# State label encoding
state_mapping = {"California": 0, "Florida": 1, "New York": 2}

# --- GUI setup ---
root = tk.Tk()
root.title("Startup Profit Predictor")
root.geometry("700x650")

# ---------- INPUT SECTION ----------
tk.Label(root, text="R&D Spend").pack()
entry1 = tk.Entry(root)
entry1.pack()

tk.Label(root, text="Administration").pack()
entry2 = tk.Entry(root)
entry2.pack()

tk.Label(root, text="Marketing Spend").pack()
entry3 = tk.Entry(root)
entry3.pack()

tk.Label(root, text="State").pack()
state_var = tk.StringVar()
state_dropdown = ttk.Combobox(root, textvariable=state_var, values=list(state_mapping.keys()))
state_dropdown.pack()

# ---------- OUTPUT TYPE SELECTION ----------
tk.Label(root, text="Select Graph Type").pack()
output_var = tk.StringVar(value="Bar")
output_dropdown = ttk.Combobox(root, textvariable=output_var, values=["Bar", "Line", "Scatter"])
output_dropdown.pack()

# ---------- MATPLOTLIB SETUP ----------
fig, ax = plt.subplots(figsize=(6, 4))
ax.set_title("Feature Impact on Predicted Profit")
ax.set_ylabel("Value ($)")
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(pady=20)

# ---------- FUNCTION TO UPDATE GRAPH ----------
def update_graph():
    try:
        # Get user inputs
        rd = float(entry1.get())
        admin = float(entry2.get())
        marketing = float(entry3.get())
        state = state_var.get()

        if state not in state_mapping:
            return  # skip until valid state selected

        state_encoded = state_mapping[state]
        input_data = np.array([[rd, admin, marketing, state_encoded]])
        predicted_profit = model.predict(input_data)[0]

        # Create labels and values for visualization
        features = ["R&D Spend", "Administration", "Marketing Spend", "Predicted Profit"]
        values = [rd, admin, marketing, predicted_profit]

        # Clear and redraw
        ax.clear()
        ax.set_title("How Inputs Affect Predicted Profit")
        ax.set_ylabel("Value ($)")

        chart_type = output_var.get()

        if chart_type == "Bar":
            colors = ["skyblue", "orange", "lightgreen", "red"]
            ax.bar(features, values, color=colors)
        elif chart_type == "Line":
            ax.plot(features, values, marker='o', color='purple')
        elif chart_type == "Scatter":
            ax.scatter(features, values, color='darkgreen', s=100)

        # Annotate predicted profit
        ax.text(3, predicted_profit, f"${predicted_profit:,.2f}", ha='center', va='bottom', fontsize=10, color='red')

        canvas.draw()

    except Exception:
        pass  # ignore invalid or incomplete input

# ---------- AUTO UPDATE ON CHANGE ----------
def on_change(event=None):
    update_graph()

# Bind updates to input fields and dropdowns
for widget in [entry1, entry2, entry3, state_dropdown, output_dropdown]:
    widget.bind("<KeyRelease>", on_change)
    widget.bind("<<ComboboxSelected>>", on_change)

# ---------- PREDICT BUTTON ----------
def predict_button_action():
    try:
        rd = float(entry1.get())
        admin = float(entry2.get())
        marketing = float(entry3.get())
        state = state_var.get()

        if state not in state_mapping:
            raise ValueError("Please select a valid state.")

        state_encoded = state_mapping[state]
        input_data = np.array([[rd, admin, marketing, state_encoded]])
        predicted_profit = model.predict(input_data)[0]

        messagebox.showinfo("Predicted Profit", f"Predicted Profit: ${predicted_profit:,.2f}")

        update_graph()

    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {e}")

tk.Button(root, text="Predict", command=predict_button_action, bg="lightblue").pack(pady=10)

root.mainloop()
