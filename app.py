import sys
import types
sys.modules['torch.classes'] = types.ModuleType('torch.classes')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import torch
import torch.nn as nn
import joblib
import shap

st.set_page_config(page_title="Customer Churn & LTV Dashboard", layout="wide")

# 🧠 Multi-task Model Definition
class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.churn_head = nn.Linear(16, 1)  # binary
        self.ltv_head = nn.Linear(16, 1)    # regression

    def forward(self, x):
        shared = self.shared(x)
        churn_out = self.churn_head(shared)
        ltv_out = self.ltv_head(shared)
        return churn_out, ltv_out

# 📦 Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("features_customer_churn_ltv.csv")
    df['is_churned'] = df['is_churned'].map({0: 'Not Churned', 1: 'Churned'})
    return df

df = load_data()

# 🔁 Load Model & Scaler
@st.cache_resource
def load_model():
    model = MultiTaskModel()
    model.load_state_dict(torch.load("multitask_model.pt", map_location=torch.device("cpu")))
    model.eval()
    return model

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

model = load_model()
scaler = load_scaler()

# ✅ Helper to run prediction
def predict_churn_ltv(order_count, ltv, days_since):
    input_array = np.array([[order_count, ltv, days_since]])
    scaled_input = scaler.transform(input_array)
    input_tensor = torch.tensor(scaled_input, dtype=torch.float32)

    with torch.no_grad():
        churn_logit, ltv_pred = model(input_tensor)
        churn_prob = torch.sigmoid(churn_logit).item()
        ltv_value = ltv_pred.item()

    return churn_prob, ltv_value

# 🔍 Sidebar Filters
st.sidebar.header("🔍 Filter Data")
min_ltv, max_ltv = st.sidebar.slider("Select Lifetime Value Range", 
                                     float(df['lifetime_value'].min()), 
                                     float(df['lifetime_value'].max()), 
                                     (float(df['lifetime_value'].min()), float(df['lifetime_value'].max())))

churn_status = st.sidebar.multiselect("Select Churn Status", 
                                      options=df['is_churned'].unique(), 
                                      default=df['is_churned'].unique())

filtered_df = df[(df['lifetime_value'] >= min_ltv) & 
                 (df['lifetime_value'] <= max_ltv) & 
                 (df['is_churned'].isin(churn_status))]

# 📊 Dashboard
st.title("📊 Customer Churn & Lifetime Value Dashboard")

st.markdown("### 1️⃣ Churn Distribution")
churn_counts = filtered_df['is_churned'].value_counts().reset_index()
churn_counts.columns = ['Churn Status', 'Customer Count']
churn_fig = px.bar(churn_counts, x='Churn Status', y='Customer Count', color='Churn Status', text_auto=True)
st.plotly_chart(churn_fig, use_container_width=True)

st.markdown("### 2️⃣ LTV by Churn Status")
ltv_fig = px.box(filtered_df, x='is_churned', y='lifetime_value', color='is_churned')
st.plotly_chart(ltv_fig, use_container_width=True)

st.markdown("### 3️⃣ Order Count vs LTV")
scatter_fig = px.scatter(filtered_df, x='order_count', y='lifetime_value', color='is_churned', hover_data=['customer_id'])
st.plotly_chart(scatter_fig, use_container_width=True)

# 🔮 Prediction Interface
st.markdown("### 🔮 Predict Churn & LTV for a New Customer")
col1, col2 = st.columns(2)
with col1:
    order_count = st.number_input("Order Count", min_value=0, step=1)
    lifetime_value = st.number_input("Lifetime Value", min_value=0.0, step=10.0)
with col2:
    days_since_last_order = st.number_input("Days Since Last Order", min_value=0, step=1)

# 🔘 Predict Button
if st.button("Predict Churn & LTV"):
    user_input = np.array([order_count, lifetime_value, days_since_last_order])
    
    # 🚫 Input validation
    if np.all(user_input == 0):
        st.warning("⚠️ Please enter realistic non-zero values for prediction.")
    else:
        with st.spinner("Predicting..."):
            churn_prob, ltv_value = predict_churn_ltv(*user_input)

            label = "Churned" if churn_prob >= 0.5 else "Not Churned"
            st.success(f"🧠 Prediction: **{label}** (Confidence: {churn_prob:.2f})")
            st.info(f"💰 Predicted LTV: **${ltv_value:.2f}**")

            if abs(churn_prob - 0.5) < 0.1:
                st.warning("⚠️ The prediction is uncertain. We recommend checking the customer's recent activity or using more information for a clearer result.")

            # SHAP Explanations
            st.markdown("### 🔎 Why this prediction?")
            background_data = df[["order_count", "lifetime_value", "days_since_last_order"]].sample(n=50, random_state=42).fillna(0)
            background_scaled = scaler.transform(background_data)

            def model_predict_fn(input_array):
                tensor = torch.tensor(input_array, dtype=torch.float32)
                with torch.no_grad():
                    churn_logits, _ = model(tensor)
                    return torch.sigmoid(churn_logits).numpy()

            explainer = shap.Explainer(model_predict_fn, background_scaled)
            shap_values = explainer(scaler.transform([user_input]))

            try:
                fig, ax = plt.subplots()
                shap.plots.bar(shap_values[0], show=False, max_display=5)
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"⚠️ SHAP plot failed to render: {e}")


# 🔍 Data Sample
st.markdown("### 📋 Sample Data")
st.dataframe(filtered_df.head(50))

# ✅ Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Created with ❤️ by <b>Prudhvi Raj</b> using PyTorch, SHAP, and Streamlit"
    "</div>",
    unsafe_allow_html=True
)
