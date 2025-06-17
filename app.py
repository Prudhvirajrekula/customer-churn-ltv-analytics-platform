
import sys
import types

# Prevent Streamlit from trying to inspect torch.classes
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

st.set_page_config(page_title="Churn & LTV Insights", layout="wide")

# 🧠 Define the PyTorch Model
class ChurnModel(nn.Module):
    def __init__(self):
        super(ChurnModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

# 📦 Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("features_customer_churn_ltv.csv")
    df['is_churned'] = df['is_churned'].map({0: 'Not Churned', 1: 'Churned'})
    return df

# 🔁 Load Model and Scaler
@st.cache_resource
def load_model():
    model = torch.load("churn_model.pt", map_location=torch.device("cpu"), weights_only=False)
    model.eval()
    return model

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

@st.cache_resource
def get_shap_explainer(_model, background_scaled):
    def model_predict(input_array):
        input_tensor = torch.tensor(input_array, dtype=torch.float32)
        with torch.no_grad():
            logits = _model(input_tensor)
            probs = torch.sigmoid(logits).numpy()
        return probs
    return shap.Explainer(model_predict, background_scaled)

# Initialize
df = load_data()
model = load_model()
scaler = load_scaler()

# 🌐 Page Title
st.title("📊 Customer Churn & Lifetime Value Analytics")

# 🔍 Sidebar Filters
st.sidebar.title("🔧 Data Filters")
min_ltv, max_ltv = st.sidebar.slider(
    "Select Lifetime Value Range",
    float(df['lifetime_value'].min()),
    float(df['lifetime_value'].max()),
    (float(df['lifetime_value'].min()), float(df['lifetime_value'].max()))
)

churn_status = st.sidebar.multiselect(
    "Select Churn Status",
    options=df['is_churned'].unique(),
    default=df['is_churned'].unique()
)

filtered_df = df[
    (df['lifetime_value'] >= min_ltv) &
    (df['lifetime_value'] <= max_ltv) &
    (df['is_churned'].isin(churn_status))
]

# 📈 Charts
st.markdown("## 🧮 Customer Insights Dashboard")

st.markdown("### 1️⃣ Churn Distribution")
churn_counts = filtered_df['is_churned'].value_counts().reset_index()
churn_counts.columns = ['Churn Status', 'Customer Count']
fig1 = px.bar(churn_counts, x='Churn Status', y='Customer Count', color='Churn Status', text_auto=True)
st.plotly_chart(fig1, use_container_width=True)

st.markdown("### 2️⃣ Lifetime Value by Churn Status")
fig2 = px.box(filtered_df, x='is_churned', y='lifetime_value', color='is_churned')
st.plotly_chart(fig2, use_container_width=True)

st.markdown("### 3️⃣ Order Count vs Lifetime Value")
fig3 = px.scatter(
    filtered_df,
    x='order_count',
    y='lifetime_value',
    color='is_churned',
    hover_data=['customer_id'],
    title="Orders vs LTV Colored by Churn"
)
st.plotly_chart(fig3, use_container_width=True)

# 🤖 Prediction Section
st.markdown("## 🔮 Predict Churn for a New Customer")

st.info("Adjust the inputs below to see how they impact churn prediction.")

col1, col2, col3 = st.columns(3)
with col1:
    order_count = st.number_input("🛒 Order Count", min_value=0, step=1, help="Total number of orders placed by the customer.")
with col2:
    lifetime_value = st.number_input("💰 Lifetime Value", min_value=0.0, step=10.0, help="Cumulative revenue from the customer.")
with col3:
    days_since_last_order = st.number_input("📆 Days Since Last Order", min_value=0, step=1, help="Time since the customer's last order.")

if st.button("📍 Predict Churn"):
    with st.spinner("Running prediction and generating SHAP explanation..."):
        # Step 1: Make prediction
        input_data = np.array([[order_count, lifetime_value, days_since_last_order]])
        scaled_input = scaler.transform(input_data)
        tensor_input = torch.tensor(scaled_input, dtype=torch.float32)

        with torch.no_grad():
            logits = model(tensor_input)
            prob = torch.sigmoid(logits).item()

        label = "Churned" if prob >= 0.5 else "Not Churned"
        color = "🔴" if label == "Churned" else "🟢"

        # Step 2: Generate SHAP
        background_data = df[["order_count", "lifetime_value", "days_since_last_order"]].sample(n=50, random_state=42).fillna(0)
        background_scaled = scaler.transform(background_data)
        explainer = get_shap_explainer(model, background_scaled)
        shap_values = explainer(scaled_input)
        shap_values.feature_names = ["Order Count", "Lifetime Value", "Days Since Last Order"]

    # 🎯 Results (displayed AFTER everything is done)
    st.metric(label="🧠 Prediction", value=f"{color} {label}", delta=f"Confidence: {prob:.2f}")

    with st.expander("🔎 Show SHAP Explanation", expanded=True):
        st.caption("This breakdown shows how each feature influenced the model's prediction.")
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values[0], show=False)
        st.pyplot(fig)


# 📋 Preview Data
st.markdown("## 🔍 Sample of Filtered Customer Data")
st.dataframe(filtered_df.head(50))

# 👣 Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Created with ❤️ by <b>Prudhvi Raj</b> | Powered by PyTorch, SHAP & Streamlit"
    "</div>",
    unsafe_allow_html=True
)
