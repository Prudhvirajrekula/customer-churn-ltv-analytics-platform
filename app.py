import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Customer Churn & LTV Dashboard", layout="wide")

# Load data
df = pd.read_csv("features_customer_churn_ltv.csv")
df['is_churned'] = df['is_churned'].map({0: 'Not Churned', 1: 'Churned'})

# Sidebar filters
st.sidebar.header("🔍 Filter Data")
min_ltv, max_ltv = st.sidebar.slider("Select Lifetime Value Range", 
                                     float(df['lifetime_value'].min()), 
                                     float(df['lifetime_value'].max()), 
                                     (float(df['lifetime_value'].min()), float(df['lifetime_value'].max())))

churn_status = st.sidebar.multiselect("Select Churn Status", 
                                      options=df['is_churned'].unique(), 
                                      default=df['is_churned'].unique())

# Apply filters
filtered_df = df[(df['lifetime_value'] >= min_ltv) & 
                 (df['lifetime_value'] <= max_ltv) & 
                 (df['is_churned'].isin(churn_status))]

# Title
st.title("📊 Customer Churn & Lifetime Value Dashboard")

# Churn Distribution
st.markdown("### 1️⃣ Churn Distribution")
churn_counts = filtered_df['is_churned'].value_counts().reset_index()
churn_counts.columns = ['Churn Status', 'Customer Count']
churn_fig = px.bar(churn_counts, x='Churn Status', y='Customer Count', 
                   color='Churn Status', text_auto=True,
                   title="Customer Distribution by Churn Status")
churn_fig.update_traces(hovertemplate='Status: %{x}<br>Count: %{y}')
st.plotly_chart(churn_fig, use_container_width=True)

# LTV by Churn - Boxplot
st.markdown("### 2️⃣ Lifetime Value Distribution by Churn Status")
ltv_fig = px.box(filtered_df, x='is_churned', y='lifetime_value', color='is_churned',
                 labels={'is_churned': 'Churn Status', 'lifetime_value': 'Lifetime Value'},
                 title="LTV Spread by Churn")
ltv_fig.update_traces(hovertemplate='Churn: %{x}<br>LTV: %{y}')
st.plotly_chart(ltv_fig, use_container_width=True)

# Order Count vs LTV Scatter
st.markdown("### 3️⃣ Order Count vs LTV")
scatter_fig = px.scatter(filtered_df, x='order_count', y='lifetime_value', color='is_churned',
                         hover_data=['customer_id'],
                         labels={'order_count': 'Order Count', 'lifetime_value': 'Lifetime Value', 'is_churned': 'Churn Status'},
                         title="Order Count vs LTV by Churn")
scatter_fig.update_traces(marker=dict(size=8), hovertemplate="Customer ID: %{customdata[0]}<br>Orders: %{x}<br>LTV: %{y}<br>Churn: %{marker.color}")
st.plotly_chart(scatter_fig, use_container_width=True)

# Show sample data
st.markdown("### 🔍 Sample Data Preview")
st.dataframe(filtered_df.head(50))

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Created with ❤️ by <b>Prudhvi Raj</b> using Streamlit"
    "</div>",
    unsafe_allow_html=True
)