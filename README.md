# 📊 Customer Churn & LTV Analytics & Prediction Platform

An end-to-end full-stack analytics/prediction solution to extract, engineer, and analyze customer churn and lifetime value (LTV) using SQL, Python, and deep learning — with an interactive Streamlit dashboard and SHAP explainability.

---

## 📁 Project Structure

```
.
├── datasets/
│   ├── csv-files/                 # Raw CRM & ERP data (bronze/silver/gold layers)
│   └── DataWarehouseAnalytics.bak # Backup of MySQL data warehouse
├── scripts/
│   ├── *.sql                      # Modular SQL scripts for feature engineering
│   ├── import_gold_to_mysql.py   # Load CSVs into MySQL
│   └── python_integration.py     # Feature generation, LTV calculation
├── features_customer_churn_ltv.csv # Final ML-ready dataset
├── churn_model.pt                # Trained single-task PyTorch model
├── multitask_model.pt           # Trained multi-task PyTorch model (churn + LTV)
├── scaler.pkl                    # StandardScaler for model inputs
├── train_multitask_model.py     # Script to train multi-task model
├── test_model.py                # Test script to run predictions from CLI
├── app.py                        # Streamlit dashboard (UI + SHAP)
├── requirements.txt
└── README.md
```

---

## 🧠 Key Highlights

### ✅ SQL Feature Engineering (First Phase)
- 14+ modular SQL scripts for customer segmentation, order frequency, LTV, churn flags, and recency.
- Designed using bronze → silver → gold data warehouse structure.
- MySQL integration with automated table creation from raw CSVs.

### 🔄 ETL + Python Integration
- Automated ingestion pipeline to load CSVs → MySQL using `import_gold_to_mysql.py`.
- EDA and feature joining scripts in Python to generate final `features_customer_churn_ltv.csv`.

### 🔮 Deep Learning (Multi-Task Learning)
- Trained PyTorch model to predict both churn (classification) and LTV (regression) in a **single network**.
- Balanced dataset using upsampling; multi-task loss optimized jointly.
- Achieved significant model generalization and business value prediction.

### 🧠 SHAP Explainability
- Integrated SHAP bar plots to explain churn predictions for individual customers.
- Lightweight `ExactExplainer` used for fast explanations.

### 📊 Streamlit Dashboard (Interactive UI)
- Filters for churn status, LTV range.
- Visuals: bar charts, scatter plots, box plots via Plotly.
- Real-time churn + LTV prediction with SHAP insights.
- Borderline warnings & input validation to guide non-technical users.

---

## 🚀 Run Locally

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train_multitask_model.py
```

### 3. Test the Model
```bash
python test_model.py
```

### 4. Launch Streamlit Dashboard
```bash
streamlit run app.py
```

---

## 📌 Screenshots

> 📉 Churn/LTV Distribution · 🔎 SHAP Explanation · 🧠 Real-time Prediction

![LTV Boxplot](churn_ltv_boxplot.png)

---

## View Live
[Streamlit](https://customer-churn-ltv-prediction-platform-prudhviraj.streamlit.app/)
