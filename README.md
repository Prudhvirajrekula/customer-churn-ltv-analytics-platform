# 📊 Customer Churn & LTV Analytics Toolkit

An end-to-end solution for data analysts and data scientists to analyze customer churn and lifetime value using SQL-based feature engineering, Python integration, and an interactive Streamlit dashboard.

**🔗 GitHub Repo**: [customer-churn-ltv-analytics-toolkit](https://github.com/Prudhvirajrekula/customer-churn-ltv-analytics-toolkit)

---

## 📦 Project Structure

```
.
├── datasets/
│   ├── csv-files/                # Source CSV data
│   └── DataWarehouseAnalytics.bak
├── scripts/
│   ├── *.sql                     # Modular SQL scripts for feature generation
│   ├── import_gold_to_mysql.py  # Load data into MySQL from CSVs
│   └── python_integration.py    # SQL + Python: feature generation and EDA
├── app.py                        # Streamlit dashboard
├── features_customer_churn_ltv.csv
├── churn_ltv_boxplot.png
├── requirements.txt
└── README.md
```

---

## 🧠 Key Features

- 🔧 14+ modular SQL scripts (exploration, segmentation, performance, etc.)
- 📥 CSV-to-MySQL importer with automatic table creation
- 🧮 Feature engineering for:
  - Churn flag
  - Lifetime Value (LTV)
  - Recency and order frequency
- 📈 Enhanced Streamlit dashboard with:
  - Filters
  - Hover tooltips
  - Interactive visuals

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Import CSVs to MySQL
```bash
python scripts/import_gold_to_mysql.py
```

### 3. Run Python Feature Generator
```bash
python scripts/python_integration.py
```

### 4. Launch Interactive Dashboard
```bash
streamlit run app.py
```

---

## 📊 Visuals

![LTV Boxplot](churn_ltv_boxplot.png)

---

## 👨‍💻 Created By

**Prudhvi Raj Rekula**  
Built with ❤️ using SQL, Python, and Streamlit  
[GitHub Profile](https://github.com/Prudhvirajrekula)