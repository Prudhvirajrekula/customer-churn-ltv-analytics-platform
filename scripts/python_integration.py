
import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt
import seaborn as sns

# MySQL connection details
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='0000',
    database='analytics_db'
)

# Feature query example: customer churn and LTV
query = """
SELECT 
    customer_key AS customer_id,
    MAX(order_date) AS last_order_date,
    COUNT(order_number) AS order_count,
    SUM(sales_amount) AS lifetime_value,
    DATEDIFF(CURDATE(), MAX(order_date)) AS days_since_last_order,
    CASE 
        WHEN DATEDIFF(CURDATE(), MAX(order_date)) > 90 THEN 1
        ELSE 0
    END AS is_churned
FROM fact_sales
GROUP BY customer_key
"""


# Load data
df = pd.read_sql(query, conn)

# Save results
df.to_csv("features_customer_churn_ltv.csv", index=False)

# Plotting churn vs LTV
plt.figure(figsize=(10,6))
sns.boxplot(x="is_churned", y="lifetime_value", data=df)
plt.title("Lifetime Value by Churn Status")
plt.xlabel("Churned")
plt.ylabel("LTV")
plt.tight_layout()
plt.savefig("churn_ltv_boxplot.png")
plt.show()

conn.close()
