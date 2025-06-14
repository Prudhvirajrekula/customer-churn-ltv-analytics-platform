import os
import pandas as pd
import mysql.connector

# === Directory containing all CSVs ===
csv_dir = "datasets/csv-files"

# === MySQL connection ===
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='0000',
    database='analytics_db'
)
cursor = conn.cursor()

# === Only import the "gold" ones based on filename pattern ===
csv_files = {
    "fact_sales": "gold.fact_sales.csv",
    "dim_customers": "gold.dim_customers.csv",
    "dim_products": "gold.dim_products.csv",
    "report_customers": "gold.report_customers.csv",
    "report_products": "gold.report_products.csv"
}

# === Import logic ===
for table, filename in csv_files.items():
    file_path = os.path.join(csv_dir, filename)

    print(f"📄 Reading {file_path}")
    df = pd.read_csv(file_path)
    df.columns = [col.strip().replace(" ", "_") for col in df.columns]

    # Create table with all TEXT columns for simplicity
    column_defs = ", ".join([f"`{col}` TEXT" for col in df.columns])
    cursor.execute(f"DROP TABLE IF EXISTS `{table}`")
    cursor.execute(f"CREATE TABLE `{table}` ({column_defs})")

    for row in df.itertuples(index=False, name=None):
        placeholders = ", ".join(["%s"] * len(row))
        sql = f"INSERT INTO `{table}` VALUES ({placeholders})"
        cursor.execute(sql, row)

    print(f"✅ Imported: {table}")

# Finalize
conn.commit()
cursor.close()
conn.close()
print("🎉 All gold-level CSVs imported successfully.")