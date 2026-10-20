import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

# ============================================================
# PIPELINE MODULE IMPORTS
# ============================================================

for subpath in [
    BASE_DIR,
    BASE_DIR / "zildio" / "Data_cleaning_model",
    BASE_DIR / "zildio" / "Forcasting" / "Code",
    BASE_DIR / "zildio" / "Risk_Mangment" / "Code",
]:
    if str(subpath) not in sys.path:
        sys.path.insert(0, str(subpath))

import Data_cleaning_main
import forcasting
import Risk_managment


# ============================================================
# CHECK WHICH RISK FILE PYTHON IS USING
# ============================================================

print("\nRisk Management module:")
print(Risk_managment.__file__)


# ============================================================
# LOAD DATA
# ============================================================

df_sales = pd.read_csv(
    BASE_DIR / "zildio" / "Data" / "sales_history.csv"
)

df_products = pd.read_csv(
    BASE_DIR / "zildio" / "Data" / "sku_master.csv"
)

df_inventory = pd.read_csv(
    BASE_DIR / "zildio" / "Data" / "inventory_snapshots.csv"
)


# ============================================================
# DATA CLEANING
# ============================================================

Cleaned_sales = Data_cleaning_main.clean_sales_data(
    df_sales,
    df_products
)


# ============================================================
# FORECASTING
# ============================================================

forecasted_sales = forcasting.forecast_sales(
    Cleaned_sales
)


# ============================================================
# DEBUG FORECAST
# ============================================================

print("\nForecast shape:")
print(forecasted_sales.shape)

print("\nForecast columns:")
print(forecasted_sales.columns)

print("\nForecast sample:")
print(forecasted_sales.head())


# ============================================================
# DEBUG INVENTORY
# ============================================================

print("\nInventory columns:")
print(df_inventory.columns)

print("\nInventory sample:")
print(df_inventory.head())


# ============================================================
# INVENTORY RISK & FINANCIAL IMPACT
# ============================================================

inventory_risk = Risk_managment.calculate_inventory_risk(
    df_inventory,
    forecasted_sales,
    df_products
)

# Also save SKU master to DB for dashboard and scoring service
from zildio.Data_base import Data_base
Data_base.save_data_to_db(df_products, "sku_master")


# ============================================================
# RESULTS & BUSINESS IMPACT
# ============================================================

print("\n========================================")
print("Cleaned Sales Data (Sample):")
print("========================================")
print(Cleaned_sales.head())


print("\n========================================")
print("Forecasted Sales Data (Sample with Intervals):")
print("========================================")
print(forecasted_sales.head())


print("\n========================================")
print("Inventory Risk & Decisioning Data (Sample):")
print("========================================")
print(inventory_risk.head(10))

print("\n========================================")
print("NorthBay Living - Portfolio Impact Summary:")
print("========================================")
total_sales_at_risk = inventory_risk["sales_at_risk"].sum() if "sales_at_risk" in inventory_risk.columns else 0
total_capital_locked = inventory_risk["capital_locked"].sum() if "capital_locked" in inventory_risk.columns else 0
print(f"Total Monitored SKUs   : {len(inventory_risk)}")
print(f"Total Sales at Risk(INR): Rs. {total_sales_at_risk:,.2f}")
print(f"Total Capital Locked(INR): Rs. {total_capital_locked:,.2f}")
if "Action" in inventory_risk.columns:
    print("\nAction Distribution:")
    print(inventory_risk["Action"].value_counts().to_string())
print("========================================\n")



