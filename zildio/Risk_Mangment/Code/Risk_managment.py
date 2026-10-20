import pandas as pd
from sqlalchemy import create_engine, text
import sys
import logging

from zildio.Data_base import Data_base

logger = logging.getLogger(__name__)

def calculate_inventory_risk(df_inventory, df_forecast, df_sku_master=None):
    """
    Calculate inventory risk metrics for each SKU.

    Parameters
    ----------
    df_inventory : pd.DataFrame
        Inventory data with columns: sku_id, date, on_hand_units, on_order_units,
        lead_time_days, reorder_point.
    df_forecast : pd.DataFrame
        Forecast data with columns: sku_id, ds, yhat.
    df_sku_master : pd.DataFrame or None, optional
        SKU master data containing unit_cost and list_price keyed on sku_id.
        If None, unit_cost and list_price will default to 0.

    Returns
    -------
    pd.DataFrame
        Risk DataFrame with columns: sku_id, Total_stock, lead_time_days,
        reorder_point, Forecast_demand, Coverage, Coverage_Rating,
        Stockout_Risk, overstock_risk, Action, unit_cost, list_price,
        sales_at_risk, capital_locked.
    """
    df_inventory = df_inventory.copy()
    df_forecast = df_forecast.copy()

    logger.info("Risk management started")
    logger.info("Inventory columns: %s", df_inventory.columns.tolist())
    logger.info("Forecast columns: %s", df_forecast.columns.tolist())

    # ------------------------------------------------------------------
    # 1. Clean and aggregate inventory to the latest record per SKU
    # ------------------------------------------------------------------
    df_inventory["date"] = pd.to_datetime(df_inventory["date"], errors="coerce")
    df_inventory["sku_id"] = df_inventory["sku_id"].astype(str)
    df_inventory["on_hand_units"] = pd.to_numeric(df_inventory["on_hand_units"], errors="coerce").fillna(0)
    df_inventory["on_order_units"] = pd.to_numeric(df_inventory["on_order_units"], errors="coerce").fillna(0)
    df_inventory["lead_time_days"] = pd.to_numeric(df_inventory["lead_time_days"], errors="coerce").fillna(0)
    df_inventory["reorder_point"] = pd.to_numeric(df_inventory["reorder_point"], errors="coerce").fillna(0)

    df_inventory["Total_stock"] = df_inventory["on_hand_units"] + df_inventory["on_order_units"]
    df_inventory = (
        df_inventory.sort_values(["sku_id", "date"])
        .groupby("sku_id", as_index=False)
        .last()
    )
    df_inventory = df_inventory[["sku_id", "date", "Total_stock", "lead_time_days", "reorder_point"]]

    # ------------------------------------------------------------------
    # 2. Clean and aggregate forecast demand per SKU
    # ------------------------------------------------------------------
    df_forecast["ds"] = pd.to_datetime(df_forecast["ds"], errors="coerce")
    df_forecast["sku_id"] = df_forecast["sku_id"].astype(str)
    df_forecast["yhat"] = pd.to_numeric(df_forecast["yhat"], errors="coerce")
    df_forecast = df_forecast.dropna(subset=["sku_id", "ds", "yhat"])

    forecast_demand = (
        df_forecast.groupby("sku_id")["yhat"]
        .sum()
        .reset_index()
        .rename(columns={"yhat": "Forecast_demand"})
    )

    # ------------------------------------------------------------------
    # 3. Merge inventory with forecast and compute coverage
    # ------------------------------------------------------------------
    coverage_ratio = pd.merge(df_inventory, forecast_demand, on="sku_id", how="left")
    coverage_ratio["Forecast_demand"] = coverage_ratio["Forecast_demand"].fillna(0)

    coverage_ratio["Coverage"] = (
        coverage_ratio["Total_stock"]
        / coverage_ratio["Forecast_demand"].replace(0, pd.NA)
    )

    # ------------------------------------------------------------------
    # 4. Lead-time scoring (1 = best, 5 = worst)
    # ------------------------------------------------------------------
    def score_lead_time(lead_time):
        if lead_time <= 7:
            return 1
        elif lead_time <= 14:
            return 2
        elif lead_time <= 21:
            return 3
        elif lead_time <= 28:
            return 4
        else:
            return 5

    coverage_ratio["Lead_Time_Points"] = coverage_ratio["lead_time_days"].apply(score_lead_time)

    # ------------------------------------------------------------------
    # 5. Coverage scoring (1 = best, 4 = worst)
    # ------------------------------------------------------------------
    def classify_coverage(coverage):
        if pd.isna(coverage):
            return 4
        elif coverage < 0.5:
            return 4
        elif coverage < 1:
            return 3
        elif coverage < 2:
            return 2
        else:
            return 1

    coverage_ratio["Coverage_Rating"] = coverage_ratio["Coverage"].apply(classify_coverage)

    # ------------------------------------------------------------------
    # 6. Stockout risk (0–1 scale)
    # ------------------------------------------------------------------
    coverage_ratio["Total_Points"] = coverage_ratio["Lead_Time_Points"] + coverage_ratio["Coverage_Rating"]
    coverage_ratio["Stockout_Risk"] = coverage_ratio["Total_Points"] / 9

    # ------------------------------------------------------------------
    # 7. Overstock risk (0–1 scale)
    # ------------------------------------------------------------------
    coverage_ratio["Excess_Ratio"] = (
        (coverage_ratio["Total_stock"] - coverage_ratio["reorder_point"])
        / coverage_ratio["reorder_point"].replace(0, pd.NA)
    )

    def classify_overstock(excess_ratio):
        if pd.isna(excess_ratio):
            return 1
        elif excess_ratio < 0.5:
            return 1
        elif excess_ratio < 1:
            return 2
        elif excess_ratio < 2:
            return 3
        elif excess_ratio < 4:
            return 4
        else:
            return 5

    coverage_ratio["Overstock_Points"] = coverage_ratio["Excess_Ratio"].apply(classify_overstock)
    coverage_ratio["overstock_risk"] = coverage_ratio["Overstock_Points"] / 5

    # ------------------------------------------------------------------
    # 8. Action classification (stockout takes priority over overstock)
    # ------------------------------------------------------------------
    def action_from_risk(row):
        stockout = row["Stockout_Risk"]
        overstock = row["overstock_risk"]
        if stockout >= 0.85:
            return "Order ASAP"
        elif stockout >= 0.60:
            return "Monitor"
        elif overstock >= 0.60:
            return "Overstock"
        else:
            return "Fine"

    coverage_ratio["Action"] = coverage_ratio.apply(action_from_risk, axis=1)

    # ------------------------------------------------------------------
    # 9. Merge SKU master for unit_cost and list_price
    # ------------------------------------------------------------------
    if df_sku_master is not None:
        sku_master = df_sku_master.copy()
        sku_master["sku_id"] = sku_master["sku_id"].astype(str)
        # Keep only the columns we need to avoid column conflicts
        sku_cols = ["sku_id", "unit_cost", "list_price"]
        sku_master = sku_master[[c for c in sku_cols if c in sku_master.columns]]
        coverage_ratio = pd.merge(coverage_ratio, sku_master, on="sku_id", how="left")
        # Fill any unmatched SKUs with 0
        coverage_ratio["unit_cost"] = pd.to_numeric(coverage_ratio.get("unit_cost", 0), errors="coerce").fillna(0)
        coverage_ratio["list_price"] = pd.to_numeric(coverage_ratio.get("list_price", 0), errors="coerce").fillna(0)
    else:
        coverage_ratio["unit_cost"] = 0
        coverage_ratio["list_price"] = 0

    # ------------------------------------------------------------------
    # 10. Financial risk metrics
    # ------------------------------------------------------------------
    # sales_at_risk: revenue at risk when stockout is likely
    coverage_ratio["sales_at_risk"] = coverage_ratio.apply(
        lambda row: row["Forecast_demand"] * row["list_price"]
        if row["Stockout_Risk"] > 0.55
        else 0,
        axis=1,
    )

    # capital_locked: idle capital tied up in excess inventory
    coverage_ratio["capital_locked"] = coverage_ratio.apply(
        lambda row: max(0, row["Total_stock"] - row["Forecast_demand"]) * row["unit_cost"]
        if row["overstock_risk"] > 0.4
        else 0,
        axis=1,
    )

    # ------------------------------------------------------------------
    # 11. Select final columns and persist to DB
    # ------------------------------------------------------------------
    final_columns = [
        "sku_id",
        "Total_stock",
        "lead_time_days",
        "reorder_point",
        "Forecast_demand",
        "Coverage",
        "Coverage_Rating",
        "Stockout_Risk",
        "overstock_risk",
        "Action",
        "unit_cost",
        "list_price",
        "sales_at_risk",
        "capital_locked",
    ]

    result = coverage_ratio[final_columns]

    Data_base.save_data_to_db(result, "inventory_risk")
    logger.info("Risk data saved successfully!")

    return result