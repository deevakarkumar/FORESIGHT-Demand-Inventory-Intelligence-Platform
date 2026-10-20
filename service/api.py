"""
FORESIGHT — Scoring Service (D6)
=================================
FastAPI endpoint that returns demand forecast and inventory risk
for a given SKU or a batch of SKUs.

Usage:
    uvicorn service.api:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health                     – liveness check
    GET  /forecast/{sku_id}          – forecast + risk for one SKU
    POST /batch                      – forecast + risk for multiple SKUs
    GET  /skus                       – list all available SKU IDs
    GET  /metrics                    – model accuracy metrics (WAPE vs baseline)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the project root importable so we can reuse the DB helper
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv(dotenv_path=BASE_DIR / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL not set. Add it to your .env file in the project root."
    )

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="FORESIGHT Scoring Service",
    description=(
        "Returns SKU-level demand forecast and inventory risk scores for "
        "NorthBay Living's operations team. Backed by Facebook Prophet "
        "time-series models and a transparent risk-scoring layer."
    ),
    version="1.0.0",
    contact={"name": "FORESIGHT / Zidio", "email": "ops@northbayliving.example"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ForecastPoint(BaseModel):
    ds: str
    yhat: float
    yhat_lower: Optional[float] = None
    yhat_upper: Optional[float] = None
    is_future: bool


class SKURisk(BaseModel):
    sku_id: str
    total_stock: Optional[float]
    reorder_point: Optional[float]
    forecast_demand: Optional[float]
    coverage: Optional[float]
    stockout_risk: Optional[float]
    overstock_risk: Optional[float]
    action: Optional[str]
    sales_at_risk: Optional[float] = None
    capital_locked: Optional[float] = None


class SKUResponse(BaseModel):
    sku_id: str
    category: Optional[str] = None
    subcategory: Optional[str] = None
    forecast: List[ForecastPoint]
    risk: Optional[SKURisk]
    message: Optional[str] = None


class BatchRequest(BaseModel):
    sku_ids: List[str]


class BatchResponse(BaseModel):
    results: List[SKUResponse]
    skus_found: int
    skus_missing: int


class MetricsResponse(BaseModel):
    prophet_wape: Optional[float]
    baseline_wape: Optional[float]
    prophet_bias: Optional[float]
    baseline_bias: Optional[float]
    note: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_table(table: str) -> pd.DataFrame:
    """Load a full table from PostgreSQL, return empty DF on error."""
    try:
        return pd.read_sql(f'SELECT * FROM public."{table}"', engine)
    except Exception:
        return pd.DataFrame()


def _get_sku_forecast(sku_id: str, df_forecast: pd.DataFrame, last_actual_ds: pd.Timestamp) -> List[ForecastPoint]:
    """Return forecast rows for one SKU as a list of ForecastPoint objects."""
    sku_fc = df_forecast[df_forecast["sku_id"] == sku_id].copy()
    if sku_fc.empty:
        return []
    sku_fc["ds"] = pd.to_datetime(sku_fc["ds"], errors="coerce")
    sku_fc = sku_fc.sort_values("ds")
    points = []
    for _, row in sku_fc.iterrows():
        points.append(
            ForecastPoint(
                ds=str(row["ds"].date()),
                yhat=round(float(row.get("yhat", 0) or 0), 2),
                yhat_lower=round(float(row["yhat_lower"]), 2) if "yhat_lower" in row and pd.notna(row.get("yhat_lower")) else None,
                yhat_upper=round(float(row["yhat_upper"]), 2) if "yhat_upper" in row and pd.notna(row.get("yhat_upper")) else None,
                is_future=row["ds"] > last_actual_ds,
            )
        )
    return points


def _get_sku_risk(sku_id: str, df_risk: pd.DataFrame) -> Optional[SKURisk]:
    """Return risk row for one SKU, or None if not found."""
    row = df_risk[df_risk["sku_id"] == sku_id]
    if row.empty:
        return None
    r = row.iloc[0]

    def _safe_float(key: str):
        v = r.get(key)
        return round(float(v), 4) if pd.notna(v) else None

    return SKURisk(
        sku_id=sku_id,
        total_stock=_safe_float("Total_stock"),
        reorder_point=_safe_float("reorder_point"),
        forecast_demand=_safe_float("Forecast_demand"),
        coverage=_safe_float("Coverage"),
        stockout_risk=_safe_float("Stockout_Risk"),
        overstock_risk=_safe_float("overstock_risk"),
        action=str(r.get("Action", "")) or None,
        sales_at_risk=_safe_float("sales_at_risk") if "sales_at_risk" in r.index else None,
        capital_locked=_safe_float("capital_locked") if "capital_locked" in r.index else None,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health():
    """Liveness probe — returns OK if the service and DB are reachable."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok", "db": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"DB unreachable: {e}")


@app.get("/skus", tags=["Discovery"])
def list_skus(
    category: Optional[str] = Query(None, description="Filter by category name"),
):
    """
    List all SKU IDs available in the risk table.
    Optionally filter by category (requires sku_master table).
    """
    df_risk = _load_table("inventory_risk")
    if df_risk.empty:
        return {"skus": [], "count": 0, "note": "No data found. Run the pipeline first."}

    skus = sorted(df_risk["sku_id"].dropna().unique().tolist())

    if category:
        df_sku = _load_table("sku_master")
        if not df_sku.empty:
            cat_skus = df_sku[df_sku["category"].str.lower() == category.lower()]["sku_id"].tolist()
            skus = [s for s in skus if s in cat_skus]

    return {"skus": skus, "count": len(skus)}


@app.get("/metrics", tags=["Model"], response_model=MetricsResponse)
def get_metrics():
    """
    Return forecast model accuracy metrics.
    WAPE: Weighted Absolute Percentage Error (lower is better).
    Bias: systematic over/under-forecast (negative = under-forecast).
    """
    df_metrics = _load_table("forecast_metrics")
    if df_metrics.empty:
        return MetricsResponse(
            prophet_wape=None, baseline_wape=None,
            prophet_bias=None, baseline_bias=None,
            note="Metrics not yet computed. Run the pipeline with a dataset that has >= 9 months history."
        )

    def _get(metric: str, col: str):
        row = df_metrics[df_metrics["metric"] == metric]
        if row.empty:
            return None
        v = row.iloc[0].get(col)
        return round(float(v), 4) if pd.notna(v) else None

    return MetricsResponse(
        prophet_wape=_get("wape", "prophet_value"),
        baseline_wape=_get("wape", "baseline_value"),
        prophet_bias=_get("bias", "prophet_value"),
        baseline_bias=_get("bias", "baseline_value"),
        note="WAPE = sum(|actual - forecast|) / sum(actual). Bias = mean signed error / mean actual."
    )


@app.get("/forecast/{sku_id}", tags=["Scoring"], response_model=SKUResponse)
def get_sku(sku_id: str):
    """
    Return the full forecast and inventory risk for a single SKU.

    - **sku_id**: e.g. `SKU0001`
    - **forecast**: monthly demand forecast with 80% prediction intervals
    - **risk**: stockout risk, overstock risk, recommended action, and ₹ value at stake
    """
    df_forecast = _load_table("forecasted_sales")
    df_risk = _load_table("inventory_risk")
    df_sales = _load_table("cleaned_sales")
    df_sku = _load_table("sku_master")

    # Validate SKU exists
    all_skus = set()
    if not df_forecast.empty and "sku_id" in df_forecast.columns:
        all_skus |= set(df_forecast["sku_id"].unique())
    if not df_risk.empty and "sku_id" in df_risk.columns:
        all_skus |= set(df_risk["sku_id"].unique())

    if sku_id not in all_skus:
        raise HTTPException(
            status_code=404,
            detail=f"SKU '{sku_id}' not found. Use GET /skus to see available SKUs."
        )

    # Last actual sale date (for is_future flag)
    last_actual_ds = pd.Timestamp("1900-01-01")
    if not df_sales.empty and "sale_date" in df_sales.columns:
        df_sales_sku = df_sales[df_sales["sku_id"] == sku_id]
        if not df_sales_sku.empty:
            last_actual_ds = pd.to_datetime(df_sales_sku["sale_date"], errors="coerce").max()

    # Category info
    category = subcategory = None
    if not df_sku.empty and "sku_id" in df_sku.columns:
        sku_row = df_sku[df_sku["sku_id"] == sku_id]
        if not sku_row.empty:
            category = sku_row.iloc[0].get("category")
            subcategory = sku_row.iloc[0].get("subcategory")

    forecast_points = _get_sku_forecast(sku_id, df_forecast, last_actual_ds)
    risk = _get_sku_risk(sku_id, df_risk)

    message = None
    if not forecast_points:
        message = "No forecast available for this SKU. It may have fewer than 6 months of sales history."

    return SKUResponse(
        sku_id=sku_id,
        category=str(category) if category else None,
        subcategory=str(subcategory) if subcategory else None,
        forecast=forecast_points,
        risk=risk,
        message=message,
    )


@app.post("/batch", tags=["Scoring"], response_model=BatchResponse)
def get_batch(request: BatchRequest):
    """
    Return forecast and risk for a batch of SKUs in one call.

    Send up to 200 SKU IDs at a time.

    Example request body:
    ```json
    {"sku_ids": ["SKU0001", "SKU0002", "SKU0003"]}
    ```
    """
    if len(request.sku_ids) > 200:
        raise HTTPException(
            status_code=400,
            detail="Batch size too large. Maximum 200 SKUs per request."
        )

    df_forecast = _load_table("forecasted_sales")
    df_risk = _load_table("inventory_risk")
    df_sales = _load_table("cleaned_sales")
    df_sku = _load_table("sku_master")

    all_skus = set()
    if not df_forecast.empty and "sku_id" in df_forecast.columns:
        all_skus |= set(df_forecast["sku_id"].unique())
    if not df_risk.empty and "sku_id" in df_risk.columns:
        all_skus |= set(df_risk["sku_id"].unique())

    last_actual_ds = pd.Timestamp("1900-01-01")
    if not df_sales.empty and "sale_date" in df_sales.columns:
        last_actual_ds = pd.to_datetime(df_sales["sale_date"], errors="coerce").max()

    results = []
    skus_missing = 0

    for sku_id in request.sku_ids:
        if sku_id not in all_skus:
            skus_missing += 1
            results.append(SKUResponse(
                sku_id=sku_id,
                forecast=[],
                risk=None,
                message=f"SKU '{sku_id}' not found in the database."
            ))
            continue

        category = subcategory = None
        if not df_sku.empty and "sku_id" in df_sku.columns:
            sku_row = df_sku[df_sku["sku_id"] == sku_id]
            if not sku_row.empty:
                category = sku_row.iloc[0].get("category")
                subcategory = sku_row.iloc[0].get("subcategory")

        forecast_points = _get_sku_forecast(sku_id, df_forecast, last_actual_ds)
        risk = _get_sku_risk(sku_id, df_risk)
        results.append(SKUResponse(
            sku_id=sku_id,
            category=str(category) if category else None,
            subcategory=str(subcategory) if subcategory else None,
            forecast=forecast_points,
            risk=risk,
        ))

    return BatchResponse(
        results=results,
        skus_found=len(request.sku_ids) - skus_missing,
        skus_missing=skus_missing,
    )
