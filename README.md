# FORESIGHT — Demand & Inventory Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![Prophet](https://img.shields.io/badge/Prophet-1.1+-blueviolet.svg)](https://facebook.github.io/prophet/)
[![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL%20%2F%20Neon-336791.svg)](https://neon.tech/)

> **Client:** NorthBay Living (D2C Home & Lifestyle Brand)  
> **Engagement:** Zidio Development · Data Science & Analytics Track · Project FORESIGHT  
> **Deliver it like a consultant. Defend it like a scientist.**

---

## 📌 Executive Summary & Problem Statement

NorthBay Living is a fast-growing direct-to-consumer home & lifestyle brand managing ~200 active SKUs across Furniture, Bedding, Bath, Kitchen, and Decor. Previously, inventory planning relied on gut feel and static spreadsheets, resulting in two simultaneous money-losing challenges:
1. **Stockouts on Best-Sellers:** High-velocity products ran out of stock during critical customer purchase windows, permanently losing revenue.
2. **Overstock on Slow Movers:** Cash was trapped in dormant inventory that eventually required aggressive, margin-eroding markdowns.

**Project FORESIGHT** provides an AI-powered demand forecasting and inventory intelligence system that ingests raw operational extracts, predicts weekly SKU-level demand with uncertainty intervals, scores stockout/overstock risks, calculates the rupee value at stake, and presents clear stocking decisions through an interactive planning dashboard and scoring microservice.

---

## 📸 Platform Demo & Screenshots

### 1. Executive Action Center & Portfolio Health Overview
![Executive Action Center](screenshots/02_action_center_overview.png)

### 2. AI Demand Forecasting with Prophet (80% Confidence Intervals)
![Demand Forecast](screenshots/03_demand_forecast.png)

### 3. Inventory Risk Matrix & Interactive What-If Stress Testing
![Inventory Risk Matrix](screenshots/04_inventory_risk_matrix.png)

### 4. Interactive Data Onboarding Wizard
![Data Onboarding Wizard](screenshots/01_landing_onboarding.png)

### 5. FastAPI Scoring Microservice (Interactive Swagger UI)
![FastAPI Scoring Docs](screenshots/05_fastapi_scoring_docs.png)

---

## 📊 Deliverables (D1 – D7) Mapping

| Deliverable | Description | Location / Status |
|---|---|---|
| **D1: Data Pipeline** | Ingestion & cleaning of all 4 extracts into an analysis-ready database | [`zildio/Data_cleaning_model/`](zildio/Data_cleaning_model/Data_cleaning_main.py) |
| **D2: EDA Memo** | Data quality findings, Pareto revenue concentration, top movers, dead stock | [`reports/D2_EDA_and_Data_Quality_Memo.md`](reports/D2_EDA_and_Data_Quality_Memo.md) |
| **D3: Forecast Model** | Prophet SKU-level monthly forecasts + 80% intervals, evaluated vs Seasonal-Naive | [`zildio/Forcasting/Code/`](zildio/Forcasting/Code/forcasting.py) |
| **D4: Risk Scoring** | Stockout / overstock 4-quadrant decisioning with ₹ value at stake | [`zildio/Risk_Mangment/Code/`](zildio/Risk_Mangment/Code/Risk_managment.py) |
| **D5: Planning Dashboard**| Streamlit multi-view app with What-If simulations & priority lists | [`app.py`](app.py) |
| **D6: Scoring Service** | FastAPI microservice for single-SKU and batch scoring | [`service/api.py`](service/api.py) |
| **D7: Executive Readout** | 9-slide executive briefing deck & handover memorandum | [`reports/D7_Executive_Readout.md`](reports/D7_Executive_Readout.md) |

---

## 🏗️ Solution Architecture

```
Raw Client Extracts
├── sales_history.csv       (16,240 rows: sale_date, sku_id, quantity, unit_price, discount_pct, revenue)
├── sku_master.csv          (200 rows: sku_id, category, subcategory, unit_cost, list_price)
├── calendar.csv            (calendar attributes: week, month, season, is_holiday, promo_event)
└── inventory_snapshots.csv (on_hand_units, on_order_units, lead_time_days, reorder_point)
      │
      ▼
[1. Data Cleaning & Standardization]
  - Date harmonization, deduplication, price/discount sanity checks, ghost-SKU filtering
      │
      ▼
[2. Central PostgreSQL / Neon Storage] ◄─── Persistent Star Schema Tables
      │
      ├─────────────────────────────────────────┐
      ▼                                         ▼
[3. Facebook Prophet ML Engine]       [4. Inventory Risk Scoring Layer]
  - Log-transformed SKU time-series     - Total stock = On Hand + On Order
  - 80% prediction intervals            - Coverage ratio = Stock / Forecast Demand
  - Seasonal-naive baseline benchmark   - Lead time penalty & stockout scoring
  - Rolling-origin WAPE evaluation      - Excess ratio & overstock scoring
      │                                 - Financial valuation: ₹ Sales at Risk & ₹ Locked
      │                                         │
      └────────────────────┬────────────────────┘
                           ▼
[5. Stakeholder Consumption Layer]
  ├── Planning Dashboard (Streamlit): Overview · EDA · Forecast · Risk · SKU Intel · Model Eval
  ├── Deployed Scoring API (FastAPI): /forecast/{sku_id} · /batch · /metrics · /skus
  └── Business Intelligence: Power BI Dashboard (Dashboard.pbix)
```

---

## 📈 Model Validation: Backtest Results vs Baseline

FORESIGHT enforces a strict rolling-origin cross-validation discipline:
* **Evaluation Holdout:** The last 3 months of history are held out for all SKUs with ≥9 months history.
* **Baseline Bar:** Seasonal-Naive predictor (demand equal to the same month 1 year prior, with mean fallback).
* **Primary Metric:** **WAPE** (Weighted Absolute Percentage Error), which weights errors by volume to remain robust on low-velocity SKUs where MAPE explodes.

### Headline Benchmark Metrics
| Model | WAPE (Lower is Better) | Forecast Bias | Status |
|---|---|---|---|
| **Seasonal-Naive Baseline** | **42.6%** | -6.4% | Benchmark |
| **Prophet Machine Learning** | **24.2%** | **-2.8%** | **Winner (+18.4% improvement)** |

> **Finding:** Prophet demonstrably beats the seasonal-naive baseline across the portfolio. The slight negative bias (-2.8%) is an asset in retail inventory planning, preventing artificial forecast inflation that causes runaway overstock.

---

## 💰 Quantified Business Impact

Portfolio analysis of NorthBay Living's 200 catalog SKUs reveals:

* **💸 Sales at Risk (Stockout Exposure):** **₹4.85 Lakhs**
  * 23 critical SKUs projected to breach safety stock during vendor lead times.
  * Reordering the top 5 urgent SKUs protects **₹2.0+ Lakhs** in high-margin revenue.
* **🔒 Capital Locked (Overstock Surplus):** **₹4.98 Lakhs**
  * 48 SKUs holding more than 6 months of forward projected customer demand.
  * Targeted 60-day promotional markdown campaigns can release **₹1.5+ Lakhs** in liquid cash.

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites & Environment
Ensure Python 3.12 or 3.13 is installed.

```powershell
# Clone or navigate to the project directory
cd "FORESIGHT-AI-Powered-Demand-Inventory-Intelligence-Platform-main"

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file in the project root (see `.env.example`) and supply your PostgreSQL / Neon connection string:
```env
DATABASE_URL=postgresql+psycopg://username:password@ep-your-database-pooler.region.aws.neon.tech/neondb?sslmode=require
```

### 3. Run the Full Analytics Pipeline (CLI)
Executes data cleaning, Prophet SKU forecasting with prediction intervals, baseline WAPE evaluation, and inventory risk scoring:
```powershell
python main.py
```

### 4. Launch the Interactive Streamlit Dashboard
```powershell
python -m streamlit run app.py
```
Open your browser at **[http://localhost:8501](http://localhost:8501)**.

### 5. Launch the FastAPI Scoring Service
```powershell
uvicorn service.api:app --host 0.0.0.0 --port 8000
```
* **API Documentation (Swagger UI):** [http://localhost:8000/docs](http://localhost:8000/docs)
* **Single SKU Score:** `GET http://localhost:8000/forecast/SKU0001`
* **Model Accuracy Metrics:** `GET http://localhost:8000/metrics`

---

## 🎯 Key Assumptions & Limitations

1. **Lead Time Reliability:** Vendor lead times provided in `inventory_snapshots.csv` are assumed to be stable. Supply chain delays can be tested using the dashboard's **What-If Simulation** slider.
2. **Minimum History Threshold:** Prophet requires at least 6 months of historical observations to train reliably; SKUs with <6 months are flagged for manual review or category-level velocity modeling.
3. **Unconstrained Demand:** Historical sales are treated as a proxy for customer demand. True unconstrained demand may have been higher during periods when stockouts occurred.
4. **Scope Exclusions:** Live direct-write integrations to warehouse ERPs and automated PO placement are explicitly excluded per engagement scope.
