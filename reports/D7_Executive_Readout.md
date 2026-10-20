# EXECUTIVE READOUT: PROJECT FORESIGHT (D7)
## Demand & Inventory Intelligence Platform Handover
**Client:** NorthBay Living  
**Audience:** Head of Operations & Finance Leadership  
**Engagement:** Zidio 4-Week Data Science Client Engagement  
**Date:** October 16, 2026  

---

### SLIDE 1: Executive Title & Context
# Project FORESIGHT: From Guesswork to Intelligence
### Transforming NorthBay Living's Inventory Operations from Spreadsheet Intuition to Data-Driven Precision

* **The Problem:** NorthBay Living was losing money on two fronts simultaneously: high-demand best-sellers were stocking out (lost sales), while slow-moving catalog items piled up (cash locked in warehouse stock).
* **The Solution:** An end-to-end forecasting and inventory risk decisioning platform that monitors ~200 active SKUs, predicts monthly customer demand, flags stockout and overstock risks, and tells the planning team exactly what to order, clear, or monitor.
* **The Bottom Line:** Quantified financial impact across the entire portfolio, delivered through an interactive Operations Dashboard and a programmatic Scoring API.

---

### SLIDE 2: Headline Financial Impact & Executive Summary
# Headline Impact: ₹9.8+ Lakhs in Quantified Business Value

| Metric | Business Reality | Value at Stake | Recommended Action |
|---|---|---|---|
| **Sales at Risk (Stockout)** | 23 critical SKUs projected to stock out within supplier lead times | **₹4.85 Lakhs** | Immediate replenishment purchase orders (Priority 1) |
| **Capital Locked (Overstock)** | 48 SKUs holding stock > 6 months of forward customer demand | **₹4.98 Lakhs** | Tiered promotional clearance & markdown bundles |
| **Healthy Catalog Velocity** | 129 SKUs operating within balanced buffer thresholds | **₹9.40 Lakhs** | Maintain current replenishment schedules |
| **Forecast Accuracy Uplift** | Prophet machine learning model vs. Seasonal-Naive baseline | **+18.4% WAPE Improvement** | Adopt ML demand forecasts over spreadsheet moving averages |

> **Key Takeaway for Leadership:** Addressing just the top 10 critical stockout SKUs protects approximately **₹2.6 Lakhs in gross margin** this quarter, while releasing over **₹1.8 Lakhs in trapped cash** from dormant overstocked products.

---

### SLIDE 3: The 4-Quadrant Decisioning Matrix
# Inventory Triage: 200 SKUs at a Glance

Every SKU is scored on two independent risk axes (Stockout Risk vs. Overstock Risk) and assigned to one of four action quadrants:

```
               STOCKOUT RISK ↑ (Lead Time & Demand Coverage)
                 │
  [🔴 REORDER NOW]                │    [🟠 WATCH / VOLATILE]
  High Stockout · Low Overstock   │    High Stockout · High Overstock
  23 SKUs (₹4.85L at risk)        │    8 SKUs
  Action: Immediate PO to vendor  │    Action: Demand anomaly / review lead times
──────────────────────────────────┼──────────────────────────────────
  [🟢 HEALTHY]                    │    [🟡 MARKDOWN / CLEAR]
  Low Stockout · Low Overstock    │    Low Stockout · High Overstock
  121 SKUs                        │    48 SKUs (₹4.98L locked)
  Action: Stock levels balanced   │    Action: Promotional discounts & marketing
                 │
                 └──────────────────────────────→ OVERSTOCK RISK
```

* **Immediate Operations Action:** The merchandising team can filter directly for **🔴 Order ASAP** to generate weekly vendor orders without manually auditing 200 individual spreadsheets.

---

### SLIDE 4: Top 5 Replenishment Priorities (Immediate Action)
# Urgent Purchase Orders: What to Order Today

These five products carry the highest immediate sales-at-risk exposure due to low inventory coverage combined with extended supplier replenishment lead times:

| SKU ID | Category | Current Stock | 3-Month Forecast | Supplier Lead Time | Days of Coverage | Sales at Risk (₹) |
|---|---|---|---|---|---|---|
| **SKU0041** | Bedding | 18 units | 142 units | 14 Days | ~11 Days | **₹48,200** |
| **SKU0140** | Furniture | 12 units | 86 units | 28 Days | ~13 Days | **₹44,650** |
| **SKU0158** | Furniture | 9 units | 74 units | 21 Days | ~11 Days | **₹41,200** |
| **SKU0068** | Bath | 24 units | 118 units | 14 Days | ~18 Days | **₹37,800** |
| **SKU0131** | Bedding | 20 units | 96 units | 14 Days | ~19 Days | **₹32,150** |

* **Total Capital Required for Top 5 Reorder:** Approximately ₹1.2 Lakhs.
* **Protected Revenue:** Over ₹2.0 Lakhs in high-margin sales.

---

### SLIDE 5: Top 5 Markdown & Clearance Targets
# Cash Unfreezing: Surplus Inventory Clearance

These five SKUs have accumulated excessive stock relative to projected customer demand, locking up valuable warehouse footprint and capital:

| SKU ID | Category | Stock on Hand | 3-Month Forecast | Months of Supply | Capital Locked (₹) | Recommended Action |
|---|---|---|---|---|---|---|
| **SKU0094** | Decor | 485 units | 38 units | 38.3 Mo | **₹38,400** | 30% Flash Markdown |
| **SKU0199** | Decor | 465 units | 42 units | 33.2 Mo | **₹35,100** | Cross-sell bundle with Furniture |
| **SKU0052** | Kitchen | 390 units | 51 units | 22.9 Mo | **₹29,800** | Seasonal promotion banner |
| **SKU0181** | Decor | 340 units | 36 units | 28.3 Mo | **₹26,400** | Volume discount (Buy 2, Get 1) |
| **SKU0117** | Kitchen | 280 units | 40 units | 21.0 Mo | **₹22,100** | B2B / wholesale batch liquidation |

* **Action Plan:** Target 60-day liquidation to recapture ~₹1.5 Lakhs in liquid working capital.

---

### SLIDE 6: Forecasting Methodology & Honest Model Validation
# Model Rigor: Earning the Right to Predict

In line with consulting standards, FORESIGHT avoids "black-box" claims and subjects all models to rigorous evaluation:

### Validation Design
* **Rolling-Origin Backtest:** Evaluated against a withheld 3-month holdout per SKU across the catalog. No future data leaked into historical training.
* **The Bar to Clear:** **Seasonal-Naive Baseline** (predicting demand equal to the same month last year, or preceding period average).

### Headline Benchmark Results
* **Prophet Model WAPE:** **24.2%** (Weighted Absolute Percentage Error across volume)
* **Seasonal-Naive WAPE:** **42.6%**
* **Model Uplift:** **+18.4 percentage points improvement** over naive guessing
* **Forecast Bias:** **-2.8%** (slight conservative under-forecast, minimizing unnecessary overstocking risks)
* **Prediction Intervals:** 80% confidence bands generated for every SKU, providing operations with optimistic and conservative boundary planning.

---

### SLIDE 7: Deliverable Architecture & Tooling Ecosystem
# What NorthBay Living Receives

```
                                  [NorthBay Extracts]
                     sales_daily · sku_master · calendar · inventory
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                REPRODUCIBLE PIPELINE                                  │
│                 Data Cleaning (D1) ──► Prophet Models (D3) ──► Risk Engine (D4)        │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
        [PLANNING DASHBOARD (D5)]                     [SCORING SERVICE API (D6)]
     Streamlit Interactive Platform                    FastAPI Microservice Engine
    • Executive KPIs & ₹ Impact Cards                • GET /forecast/{sku_id}
    • Demand Forecast & 80% Intervals                • POST /batch scoring endpoint
    • Interactive Risk Matrix (What-If)              • GET /metrics accuracy audit
    • Priority Reorder & Markdown Lists              • High availability & error resilience
```

---

### SLIDE 8: Operational Recommendations & 90-Day Roadmap
# The Path Forward: Scaling Demand Intelligence

### 🗓️ Weeks 1–4 (Immediate Adoption)
1. **Weekly Ops Cadence:** Hold weekly 30-minute replenishment review using the **Priority Reorder List** in the FORESIGHT Dashboard.
2. **Clearance Launch:** Initiate flash promotional markdowns on the top 15 overstock Decor SKUs.
3. **API Integration:** Connect internal ERP/spreadsheet tools to the `/forecast/{sku_id}` FastAPI scoring endpoint.

### 🗓️ Months 2–3 (Process Institutionalization)
1. **Dynamic Lead Time Tracking:** Update `lead_time_days` monthly based on actual vendor delivery receipts.
2. **Promotional Calendar Sync:** Feed marketing event schedules from `calendar.csv` directly into forecast adjustments.
3. **Model Drift Monitoring:** Automate monthly WAPE reporting against newly observed actuals.

---

### SLIDE 9: Governance, Risks & Mitigation
# Protecting Decision Integrity

* **Risk 1: Sparse Sales for Newly Launched SKUs**  
  *Mitigation:* Products with < 6 months history automatically fall back to category average velocity with low-confidence visual badges.
* **Risk 2: Supplier Disruption & Extended Shipping Delays**  
  *Mitigation:* The built-in **What-If Simulation** slider enables the Operations Head to model lead time shocks (+7 to +21 days) to preview buffer vulnerabilities instantly.
* **Risk 3: Model Degradation Over Time**  
  *Mitigation:* Transparent WAPE metrics exposed on the dashboard ensure accuracy drops are visible immediately.

---

**Prepared by:** Project FORESIGHT Engagement Team · Zidio Internship  
**Status:** Complete & Ready for Operations Handover  
**System Access:** Dashboard: `http://localhost:8501` · API Service: `http://localhost:8000/docs`
