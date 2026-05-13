# Product Sales Forecasting

End-to-end retail sales forecasting on **188,340 store-day observations** across **365 stores, 4 store types, and 5 location types** (Jan 2018 – May 2019). Final model: **XGBoost on engineered calendar features**, achieving **MAPE 20.90%, MAE ₹9,676, R² 0.55** on a 60-day chronological holdout, deployed as a Flask API.

> **Headline finding:** In this dataset, holidays correlate with *lower* sales, not higher. Treating the holiday flag as a positive predictor — the default assumption — would have produced confidently wrong forecasts every time a holiday landed in the prediction window.

**Links:** [Tableau Public](#) · [Technical Blog](#) · [Loom Demo](#)

---

## Problem statement

Retailers need accurate daily sales forecasts to manage inventory, plan staffing, schedule promotions, and decide where to expand. Underforecast and shelves go empty; overforecast and perishables get written off. The goal: predict daily `Sales` per store from store attributes, location, region, calendar position, and promotional activity.

**Target metric:** MAPE (Mean Absolute Percentage Error) on a chronological holdout. MAPE was chosen because it's what business stakeholders actually care about — "how far off, in percentage terms, are we?" — and because it's robust to the wide range of store sizes in the chain (a 10% miss on a ₹60K-revenue store and a 10% miss on a ₹20K-revenue store should count equally).

## Dataset

| Column | Description |
|---|---|
| `Store_id` | One of 365 stores |
| `Store_Type` | S1–S4 (4 categories) |
| `Location_Type` | L1–L5 (5 categories) |
| `Region_Code` | R1–R4 |
| `Date` | Daily, 2018-01-01 to 2019-05-31 |
| `Holiday` | Binary flag |
| `Discount` | Yes/No |
| `#Order` | Number of orders that day *(target-leaking — see below)* |
| `Sales` | **Target** — total daily sales (₹) |

- `TRAIN.csv` — 188,340 rows (training + chronological holdout)
- `TEST_FINAL.csv` — 22,265 rows (365 stores × 61 days), June 1 – July 31 2019 — forward-in-time, no `Sales` column

---

## How the project is organised

The work splits into four blocks, each in its own deliverable.

### Block 1 — Tableau dashboards
Five dashboards published as a single Tableau Public workbook. Built **before** modelling to force a thorough look at the data from angles a notebook wouldn't naturally push toward.

1. **Sales Performance** — KPI tiles, global filters, clean date axis
2. **Regional Sales Analysis** — multi-metric table by region and store type
3. **Promotional Impact Analysis** — calculated `Day Type` field (Holiday × Discount, 4 combinations). This is where the contrarian holiday finding first surfaced.
4. **Operational Insights** — orders-vs-sales scatter with per-store-type trend lines, day-of-week bars, dual-axis monthly chart, calendar heatmap
5. **Executive Summary** — one-pager with headline numbers and recommendations

### Block 2 — EDA + Hypothesis Testing
`Block2_EDA_Hypothesis_REAL.ipynb` — data quality audit, univariate and bivariate analysis, time-series exploration, and five formal hypothesis tests:

| # | Hypothesis | Test | Result |
|---|---|---|---|
| 1 | Discounts increase sales | Welch's t-test | Confirmed (large effect) |
| 2 | Holidays increase sales | Welch's t-test | **Rejected — holidays *decrease* sales** |
| 3 | Sales differ across store types | One-way ANOVA | Confirmed (S4 > S3 > S1 > S2) |
| 4 | Sales differ across regions | One-way ANOVA | Confirmed |
| 5 | Orders correlate with sales | Pearson | Strong positive (ρ ≈ 0.94) |

**Key decisions handed off to Block 3:**
- `log1p(Sales)` transformation — raw skew 1.25, post-transform −2.07 (manageable)
- Drop the 19 zero-sales rows
- **`#Order` excluded as a direct predictor — it's only known after the sales day ends.** Lagged versions only.
- Time-based train/val/test split. **Never** random — leaks future into past.

### Block 3 — ML Modelling
`Block3_ML_Modeling_tuned.ipynb` — seven models on a leaderboard:

| # | Model | Type | Tuned? |
|---|---|---|---|
| 0 | Naive (`Sales_lag_7`) | Baseline floor | — |
| 1 | Linear Regression | Regression baseline | — |
| 2 | Random Forest | Tree ensemble | ✅ GridSearchCV |
| 3 | XGBoost | Gradient boosting | ✅ GridSearchCV |
| 4 | LightGBM | Gradient boosting | ✅ GridSearchCV |
| 5 | Prophet | Time-series (per-store) | — |
| 6 | SARIMA | Time-series (chain-aggregated) | — |

Cross-validation used `TimeSeriesSplit(n_splits=5)`; final scoring on a 60-day chronological holdout. **XGBoost won** — for multi-store retail forecasting, pooled gradient boosting on calendar features beat per-series classical TS models, which only see one store's history at a time.

`Block3b_Test_Set_Predictions.ipynb` is a separate inference notebook that loads the saved artifact and generates predictions on `TEST_FINAL.csv` — validating that the artifact works in a fresh environment (the same way the Flask app uses it).

### Block 4 — Flask deployment
The best XGBoost model is serialised to `artifacts/sales_forecast_model.joblib` along with feature names, categorical vocabulary, and metrics. A Flask app loads it once at startup and exposes a `/predict` endpoint that accepts feature JSON and returns a forecast. Schema validation uses the saved training vocabulary, guaranteeing parity between training and inference.

---

## Final results

| Metric | Value |
|---|---|
| **MAPE** | **20.90%** |
| MAE | ₹9,676 |
| RMSE | (in notebook leaderboard) |
| R² | 0.549 |
| Holdout window | Last 60 days of TRAIN.csv (chronological) |
| Winning model | XGBoost (default hyperparameters beat tuned in this case) |

A MAPE of ~21% on daily store-level forecasts is a respectable result for a model that uses **only** calendar and store-attribute features — no weather, no local events, no marketing spend, no competitor data. The remaining error is dominated by store-day idiosyncrasies the features can't see.

---

## Insights & recommendations

1. **Don't assume holidays boost sales.** In this chain, holidays *reduce* daily sales relative to comparable non-holiday days. Operationally: don't over-staff or over-stock on holidays without store-level evidence; the chain-wide default is the opposite.
2. **Discounts work — but their interaction with holidays matters.** The four-way Day Type breakdown (Holiday × Discount) shows that holidays *without* a discount underperform regular days the most. If you're going to be open on a holiday, run a promotion.
3. **Sunday is the peak.** Weekly seasonality is large (~+23% Sunday over Friday). Staffing should reflect that.
4. **Store type matters more than location type.** S4 stores average ~₹60K/day vs S2 at ~₹27K. Strategic decisions about expansion should weight store format heavily.
5. **The `#Order` leakage trap is real.** A model that uses raw `#Order` as a feature gets near-perfect training scores and is *useless* at inference time — `TEST_FINAL.csv` correctly omits the column, exactly as a real deployed system would.

---

## Repository structure

```
.
├── README.md                              # this file
├── notebooks/
│   ├── Block2_EDA_Hypothesis_REAL.ipynb   # EDA + 5 hypothesis tests
│   ├── Block3_ML_Modeling_tuned.ipynb     # 7-model leaderboard + tuning
│   └── Block3b_Test_Set_Predictions.ipynb # Inference on TEST_FINAL.csv
├── deployment/
│   ├── app.py                             # Flask API
│   ├── requirements.txt
│   └── artifacts/
│       └── sales_forecast_model.joblib    # Trained model + metadata
├── data/
│   ├── TRAIN.csv
│   └── TEST_FINAL.csv
├── outputs/
│   ├── predictions.csv                    # ID, Sales (submission format)
│   └── predictions_full.csv               # Annotated with all features
└── tableau/
    └── SalesPerformanceDashboardMOHAN.twbx
```

---

## Running it locally

### Notebooks
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Run the notebooks in order: Block 2 → Block 3 → Block 3b. Block 3 saves the model artifact that Block 3b and the Flask app both consume.

### Flask API
```bash
cd deployment
pip install -r requirements.txt
python app.py
# Server runs on http://127.0.0.1:5000
```

Example request:
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Store_id": 1,
    "Store_Type": "S1",
    "Location_Type": "L3",
    "Region_Code": "R1",
    "Date": "2019-06-15",
    "Holiday": 0,
    "Discount": "Yes"
  }'
```

Response:
```json
{ "predicted_sales": 47821.34 }
```

---

## Tech stack

**Python** — pandas, NumPy, scikit-learn, XGBoost, LightGBM, Prophet, statsmodels (SARIMA), scipy.stats, matplotlib, seaborn, joblib
**Deployment** — Flask
**BI** — Tableau Public
**Environment** — Jupyter Lab, Python 3.11

---

## What I'd do next

- **Add external signals** — weather, local events, competitor promotions. The current model is calendar-only; the residuals almost certainly contain weather effects.
- **Per-store error analysis** — the 20.90% MAPE is an average. A handful of stores likely contribute disproportionately to the error and deserve dedicated models or feature engineering.
- **Prediction intervals** — point forecasts are useful; intervals are actionable. Quantile regression or conformal prediction would add this without retraining.
- **CI on the Flask app** — schema tests, a smoke test against the saved artifact, and a deploy step.

---

## Author

**Mohan** — Product Designer, Engineer, and Data Scientist.
Portfolio: [datascienceportfol.io/mohan](#) · Tableau: [mohan.dash](https://public.tableau.com/app/profile/mohan.dash)
