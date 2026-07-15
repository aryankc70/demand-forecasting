# Demand Forecasting: SARIMA vs Gradient Boosting

Compares classical time-series (SARIMA) against a feature-based XGBoost model for retail demand forecasting, evaluating MAPE across 1/3/6-month horizons.

## Dataset
Works out of the box with a generated seasonal series; swap in real data (e.g. [Store Sales — Kaggle](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)) by editing `load_data()`.

## Quickstart
```bash
pip install -r requirements.txt
python forecast.py
```

## Results

| Horizon | SARIMA MAPE | XGBoost MAPE | Winner |
|---|---|---|---|
| 1 month | 5.56% | 4.21% | XGBoost |
| 3 months | 4.90% | 3.67% | XGBoost |
| 6 months | 3.49% | 4.84% | SARIMA |

XGBoost wins short horizons: its lag features (t-1, t-2, t-3) carry strong
signal one to three steps ahead. SARIMA wins at 6 months: it explicitly
models the seasonal cycle, so it extrapolates the yearly pattern more
reliably as lag information decays. Practical takeaway — horizon should
drive model choice, and a production system might use XGBoost for
short-term operational forecasts and SARIMA for longer-range planning.

**Lag experiment:** adding a 6-month lag feature improved 1mo MAPE
(4.21% → 3.90%) but *worsened* the 6mo horizon (4.84% → 5.36%) — the
opposite of the intent. With only ~80 training rows, the extra feature
likely adds noise rather than signal at long range. Kept the change
documented as a negative result.
