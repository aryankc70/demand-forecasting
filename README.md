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
| 1 month | _fill in_ | _fill in_ | |
| 3 months | _fill in_ | _fill in_ | |
| 6 months | _fill in_ | _fill in_ | |

_Discuss: SARIMA tends to win on strongly seasonal, stable series; boosted trees win when exogenous features (promos, price) matter._
