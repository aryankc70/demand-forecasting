"""SARIMA vs XGBoost for monthly demand forecasting."""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


def load_data() -> pd.Series:
    """Synthetic 8-year monthly series: trend + seasonality + noise.
    Replace with real data: return a pd.Series indexed by month."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2018-01-01", periods=96, freq="MS")
    trend = np.linspace(100, 180, 96)
    season = 25 * np.sin(2 * np.pi * idx.month / 12)
    return pd.Series(trend + season + rng.normal(0, 8, 96), index=idx, name="demand")


def make_features(s: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"y": s})
    for lag in (1, 2, 3, 6, 12):
        df[f"lag_{lag}"] = s.shift(lag)
    df["rolling_3"] = s.shift(1).rolling(3).mean()
    df["month"] = s.index.month
    return df.dropna()


def main():
    y = load_data()
    for horizon in (1, 3, 6):
        train, test = y[:-horizon], y[-horizon:]

        sarima = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
        sarima_pred = sarima.forecast(horizon)

        feats = make_features(y)
        tr = feats[feats.index <= train.index[-1]]
        te = feats[feats.index > train.index[-1]]
        model = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05)
        model.fit(tr.drop(columns=["y"]), tr["y"])
        xgb_pred = model.predict(te.drop(columns=["y"]))

        s_mape = mean_absolute_percentage_error(test, sarima_pred)
        x_mape = mean_absolute_percentage_error(test, xgb_pred)
        print(f"horizon={horizon}mo  SARIMA MAPE={s_mape:.3%}  XGBoost MAPE={x_mape:.3%}")


if __name__ == "__main__":
    main()
