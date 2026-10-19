"""Model 3: ARIMA(5,1,2). Run this BEFORE plot_arima.py."""
import os
import numpy as np
import pandas as pd
from datetime import timedelta
from statsmodels.tsa.arima.model import ARIMA
from utils import load_data, daily_aqi, metrics, OUT_DIR

FUTURE_DAYS = 30
ARIMA_ORDER = (5, 1, 2)

df = daily_aqi(load_data())
y = df.set_index("Date")["AQI"].asfreq("D").interpolate()

train_size = int(len(y) * 0.8)
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

print(f"Fitting ARIMA{ARIMA_ORDER} on training data (n={len(y_train)}) ...")
fit = ARIMA(y_train, order=ARIMA_ORDER).fit()
pred_test = fit.forecast(steps=len(y_test))
m = metrics(y_test.values, pred_test.values)

pd.DataFrame({"Date": y_test.index, "Actual_AQI": y_test.values,
              "ARIMA_Predicted": pred_test.values}).to_csv(
    os.path.join(OUT_DIR, "arima_test_predictions.csv"), index=False)

print("Refitting ARIMA on full dataset ...")
fit_full = ARIMA(y, order=ARIMA_ORDER).fit()
pred_future = fit_full.forecast(steps=FUTURE_DAYS)
future_dates = [y.index.max() + timedelta(days=i) for i in range(1, FUTURE_DAYS + 1)]
pd.DataFrame({"Date": future_dates, "ARIMA_Forecast": pred_future.values}).to_csv(
    os.path.join(OUT_DIR, "arima_future_forecast.csv"), index=False)

print("\nARIMA evaluation on test set:")
print(f"R2: {m['R2']:.3f}\nMAE: {m['MAE']:.2f}\nRMSE: {m['RMSE']:.2f}")
print(f"Mean 30-day forecast: {pred_future.mean():.1f}")
print("\nDone. Now run plot_arima.py")