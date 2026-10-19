"""Plots results saved by arima_model.py (run that first)."""
import os
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_data, daily_aqi, OUT_DIR, save_fig

df = daily_aqi(load_data())
test_pred = pd.read_csv(os.path.join(OUT_DIR, "arima_test_predictions.csv"), parse_dates=["Date"])
future = pd.read_csv(os.path.join(OUT_DIR, "arima_future_forecast.csv"), parse_dates=["Date"])

plt.figure(figsize=(10, 5))
plt.scatter(df["Date"], df["AQI"], color="lightgray", s=6, label="Actual Data")
plt.plot(test_pred["Date"], test_pred["ARIMA_Predicted"], color="darkorange", linewidth=2, label="ARIMA Test Forecast")
plt.plot(future["Date"], future["ARIMA_Forecast"], color="red", linestyle="--", marker="o", label="Future Forecast (30 days)")
plt.title("ARIMA - Pune Daily AQI Forecast")
plt.xlabel("Date"); plt.ylabel("Daily Average AQI")
plt.grid(True, linestyle="--", alpha=0.6); plt.legend(); plt.tight_layout()
save_fig("08_arima.png"); plt.show()