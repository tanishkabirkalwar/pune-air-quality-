"""Model 4: Prophet (trend + yearly + weekly seasonality + Indian holidays)."""
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from utils import load_data, daily_aqi, metrics, OUT_DIR, save_fig

FUTURE_DAYS = 30

def build_model():
    m = Prophet(daily_seasonality=False, weekly_seasonality=True,
                yearly_seasonality=True, interval_width=0.95)
    try:
        m.add_country_holidays(country_name="IN")
    except Exception:
        pass
    return m

d = daily_aqi(load_data()).set_index("Date")["AQI"].asfreq("D").interpolate(limit_direction="both")
daily = d.reset_index(); daily.columns = ["ds", "y"]

train_size = int(len(daily) * 0.8)
train_df, test_df = daily.iloc[:train_size], daily.iloc[train_size:].reset_index(drop=True)

print(f"Fitting Prophet on train (n={len(train_df)}) ...")
m = build_model(); m.fit(train_df)
fc_test = m.predict(m.make_future_dataframe(periods=len(test_df), freq="D"))
pred_test = np.clip(fc_test["yhat"].iloc[-len(test_df):].values, 0, None)
res = metrics(test_df["y"].values, pred_test)
pd.DataFrame({"Date": test_df["ds"], "Actual": test_df["y"], "Predicted": pred_test}).to_csv(
    os.path.join(OUT_DIR, "prophet_aqi_test_predictions.csv"), index=False)
print("\nProphet evaluation on test set:")
print(f"R2 Score: {res['R2']:.3f}\nMAE: {res['MAE']:.2f}\nRMSE: {res['RMSE']:.2f}")

print("\nRefitting on full data ...")
m_full = build_model(); m_full.fit(daily)
fc = m_full.predict(m_full.make_future_dataframe(periods=FUTURE_DAYS, freq="D"))
for c in ["yhat", "yhat_lower", "yhat_upper"]:
    fc[c] = fc[c].clip(lower=0)
fc.iloc[-FUTURE_DAYS:][["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(
    columns={"ds": "Date", "yhat": "Predicted"}).to_csv(
    os.path.join(OUT_DIR, "prophet_aqi_future_forecast.csv"), index=False)
print(f"Mean 30-day forecast: {fc['yhat'].iloc[-FUTURE_DAYS:].mean():.1f}")

plt.figure(figsize=(12, 6))
plt.scatter(daily["ds"], daily["y"], color="gray", alpha=0.5, s=10, label="Actual AQI")
plt.plot(fc["ds"], fc["yhat"], color="blue", linewidth=1.5, label="Prophet Model Fit")
fut = fc.iloc[len(daily):]
plt.plot(fut["ds"], fut["yhat"], color="red", linestyle="--", marker="o", markersize=4,
         label=f"Future Forecast ({FUTURE_DAYS} days)")
plt.fill_between(fc["ds"], fc["yhat_lower"], fc["yhat_upper"], color="blue", alpha=0.15,
                 label="Confidence Interval (95%)")
plt.title("Prophet Forecast: Pune Daily Average AQI")
plt.xlabel("Date"); plt.ylabel("AQI"); plt.legend(); plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout(); save_fig("09_prophet_forecast.png"); plt.show()

m_full.plot_components(fc); save_fig("10_prophet_components.png"); plt.show()
print("\nDone.")