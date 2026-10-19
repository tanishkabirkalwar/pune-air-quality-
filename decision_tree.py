"""Model 2: Decision Tree Regressor - (a) day count only, (b) with seasonal features."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.tree import DecisionTreeRegressor
from utils import load_data, daily_aqi, metrics, save_fig

data = daily_aqi(load_data())
data["Day_Count"] = (data["Date"] - data["Date"].min()).dt.days
data["Month"] = data["Date"].dt.month
data["DayOfYear"] = data["Date"].dt.dayofyear
y = data["AQI"]
train_size = int(len(data) * 0.8)

def run(features, label):
    X = data[features]
    model = DecisionTreeRegressor(max_depth=6, random_state=42)
    model.fit(X.iloc[:train_size], y.iloc[:train_size])
    m = metrics(y.iloc[train_size:], model.predict(X.iloc[train_size:]))
    print(f"\nDecision Tree ({label}):")
    print(f"R2 Score: {m['R2']:.3f}\nMAE: {m['MAE']:.2f}\nRMSE: {m['RMSE']:.2f}")
    return model

model = run(["Day_Count"], "Day_Count only")
model_s = run(["Day_Count", "Month", "DayOfYear"], "with seasonal features")

future_dates = pd.DatetimeIndex([data["Date"].max() + timedelta(days=i) for i in range(1, 31)])
fut = pd.DataFrame({"Day_Count": (future_dates - data["Date"].min()).days,
                    "Month": future_dates.month, "DayOfYear": future_dates.dayofyear})
future_preds = model_s.predict(fut)
print(f"\nMean 30-day forecast (seasonal tree): {future_preds.mean():.1f}")

plt.figure(figsize=(10, 5))
plt.scatter(data["Date"], y, color="lightgray", s=6, label="Actual Data")
plt.plot(data["Date"], model_s.predict(data[["Day_Count", "Month", "DayOfYear"]]),
         color="green", linewidth=1.5, label="Decision Tree Fit (seasonal)")
plt.plot(future_dates, future_preds, color="red", linestyle="--", marker="o", label="Future Forecast (30 days)")
plt.title("Decision Tree Regressor - Pune Daily AQI Forecast")
plt.xlabel("Date"); plt.ylabel("Daily Average AQI"); plt.legend(); plt.tight_layout()
save_fig("07_decision_tree.png"); plt.show()