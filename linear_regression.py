"""Model 1: Linear Regression baseline (AQI vs day count) + 30-day forecast."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from utils import load_data, daily_aqi, metrics, save_fig

data = daily_aqi(load_data())
data["Day_Count"] = (data["Date"] - data["Date"].min()).dt.days
X, y = data[["Day_Count"]], data["AQI"]

train_size = int(len(data) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

model = LinearRegression().fit(X_train, y_train)
m = metrics(y_test, model.predict(X_test))
print("Linear Regression Performance:")
print(f"R2 Score: {m['R2']:.3f}\nMAE: {m['MAE']:.2f}\nRMSE: {m['RMSE']:.2f}")

future_days = 30
last_day = int(data["Day_Count"].max())
future_X = pd.DataFrame({"Day_Count": np.arange(last_day + 1, last_day + future_days + 1)})
future_dates = [data["Date"].max() + timedelta(days=i) for i in range(1, future_days + 1)]
future_preds = model.predict(future_X)
print(f"Mean 30-day forecast: {future_preds.mean():.1f}")

plt.figure(figsize=(10, 5))
plt.scatter(data["Date"], y, color="lightgray", s=6, label="Actual Data")
plt.plot(data["Date"], model.predict(X), color="blue", linewidth=2, label="Regression Line")
plt.plot(future_dates, future_preds, color="red", linestyle="--", marker="o", label="Future Forecast (30 days)")
plt.title("Linear Regression - Pune Daily AQI Forecast")
plt.xlabel("Date"); plt.ylabel("Daily Average AQI")
plt.grid(True, linestyle="--", alpha=0.6); plt.legend(); plt.tight_layout()
save_fig("06_linear_regression.png"); plt.show()