"""Compares Linear Regression, Decision Tree, ARIMA and Prophet on the last 20 %."""
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from statsmodels.tsa.arima.model import ARIMA
from utils import load_data, daily_aqi, metrics, OUT_DIR, save_fig

data = daily_aqi(load_data())
data["Day_Count"] = (data["Date"] - data["Date"].min()).dt.days
data["Month"] = data["Date"].dt.month
data["DayOfYear"] = data["Date"].dt.dayofyear
n = int(len(data) * 0.8)
tr, te = data.iloc[:n], data.iloc[n:]
rows = {}

lr = LinearRegression().fit(tr[["Day_Count"]], tr["AQI"])
rows["Linear Regression"] = metrics(te["AQI"], lr.predict(te[["Day_Count"]]))

dt = DecisionTreeRegressor(max_depth=6, random_state=42).fit(tr[["Day_Count"]], tr["AQI"])
rows["Decision Tree (day count)"] = metrics(te["AQI"], dt.predict(te[["Day_Count"]]))

f = ["Day_Count", "Month", "DayOfYear"]
dts = DecisionTreeRegressor(max_depth=6, random_state=42).fit(tr[f], tr["AQI"])
rows["Decision Tree (seasonal)"] = metrics(te["AQI"], dts.predict(te[f]))

y = data.set_index("Date")["AQI"].asfreq("D").interpolate()
k = int(len(y) * 0.8)
arima = ARIMA(y.iloc[:k], order=(5, 1, 2)).fit()
rows["ARIMA (5,1,2)"] = metrics(y.iloc[k:].values, arima.forecast(steps=len(y) - k).values)

try:
    from prophet import Prophet
    d = y.reset_index(); d.columns = ["ds", "y"]
    m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    try:
        m.add_country_holidays(country_name="IN")
    except Exception:
        pass
    m.fit(d.iloc[:k])
    p = m.predict(m.make_future_dataframe(periods=len(d) - k, freq="D"))["yhat"].iloc[-(len(d) - k):]
    rows["Prophet"] = metrics(d["y"].iloc[k:].values, np.clip(p.values, 0, None))
except Exception as e:
    print("Prophet skipped:", e)

res = pd.DataFrame(rows).T.round(3)
print("\nModel comparison (test = last 20% of days):\n")
print(res)
res.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"))

res["MAE"].plot(kind="bar", color="teal", figsize=(9, 5))
plt.title("Model Comparison - MAE (lower is better)"); plt.ylabel("MAE")
plt.xticks(rotation=30, ha="right"); plt.tight_layout()
save_fig("11_model_comparison.png"); plt.show()