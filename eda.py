"""Exploratory Data Analysis."""
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data, daily_aqi, POLLUTANTS, save_fig

df = load_data()
print(df.head())
print("\nShape:", df.shape)
print("\nMissing values:\n", df.isna().sum())
print("\nDescribe:\n", df.describe())

# I. AQI distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["AQI"].dropna(), bins=40, kde=True, color="steelblue")
plt.title("Air Quality Index Distribution (Pune)")
plt.xlabel("AQI"); plt.tight_layout(); save_fig("01_aqi_distribution.png"); plt.show()

# II. Station-wise average AQI
if "Location" in df.columns:
    avg = df.groupby("Location")["AQI"].mean().sort_values(ascending=False)
    print("\nAverage AQI by location:\n", avg.round(1))
    plt.figure(figsize=(10, 5))
    avg.plot(kind="bar", color="tomato")
    plt.title("Station-wise Average AQI"); plt.ylabel("Average AQI")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    save_fig("02_station_avg_aqi.png"); plt.show()
else:
    print("No location/station column found - skipping station-wise chart.")

# III. Pollutant levels
cols = [c for c in POLLUTANTS if c in df.columns]
plt.figure(figsize=(8, 5))
df[cols].mean().plot(kind="bar", color="seagreen")
plt.title("Average Pollutant Concentration (ug/m3)")
plt.tight_layout(); save_fig("03_pollutant_means.png"); plt.show()

# IV. Monthly pattern
d = daily_aqi(df)
d["Month"] = d["Date"].dt.month
monthly = d.groupby("Month")["AQI"].mean()
print("\nMonthly average AQI:\n", monthly.round(1))
plt.figure(figsize=(9, 5))
monthly.plot(marker="o")
plt.title("Monthly Average AQI (Pune)"); plt.xlabel("Month"); plt.ylabel("AQI")
plt.xticks(range(1, 13)); plt.grid(alpha=0.4); plt.tight_layout()
save_fig("04_monthly_aqi.png"); plt.show()