"""Pearson correlation between pollutants and AQI + heatmap."""
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data, POLLUTANTS, save_fig

data = load_data()
cols = [c for c in POLLUTANTS + ["AQI"] if c in data.columns]
corr = data[cols].corr()
print(corr.round(2))

plt.figure(figsize=(7, 5.5))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Pollutants and AQI (Pune)")
plt.tight_layout(); save_fig("05_correlation_heatmap.png"); plt.show()