"""Hypothesis Test 1: Runs test for randomness of daily AQI."""
from statsmodels.sandbox.stats.runs import runstest_1samp
from utils import load_data, daily_aqi

daily = daily_aqi(load_data()).set_index("Date")["AQI"]
z_stat, p_value = runstest_1samp(daily.values, cutoff="median", correction=False)

print("--- Runs Test for Randomness of Daily AQI ---")
print(f"Z-statistic: {z_stat:.4f}")
print(f"P-value: {p_value}")
if p_value < 0.05:
    print("\nConclusion: Reject H0 (p < 0.05). The AQI sequence is not random - it has trends/patterns.")
else:
    print("\nConclusion: Fail to reject H0 (p >= 0.05). No evidence the data is non-random.")