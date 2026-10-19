"""Hypothesis Test 2: Winter vs Monsoon AQI (Welch t-test + Mann-Whitney U check)."""
from scipy.stats import ttest_ind, mannwhitneyu
from utils import load_data

df = load_data().dropna(subset=["Date", "AQI"])
df["Month"] = df["Date"].dt.month
winter = df[df["Month"].isin([12, 1, 2])]["AQI"]
monsoon = df[df["Month"].isin([6, 7, 8, 9])]["AQI"]

t_stat, p_value = ttest_ind(winter, monsoon, equal_var=False)
u_stat, p_mw = mannwhitneyu(winter, monsoon, alternative="two-sided")

print("--- T-test for AQI Between Winter and Monsoon ---")
print(f"Mean AQI Winter : {winter.mean():.2f}")
print(f"Mean AQI Monsoon: {monsoon.mean():.2f}")
print(f"T-statistic     : {t_stat:.4f}")
print(f"P-value         : {p_value}")
print(f"Mann-Whitney U p-value (robustness check): {p_mw}")

if p_value < 0.05:
    print("\nConclusion: Reject H0 (p < 0.05). Winter and monsoon AQI differ significantly.")
else:
    print("\nConclusion: Fail to reject H0 (p >= 0.05).")