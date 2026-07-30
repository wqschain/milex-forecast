"""
STATUS (2026-07-22): round 1, negative result - resolved by round 2
(backend/test_regressor_ukraine_v2.py, 2026-07-23), which adopted a growth
cap + binary conflict flag instead of the continuous spike_ratio tested
here. Kept as-is as the record of round 1's negative result, not updated to
match round 2's resolution. See docs/ROADMAP_v2_working_notes.md step 5's finding for full detail. Short version:
the regressor mechanism works (Prophet accepts and fits it), but it does
NOT bound Ukraine's extrapolation - historical MAE improves by only 0.4%,
and two very different future spike_ratio assumptions (1.03 vs 3.88, ~4x
apart) move the 2040 forecast by only ~1 percentage point. Both are
root-caused to changepoint_prior_scale=0.8 letting Prophet's own trend
absorb the 2022 shift directly, leaving little for the regressor to add.
Three untried next directions are recorded in docs/ROADMAP_v2_working_notes.md: lower
changepoint_prior_scale, add a logistic growth cap, or use a binary
conflict-active flag instead of continuous spike_ratio. Committed as a
record of what was tried and why it didn't work yet, not as production
code - not wired into train.py/evaluate_country().

Step 5 proof-of-concept: does wiring spike_ratio into Prophet as a
regressor actually help, for Ukraine specifically (our clearest validated
case)? Two questions, in order:

1. Historical fit: on the SAME chronological train/test split already used
   for model selection (train <=2015, test 2016-2025), does adding
   spike_ratio as a regressor reduce MAE versus the current no-regressor
   model? This uses real, known historical spike_ratio for the test years
   too - no forecasting assumption needed for backtesting.
2. Forward forecast: retrained on the full 1980-2025 range, how does a
   15-year forecast compare between the current (unconstrained trend-only)
   model and the regressor-augmented model under an explicit future
   spike_ratio assumption?

Not yet wired into train.py/evaluate_country() - this is a standalone
check before building the regressor out for all flagged countries.

Note on regressor values for the appended future rows: Prophet's
make_future_dataframe() generates its own `ds` sequence for the newly
appended periods, which doesn't necessarily land on the same dates as the
`ds` column built here from Year via pd.to_datetime(Year, format="%Y")
(year-start) - freq="YE" appends year-*end* dates. A ds-based merge would
silently produce NaN for those rows. Regressor values are attached
positionally instead (matching how train.py already extracts test
predictions via .tail(n), not by date), which is correct regardless of
exact date arithmetic.
"""
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error

CHANGEPOINT_SCALE = 0.8  # Ukraine's originally selected scale (model_selection_log.json)
FORECAST_YEARS = 15

sipri = pd.read_csv("cleaned_data.csv")
gdelt = pd.read_csv("gdelt_country_year_clusters.csv")

ukraine_sipri = sipri[sipri["Country"] == "Ukraine"][["Year", "Spending"]]
ukraine_gdelt = gdelt[gdelt["Country"] == "Ukraine"][["Year", "spike_ratio"]]
ukraine = ukraine_sipri.merge(ukraine_gdelt, on="Year", how="inner").sort_values("Year").reset_index(drop=True)
assert ukraine["spike_ratio"].isna().sum() == 0, "spike_ratio must be complete for every year"

ukraine["ds"] = pd.to_datetime(ukraine["Year"], format="%Y")
ukraine["y"] = ukraine["Spending"]

# --- Question 1: does the regressor improve historical fit? ---
train = ukraine[ukraine["Year"] <= 2015].reset_index(drop=True)
test = ukraine[ukraine["Year"] > 2015].reset_index(drop=True)

baseline_model = Prophet(changepoint_prior_scale=CHANGEPOINT_SCALE)
baseline_model.fit(train[["ds", "y"]])
baseline_future = baseline_model.make_future_dataframe(periods=len(test), freq="YE")
baseline_forecast = baseline_model.predict(baseline_future)
baseline_pred = baseline_forecast["yhat"].tail(len(test)).values
baseline_mae = mean_absolute_error(test["y"], baseline_pred)

regressor_model = Prophet(changepoint_prior_scale=CHANGEPOINT_SCALE)
regressor_model.add_regressor("spike_ratio")
regressor_model.fit(train[["ds", "y", "spike_ratio"]])
regressor_future = regressor_model.make_future_dataframe(periods=len(test), freq="YE")
# Positional, not date-merged (see module docstring): first len(train) rows
# get train's real spike_ratio, last len(test) rows get test's real
# spike_ratio (a backtest - real known history, not an assumption).
regressor_future["spike_ratio"] = np.concatenate([train["spike_ratio"].values, test["spike_ratio"].values])
regressor_forecast = regressor_model.predict(regressor_future)
regressor_pred = regressor_forecast["yhat"].tail(len(test)).values
regressor_mae = mean_absolute_error(test["y"], regressor_pred)

print("=== Question 1: historical fit (train <=2015, test 2016-2025) ===")
print(f"baseline (no regressor)      MAE = {baseline_mae:.5f}")
print(f"regressor (+ spike_ratio)    MAE = {regressor_mae:.5f}")
pct_change = (regressor_mae - baseline_mae) / baseline_mae * 100
print(f"change: {pct_change:+.1f}% ({'IMPROVED' if regressor_mae < baseline_mae else 'WORSE'})")
print()
print("year-by-year comparison:")
comparison = pd.DataFrame({
    "Year": test["Year"].values,
    "actual": test["y"].values,
    "baseline_pred": baseline_pred,
    "regressor_pred": regressor_pred,
    "spike_ratio": test["spike_ratio"].values,
})
print(comparison.to_string(index=False))
print()

# --- Question 2: forward forecast under an explicit scenario assumption ---
print("=== Question 2: 15-year forward forecast ===")
full_baseline = Prophet(changepoint_prior_scale=CHANGEPOINT_SCALE)
full_baseline.fit(ukraine[["ds", "y"]])
full_baseline_future = full_baseline.make_future_dataframe(periods=FORECAST_YEARS, freq="YE")
full_baseline_forecast = full_baseline.predict(full_baseline_future)
baseline_15y = full_baseline_forecast["yhat"].tail(FORECAST_YEARS).values

full_regressor = Prophet(changepoint_prior_scale=CHANGEPOINT_SCALE)
full_regressor.add_regressor("spike_ratio")
full_regressor.fit(ukraine[["ds", "y", "spike_ratio"]])

last_value = ukraine["spike_ratio"].iloc[-1]  # 2025
post_invasion_avg = ukraine[ukraine["Year"] >= 2022]["spike_ratio"].mean()  # 2022-2025 mean
print(f"spike_ratio, last observed year (2025): {last_value:.2f}")
print(f"spike_ratio, post-invasion (2022-2025) average: {post_invasion_avg:.2f}")
print()

for label, assumed_value in [
    ("hold at last observed value (2025)", last_value),
    ("hold at post-invasion (2022-2025) average", post_invasion_avg),
]:
    future = full_regressor.make_future_dataframe(periods=FORECAST_YEARS, freq="YE")
    future["spike_ratio"] = np.concatenate([
        ukraine["spike_ratio"].values,
        np.full(FORECAST_YEARS, assumed_value),
    ])
    forecast = full_regressor.predict(future)
    pred_15y = forecast["yhat"].tail(FORECAST_YEARS).values

    print(f"--- scenario: {label} (spike_ratio={assumed_value:.2f}) ---")
    years = range(2026, 2026 + FORECAST_YEARS)
    for yr, base, reg in zip(years, baseline_15y, pred_15y):
        print(f"  {yr}  baseline={base*100:6.1f}%  regressor={reg*100:6.1f}%  diff={((reg-base)*100):+6.1f}pp")
    print()
