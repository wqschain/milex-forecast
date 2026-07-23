"""
Step 5 proof-of-concept, round 2: two fixes tested together on Ukraine, per
the conclusion in ROADMAP.md step 5's finding (see also test_regressor_ukraine.py,
kept as-is as the record of round 1's negative result).

Round 1 established two separate problems with continuous spike_ratio +
changepoint_prior_scale=0.8:
  1. The regressor barely moved the forecast because Prophet's own trend
     already absorbed the 2022 shift via its changepoint (nothing left for
     the regressor to explain).
  2. spike_ratio itself decays back toward its ~1.0 baseline by 2025 even
     though real spending is still climbing (39.6% of GDP) - it detects the
     escalation *moment*, not the sustained state, so even a fully-effective
     regressor would have "let go" of the signal by the point where it's
     needed most for the forward forecast.

This round changes two things together, matching the user's request to test
them jointly rather than in isolation:

  A. growth="logistic" with an explicit cap, so Prophet has a structural
     ceiling on spending as % of GDP regardless of what the trend/regressor
     do. This directly targets the ">100% of GDP by 2037" failure mode,
     independent of whether the regressor signal is any good.
  B. A binary conflict_active flag (1 for 2022+, 0 before) instead of
     continuous spike_ratio. Sourced from known history (the full-scale
     invasion date), not thresholded from GDELT volume - thresholding
     spike_ratio would inherit the same decay-to-baseline problem from round
     1, since spike_ratio itself reverts by 2025 regardless of where the
     threshold is drawn. The flag stays exactly as informative in 2037 as it
     is in 2023, which spike_ratio cannot.

changepoint_prior_scale is left at 0.8 (Ukraine's originally selected value)
so the comparison isolates the effect of A+B rather than mixing in a third
change; lowering it further remains an untried direction if A+B alone don't
resolve the extrapolation problem.

Cap choice (0.50, i.e. 50% of GDP): a judgment call, not an empirical fact -
recorded here for scrutiny. Ukraine was already at 39.6% in 2025 and still
rising, so the cap has to sit above that. Sustained wartime economies have
historically reached this range (UK military spending peaked near 50% of
GDP during WW2); 50% is used as a round, defensible ceiling for this proof
of concept. No floor is set (logistic growth's floor is optional; omitting
it leaves no downward constraint, which is fine here since we're only
testing the upper bound).

yearly_seasonality=False on the fixed model only: an empirical check (see
git history / conversation record for the diagnostic) found that Prophet's
default yearly seasonality - a meaningless artifact on data that already has
exactly one point per year, with no sub-year signal to seasonally model -
adds a spurious wiggle (~+/-2pp) on top of the logistic trend. Since the
trend is what respects the cap, not yhat as a whole, this wiggle was enough
to push yhat visibly above the cap in the "conflict ends" scenario. This
artifact is not new here: train.py never disables yearly_seasonality either,
so production forecasts likely carry the same noise, just masked by trend
dominating the signal. Left out of scope to fix in train.py for now; noted
here because it directly undermines the cap this test is trying to validate.
The baseline model intentionally keeps Prophet's defaults (including yearly
seasonality) to represent current production behavior unchanged, so this
test isolates the effect of growth cap + conflict flag rather than mixing in
a fourth, unrelated fix.
"""
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error

CHANGEPOINT_SCALE = 0.8  # unchanged from round 1, to isolate the effect of the two new fixes
CAP = 0.50  # see module docstring
FORECAST_YEARS = 15

sipri = pd.read_csv("cleaned_data.csv")
gdelt = pd.read_csv("gdelt_country_year_clusters.csv")

ukraine_sipri = sipri[sipri["Country"] == "Ukraine"][["Year", "Spending"]]
ukraine_gdelt = gdelt[gdelt["Country"] == "Ukraine"][["Year", "spike_ratio"]]
ukraine = ukraine_sipri.merge(ukraine_gdelt, on="Year", how="inner").sort_values("Year").reset_index(drop=True)

ukraine["ds"] = pd.to_datetime(ukraine["Year"], format="%Y")
ukraine["y"] = ukraine["Spending"]
ukraine["cap"] = CAP
# Full-scale invasion (Feb 2022) is the clear regime shift in the actual
# spending data (2021=3.4% -> 2022=25.6%); the 2014 Donbas/Crimea period only
# moved spending from ~2.4% to ~3.8%, not distinguishable from noise, so it's
# treated as flag=0 for this proof of concept (see module docstring).
ukraine["conflict_active"] = (ukraine["Year"] >= 2022).astype(int)

# --- Question 1: does growth cap + conflict flag improve historical fit? ---
train = ukraine[ukraine["Year"] <= 2015].reset_index(drop=True)
test = ukraine[ukraine["Year"] > 2015].reset_index(drop=True)

# Same trend-only baseline as round 1 (linear growth, no regressor, no cap) -
# the original unconstrained model this is meant to improve on.
baseline_model = Prophet(changepoint_prior_scale=CHANGEPOINT_SCALE)
baseline_model.fit(train[["ds", "y"]])
baseline_future = baseline_model.make_future_dataframe(periods=len(test), freq="YE")
baseline_forecast = baseline_model.predict(baseline_future)
baseline_pred = baseline_forecast["yhat"].tail(len(test)).values
baseline_mae = mean_absolute_error(test["y"], baseline_pred)

fixed_model = Prophet(changepoint_prior_scale=CHANGEPOINT_SCALE, growth="logistic", yearly_seasonality=False)
fixed_model.add_regressor("conflict_active")
fixed_model.fit(train[["ds", "y", "cap", "conflict_active"]])
fixed_future = fixed_model.make_future_dataframe(periods=len(test), freq="YE")
# Positional, not date-merged (see test_regressor_ukraine.py's docstring for
# why: make_future_dataframe's appended dates are year-end, not year-start).
fixed_future["cap"] = CAP
fixed_future["conflict_active"] = np.concatenate([train["conflict_active"].values, test["conflict_active"].values])
fixed_forecast = fixed_model.predict(fixed_future)
fixed_pred = fixed_forecast["yhat"].tail(len(test)).values
fixed_mae = mean_absolute_error(test["y"], fixed_pred)

print("=== Question 1: historical fit (train <=2015, test 2016-2025) ===")
print(f"baseline (linear, no regressor)         MAE = {baseline_mae:.5f}")
print(f"fixed (logistic cap + conflict flag)    MAE = {fixed_mae:.5f}")
pct_change = (fixed_mae - baseline_mae) / baseline_mae * 100
print(f"change: {pct_change:+.1f}% ({'IMPROVED' if fixed_mae < baseline_mae else 'WORSE'})")
print()
print("year-by-year comparison:")
comparison = pd.DataFrame({
    "Year": test["Year"].values,
    "actual": test["y"].values,
    "baseline_pred": baseline_pred,
    "fixed_pred": fixed_pred,
    "conflict_active": test["conflict_active"].values,
})
print(comparison.to_string(index=False))
print()

# --- Question 2: forward forecast, conflict continues vs conflict ends ---
print("=== Question 2: 15-year forward forecast ===")
full_baseline = Prophet(changepoint_prior_scale=CHANGEPOINT_SCALE)
full_baseline.fit(ukraine[["ds", "y"]])
full_baseline_future = full_baseline.make_future_dataframe(periods=FORECAST_YEARS, freq="YE")
full_baseline_forecast = full_baseline.predict(full_baseline_future)
baseline_15y = full_baseline_forecast["yhat"].tail(FORECAST_YEARS).values

full_fixed = Prophet(changepoint_prior_scale=CHANGEPOINT_SCALE, growth="logistic", yearly_seasonality=False)
full_fixed.add_regressor("conflict_active")
full_fixed.fit(ukraine[["ds", "y", "cap", "conflict_active"]])

for label, assumed_flag in [
    ("conflict continues", 1),
    ("conflict ends (reverts to peacetime)", 0),
]:
    future = full_fixed.make_future_dataframe(periods=FORECAST_YEARS, freq="YE")
    future["cap"] = CAP
    future["conflict_active"] = np.concatenate([
        ukraine["conflict_active"].values,
        np.full(FORECAST_YEARS, assumed_flag),
    ])
    forecast = full_fixed.predict(future)
    pred_15y = forecast["yhat"].tail(FORECAST_YEARS).values

    print(f"--- scenario: {label} (conflict_active={assumed_flag}) ---")
    years = range(2026, 2026 + FORECAST_YEARS)
    for yr, base, fix in zip(years, baseline_15y, pred_15y):
        print(f"  {yr}  baseline={base*100:6.1f}%  fixed={fix*100:6.1f}%  diff={((fix-base)*100):+6.1f}pp")
    max_fixed = pred_15y.max()
    print(f"  max over horizon: fixed={max_fixed*100:.1f}% (cap={CAP*100:.0f}%)")
    print()
