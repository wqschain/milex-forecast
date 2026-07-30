"""
Step 6 proof-of-concept: three named scenario forecasts for Ukraine, using
the conflict template decided in docs/ROADMAP_v2_working_notes.md step 6, built on the growth cap
+ binary conflict_active regressor already validated together in step 5
round 2 (backend/test_regressor_ukraine_v2.py). Standalone review artifact,
per explicit scope for this step - deliberately not wired into
train.py/main.py/the frontend at the time this was written; the same
template was later trained and served for real via scenario_templates.py
(see docs/ROADMAP_v2_working_notes.md). Kept as the standalone record of
what was proven before that happened, not updated to match the production
wiring.

Model config matches what's already been adopted, unchanged: Ukraine's
originally-selected changepoint_prior_scale=0.8, growth="logistic" with
cap=0.50 (50% of GDP - see docs/ROADMAP_v2_working_notes.md for the empirical + historical
grounding), yearly_seasonality=False (required for the cap to actually
hold, not optional - see docs/ROADMAP_v2_working_notes.md step 5 round 2), and conflict_active
(1 for 2022+, the full-scale invasion) as an added regressor whose fitted
magnitude is a hand-set assumption, not a statistically validated
coefficient (already decided and documented, not re-litigated here).

Three scenarios, matching step 6's conflict template wording exactly for
the first two; the third ("gradual de-escalation") requires a concrete
choice the template left open, explained below:

1. "Conflict continues at current intensity" - conflict_active held at 1
   for the full 15-year horizon.
2. "Conflict resolves within 3 years" - conflict_active at 1 for years 1-3
   (2026-2028), 0 from year 4 (2029) on - a sharp cutoff, matching the
   template's literal wording.
3. "Gradual de-escalation" - conflict_active declines *linearly* from 1.0
   to 0.0 over DEESCALATION_YEARS (chosen as 6), then stays at 0. Fed as a
   continuous value in [0,1] rather than a second binary schedule: Prophet
   applies a single fitted coefficient as beta * conflict_active, so a
   fractional value just scales the effect proportionally - no retraining
   or redefinition of the regressor is needed to do this. The 6-year
   window is a deliberate choice, not a rounding of scenario 2's 3 years:
   it needs to be long enough that this scenario reads as a materially
   different real-world assumption (a protracted wind-down) rather than a
   near-duplicate of "resolves within 3 years" with the corner smoothed
   off. If the two scenarios converged to the same shape, one of them
   would be redundant.
"""
import numpy as np
import pandas as pd
from prophet import Prophet

CHANGEPOINT_SCALE = 0.8
CAP = 0.50
FORECAST_YEARS = 15
DEESCALATION_YEARS = 6

sipri = pd.read_csv("cleaned_data.csv")
ukraine = sipri[sipri["Country"] == "Ukraine"][["Year", "Spending"]].sort_values("Year").reset_index(drop=True)
ukraine["ds"] = pd.to_datetime(ukraine["Year"], format="%Y")
ukraine["y"] = ukraine["Spending"]
ukraine["cap"] = CAP
ukraine["conflict_active"] = (ukraine["Year"] >= 2022).astype(int)

model = Prophet(changepoint_prior_scale=CHANGEPOINT_SCALE, growth="logistic", yearly_seasonality=False)
model.add_regressor("conflict_active")
model.fit(ukraine[["ds", "y", "cap", "conflict_active"]])


def continues():
    return np.ones(FORECAST_YEARS)


def resolves_3y():
    flag = np.zeros(FORECAST_YEARS)
    flag[:3] = 1.0
    return flag


def gradual_deescalation():
    return np.clip(1 - np.arange(FORECAST_YEARS) / DEESCALATION_YEARS, 0, 1)


scenarios = {
    "Conflict continues at current intensity": continues(),
    "Conflict resolves within 3 years": resolves_3y(),
    "Gradual de-escalation": gradual_deescalation(),
}

results = {}
for name, future_flags in scenarios.items():
    future = model.make_future_dataframe(periods=FORECAST_YEARS, freq="YE")
    future["cap"] = CAP
    future["conflict_active"] = np.concatenate([ukraine["conflict_active"].values, future_flags])
    forecast = model.predict(future)
    results[name] = forecast["yhat"].tail(FORECAST_YEARS).values

years = list(range(2026, 2026 + FORECAST_YEARS))

print("=== conflict_active schedules fed to each scenario (future years only) ===")
for name, flags in scenarios.items():
    print(f"{name}:")
    print("  " + ", ".join(f"{y}={f:.2f}" for y, f in zip(years, flags)))
print()

print("=== Ukraine: three scenario forecasts (% of GDP), cap=50% ===")
header = f"{'Year':>6}" + "".join(f"{name[:28]:>32}" for name in results)
print(header)
for i, yr in enumerate(years):
    row = f"{yr:>6}" + "".join(f"{results[name][i]*100:>31.1f}%" for name in results)
    print(row)
print()

print("=== summary ===")
for name, vals in results.items():
    print(f"{name}: max over horizon = {vals.max()*100:.1f}%, value at 2040 = {vals[-1]*100:.1f}%")
