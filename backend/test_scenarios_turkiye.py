"""
Step 6 proof-of-concept: two named scenario forecasts for Türkiye, using the
currency-instability template decided in ROADMAP.md step 6. Standalone
review artifact, mirroring exactly how Ukraine's scenarios were proven out
in backend/test_scenarios_ukraine.py before anything touched main.py or the
frontend - same discipline applies here, not yet wired in.

Per the corrected design (ROADMAP.md, 2026-07-23): Türkiye is a founding,
hand-verified validation case exactly like Ukraine, not an "undetermined"
country - the doubly-confirmed GDELT/volatility blind spot means our
automated tools can't detect this crisis on their own, but the crisis
itself is independently researched and confirmed (the lira's real,
well-documented depreciation since 2018), the same epistemic basis Ukraine's
conflict_active flag was built on.

Three things had to be checked empirically before building this, none of
which could be assumed from Ukraine's precedent:

1. Which model actually needs overriding. model_selection_log.json has
   Türkiye's chosen production model as LINEAR regression (MAE 0.00293,
   decisively better than Prophet's best of 0.00727 across all tested
   scales) - not Prophet, unlike Ukraine. Linear regression has no mechanism
   to condition a forecast on a future assumption, so building any scenario
   at all requires deliberately using Prophet here regardless of it being
   the worse-fitting model by MAE. This is a real, explicit trade-off:
   scenario capability is being bought at the cost of raw historical fit.
2. Whether a growth cap is needed. Checked directly: Türkiye's historical
   max is 4.30% of GDP (1982), and every year since the 2018 lira crisis
   onset is at or below that (2018-2025 range: 1.6%-2.6%) - the "instability"
   shows up as noisy fluctuation within an already-low, non-exploding band,
   not runaway growth. No growth cap applied - an explicit, evidence-based
   decision (Türkiye's own historical data checked directly), not a silent
   omission. Per the two-gate design in ROADMAP.md, this is what "no cap
   needed" is supposed to look like: a deliberate, documented judgment, not
   an absence.
3. How "stabilizes" should transition. ROADMAP's template says "reverts
   toward the country's own pre-crisis baseline within the forecast
   horizon" without specifying a timeframe. Checked real historical
   precedent rather than picking one: the 1992 European ERM currency crises
   took roughly 3 years (UK) to 4 years (Sweden) from onset to full
   resolution. A currency crisis is a multi-year credibility-rebuilding
   process, not a discrete event like a ceasefire, so - unlike Ukraine's
   conflict template, which has both a sharp 3-year cutoff and a separate
   gradual scenario - Türkiye's single "stabilizes" scenario uses a gradual
   linear decline, DEESCALATION_YEARS=4, landing at the upper end of that
   historical range since Türkiye's crisis (7+ years unresolved as of 2025)
   has already run longer than either reference case.

currency_active: 1 for 2018+ (the lira crisis onset - "lira lost 80%+ value
since 2018" per train.py's original investigation), 0 before.

yearly_seasonality=False on every model here, applied from the start this
time rather than caught after the fact: the same artifact found and fixed
for Ukraine (ROADMAP.md step 5 round 2) - a meaningless seasonal term on
data that's already one point per year - applies identically here. Confirmed
by first running without it: the forecast oscillated year-to-year and even
went negative by 2040, which is nonsensical for a spending share and was
the seasonality artifact, not a real result - caught before being shown
to anyone, not shipped as a finding.
"""
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error

CHANGEPOINT_SCALE = 0.05  # Türkiye's best-tested Prophet scale per model_selection_log.json (best_prophet_scale),
# even though linear regression scored lower MAE overall and was what actually got chosen for production
FORECAST_YEARS = 15
STABILIZE_YEARS = 4

sipri = pd.read_csv("cleaned_data.csv")
turkiye = sipri[sipri["Country"] == "Türkiye"][["Year", "Spending"]].sort_values("Year").reset_index(drop=True)
turkiye["ds"] = pd.to_datetime(turkiye["Year"], format="%Y")
turkiye["y"] = turkiye["Spending"]
turkiye["currency_active"] = (turkiye["Year"] >= 2018).astype(int)

print(f"Historical max: {turkiye['y'].max()*100:.2f}% of GDP (year {turkiye.loc[turkiye['y'].idxmax(), 'Year']})")
print(f"2018-2025 range: {turkiye[turkiye['Year']>=2018]['y'].min()*100:.2f}% - "
      f"{turkiye[turkiye['Year']>=2018]['y'].max()*100:.2f}% of GDP")
print("-> no growth cap applied (see module docstring)")
print()

# --- Question 1: backtest, same discipline as Ukraine's round 2 - and the
# same invalidity is expected here too, for the same structural reason. ---
train = turkiye[turkiye["Year"] <= 2015].reset_index(drop=True)
test = turkiye[turkiye["Year"] > 2015].reset_index(drop=True)

baseline_model = Prophet(changepoint_prior_scale=CHANGEPOINT_SCALE, yearly_seasonality=False)
baseline_model.fit(train[["ds", "y"]])
baseline_future = baseline_model.make_future_dataframe(periods=len(test), freq="YE")
baseline_pred = baseline_model.predict(baseline_future)["yhat"].tail(len(test)).values
baseline_mae = mean_absolute_error(test["y"], baseline_pred)

regressor_model = Prophet(changepoint_prior_scale=CHANGEPOINT_SCALE, yearly_seasonality=False)
regressor_model.add_regressor("currency_active")
regressor_model.fit(train[["ds", "y", "currency_active"]])
regressor_future = regressor_model.make_future_dataframe(periods=len(test), freq="YE")
regressor_future["currency_active"] = np.concatenate([train["currency_active"].values, test["currency_active"].values])
regressor_pred = regressor_model.predict(regressor_future)["yhat"].tail(len(test)).values
regressor_mae = mean_absolute_error(test["y"], regressor_pred)

print("=== Question 1: historical backtest (train <=2015, test 2016-2025) ===")
print(f"baseline (no regressor)        MAE = {baseline_mae:.5f}")
print(f"regressor (+currency_active)   MAE = {regressor_mae:.5f}")
n_train_active = train["currency_active"].sum()
print(f"currency_active=1 rows in training window: {n_train_active} "
      f"({'INVALID BACKTEST - same structural issue as Ukraine round 2' if n_train_active == 0 else 'has training examples'})")
print()

# --- Question 2: full retrain, two named scenarios ---
full_model = Prophet(changepoint_prior_scale=CHANGEPOINT_SCALE, yearly_seasonality=False)
full_model.add_regressor("currency_active")
full_model.fit(turkiye[["ds", "y", "currency_active"]])


def continues():
    return np.ones(FORECAST_YEARS)


def stabilizes():
    return np.clip(1 - np.arange(FORECAST_YEARS) / STABILIZE_YEARS, 0, 1)


scenarios = {
    "Currency instability continues": continues(),
    "Currency stabilizes": stabilizes(),
}

results = {}
for name, future_flags in scenarios.items():
    future = full_model.make_future_dataframe(periods=FORECAST_YEARS, freq="YE")
    future["currency_active"] = np.concatenate([turkiye["currency_active"].values, future_flags])
    forecast = full_model.predict(future)
    results[name] = forecast["yhat"].tail(FORECAST_YEARS).values

years = list(range(2026, 2026 + FORECAST_YEARS))
print("=== Question 2: 15-year forward forecast, two scenarios (% of GDP) ===")
header = f"{'Year':>6}" + "".join(f"{name[:30]:>34}" for name in results)
print(header)
for i, yr in enumerate(years):
    row = f"{yr:>6}" + "".join(f"{results[name][i]*100:>33.2f}%" for name in results)
    print(row)
print()
for name, vals in results.items():
    print(f"{name}: max over horizon = {vals.max()*100:.2f}%, value at 2040 = {vals[-1]*100:.2f}%")
