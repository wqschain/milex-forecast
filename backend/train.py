import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from prophet import Prophet
import joblib
import json

df = pd.read_csv("cleaned_data.csv")
test_countries = [
    "United States of America", "Canada", "Mexico", "Brazil", "Argentina", "Colombia",
    "United Kingdom", "Germany", "France", "Italy", "Poland", "Russia", "Ukraine",
    "Israel", "Saudi Arabia", "Türkiye", "Iran", "China", "India", "Japan",
    "Korea, South", "Pakistan",
    "Australia", "Indonesia", "Viet Nam", "South Africa", "Nigeria", "Egypt", "Algeria",
    "Sweden", "Norway"
]


def evaluate_country(country_name, changepoint_prior_scale=0.05):
    country_data = df[df["Country"] == country_name].dropna()

    train_data = country_data[country_data["Year"] <= 2015]
    test_data = country_data[country_data["Year"] > 2015]

    X_train = train_data[["Year"]]
    y_train = train_data["Spending"]
    X_test = test_data[["Year"]]
    y_test = test_data["Spending"]

    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    linear_predictions = linear_model.predict(X_test)
    linear_mae = mean_absolute_error(y_test, linear_predictions)

    prophet_df = country_data[["Year", "Spending"]].copy()
    prophet_df["ds"] = pd.to_datetime(prophet_df["Year"], format="%Y")
    prophet_df["y"] = prophet_df["Spending"]
    prophet_df = prophet_df[["ds", "y"]]

    prophet_train = prophet_df[prophet_df["ds"] <= "2015-01-01"]
    prophet_test = prophet_df[prophet_df["ds"] > "2015-01-01"]

    prophet_model = Prophet(changepoint_prior_scale=changepoint_prior_scale)
    prophet_model.fit(prophet_train)

    prophet_future = prophet_model.make_future_dataframe(periods=len(prophet_test), freq="YE")
    prophet_forecast = prophet_model.predict(prophet_future)

    prophet_predictions = prophet_forecast["yhat"].tail(len(prophet_test)).values
    prophet_mae = mean_absolute_error(prophet_test["y"], prophet_predictions)

    return linear_mae, prophet_mae


def calculate_volatility(country_name):
    country_data = df[df["Country"] == country_name].dropna().sort_values("Year")
    year_over_year_change = country_data["Spending"].pct_change()
    return year_over_year_change.std()


# Growth cap rollout (see ROADMAP.md step 5 round 2 finding): a Prophet
# logistic-growth ceiling on spending as % of GDP, for countries whose
# volatility is a statistical outlier (>2 std dev above the mean across all
# 31 countries - the threshold decided in step 6, applied here for real
# rather than assumed). Flagged countries are computed fresh from current
# data every run, not hardcoded, so this extends automatically if another
# country ever crosses the threshold.
#
# Cap values are deliberately NOT auto-computed from data - each one is a
# hand-set, documented judgment call (same category as the scenario
# templates in step 6), because a formula-derived cap would be presented as
# more rigorous than it actually is. A flagged country with no entry here
# is left uncapped and printed as a warning below, rather than silently
# guessing a value - matching the project's "undetermined, not silently
# defaulted" branching philosophy (step 4).
GROWTH_CAPS = {
    # 0.50 = 50% of GDP. Ukraine was already at 39.6% (2025) and still
    # rising, so the cap must sit above that; sustained wartime economies
    # have historically reached this range (UK military spending peaked
    # near 50% of GDP in WW2). See ROADMAP.md for full reasoning.
    "Ukraine": 0.50,
}

volatilities = {c: calculate_volatility(c) for c in test_countries}
_vol_values = list(volatilities.values())
_vol_threshold = (sum(_vol_values) / len(_vol_values)) + 2 * pd.Series(_vol_values).std()
flagged_countries = [c for c in test_countries if volatilities[c] > _vol_threshold]
print(f"Volatility-flagged countries (>{_vol_threshold:.4f}): {flagged_countries}")
for c in flagged_countries:
    if c not in GROWTH_CAPS:
        print(f"  WARNING: '{c}' is flagged but has no GROWTH_CAPS entry - left uncapped, unresolved.")


scales_to_test = [0.05, 0.3, 0.8]

# --- EXPLORATION PHASE 1: initial 6-country baseline comparison ---
# Found linear regression winning every test at Prophet's default scale (0.05).
# for country in test_countries:
#     lin_mae, _ = evaluate_country(country, changepoint_prior_scale=0.05)
#     print(f"\n{country} (Linear MAE: {lin_mae:.5f})")
#     for scale in scales_to_test:
#         _, proph_mae = evaluate_country(country, changepoint_prior_scale=scale)
#         print(f"  Prophet (scale={scale}): {proph_mae:.5f}")

# --- EXPLORATION PHASE 2: volatility investigation ---
# Tested whether year-over-year spending volatility predicts which model wins.
# Türkiye was the extreme outlier (volatility ~1.38 vs typical 0.03-0.3 range) —
# traced this to currency/inflation instability (lira lost 80%+ value since 2018),
# not real defense policy shifts, distorting the %GDP metric.
# for country in test_countries:
#     print(f"{country}: {calculate_volatility(country):.5f}")

# --- PRODUCTION PHASE: per-country model selection based on measured MAE ---
results_log = {}

for country in test_countries:
    try:
        lin_mae, _ = evaluate_country(country, changepoint_prior_scale=0.05)

        best_prophet_mae = None
        best_scale = None
        for scale in scales_to_test:
            _, proph_mae = evaluate_country(country, changepoint_prior_scale=scale)
            if best_prophet_mae is None or proph_mae < best_prophet_mae:
                best_prophet_mae = proph_mae
                best_scale = scale

        if lin_mae < best_prophet_mae:
            chosen_model = "linear"
            chosen_mae = lin_mae
        else:
            chosen_model = "prophet"
            chosen_mae = best_prophet_mae

        results_log[country] = {
            "chosen_model": chosen_model,
            "mae": chosen_mae,
            "linear_mae": lin_mae,
            "best_prophet_scale": best_scale,
            "best_prophet_mae": best_prophet_mae
        }
    except Exception as e:
        results_log[country] = {"error": str(e)}

with open("model_selection_log.json", "w") as f:
    json.dump(results_log, f, indent=2)

print("Done. Results saved to model_selection_log.json")

final_models = {}
growth_caps_used = {}

for country, result in results_log.items():
    if "error" in result:
        continue

    country_data = df[df["Country"] == country].dropna()
    cap = GROWTH_CAPS.get(country) if country in flagged_countries else None

    if result["chosen_model"] == "linear":
        # Growth cap is a Prophet-only mechanism (logistic growth). A flagged
        # country selected as linear would need a different fix; none exist
        # currently, so this isn't built - see ROADMAP.md.
        if cap is not None:
            print(f"  NOTE: '{country}' is flagged with a growth cap defined, but its chosen model is "
                  f"linear (no growth-cap mechanism for linear models) - cap not applied.")
        X = country_data[["Year"]]
        y = country_data["Spending"]
        final_model = LinearRegression()
        final_model.fit(X, y)
    else:
        prophet_df = country_data[["Year", "Spending"]].copy()
        prophet_df["ds"] = pd.to_datetime(prophet_df["Year"], format="%Y")
        prophet_df["y"] = prophet_df["Spending"]
        if cap is not None:
            # yearly_seasonality=False is required, not optional, for the cap
            # to actually hold: Prophet's default yearly seasonality adds a
            # +/-2pp wiggle on top of the capped trend on this yearly-only
            # data (no sub-year signal exists to seasonally model), which was
            # enough to push forecasts above the cap in testing - see
            # ROADMAP.md step 5 round 2. Only applied for capped countries,
            # not all 31 - the general fix is filed as a separate follow-up.
            prophet_df["cap"] = cap
            final_model = Prophet(changepoint_prior_scale=result["best_prophet_scale"],
                                   growth="logistic", yearly_seasonality=False)
            final_model.fit(prophet_df[["ds", "y", "cap"]])
            growth_caps_used[country] = cap
        else:
            final_model = Prophet(changepoint_prior_scale=result["best_prophet_scale"])
            final_model.fit(prophet_df[["ds", "y"]])

    final_models[country] = final_model

joblib.dump(final_models, "final_models.pkl")
print(f"Trained and saved {len(final_models)} final models.")

# Both "capped" and "flagged" are saved, not just "capped", so main.py can
# tell "flagged but not yet capped" apart from "never flagged" at serving
# time - collapsing those into the same absence-of-a-cap-entry, as an
# earlier version of this file did, meant a flagged country with no cap
# (missing GROWTH_CAPS entry, or an unsupported chosen_model like "linear")
# would silently be served a completely normal, unbounded forecast with no
# indication anything was different. See ROADMAP.md.
growth_cap_status = {
    "capped": growth_caps_used,
    "flagged": flagged_countries,
}
with open("growth_caps.json", "w") as f:
    json.dump(growth_cap_status, f, indent=2)
uncapped_flagged = [c for c in flagged_countries if c not in growth_caps_used]
print(f"Saved growth cap status to growth_caps.json: {len(growth_caps_used)} capped, "
      f"{len(uncapped_flagged)} flagged-but-uncapped ({uncapped_flagged})")