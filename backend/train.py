import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from prophet import Prophet
import joblib
import json
from scenario_pipeline import determine_cap
from scenario_templates import SCENARIO_TEMPLATES, train_scenario_model

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

    # yearly_seasonality=False: this data is already one point per year (no
    # sub-year signal exists to seasonally model), so Prophet's default
    # yearly term is a meaningless artifact here, not a real pattern - see
    # ROADMAP.md step 5 round 2. Applied here (not just on the final model)
    # so model selection itself (which model wins, which scale) is
    # evaluated under the same corrected setting the final model uses -
    # otherwise the chosen model/scale would be stale, picked to fit noise
    # this fix removes.
    prophet_model = Prophet(changepoint_prior_scale=changepoint_prior_scale, yearly_seasonality=False)
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
# GROWTH_CAPS holds hand-researched values only - each one a documented
# judgment call grounded in a real, externally-checkable historical anchor
# (e.g. Ukraine's UK WW2 defense-spending precedent). See ROADMAP.md.
GROWTH_CAPS = {
    # 0.50 = 50% of GDP. Ukraine was already at 39.6% (2025) and still
    # rising, so the cap must sit above that; sustained wartime economies
    # have historically reached this range (UK military spending peaked
    # near 50% of GDP in WW2). See ROADMAP.md for full reasoning.
    "Ukraine": 0.50,
}

# Countries explicitly checked and found NOT to need a growth cap - a
# deliberate, evidence-based decision, distinct from a country simply being
# absent from GROWTH_CAPS (which means "unresolved" and routes to the
# fail-safe below). See ROADMAP.md, 2026-07-24, for why this distinction
# matters: without it, a country's already-made "no cap needed" decision
# would silently turn into an unexplained refusal if it were ever newly
# flagged by calculate_volatility.
NO_CAP_NEEDED = {
    "Türkiye": ("Historical max 4.30% of GDP (1982); the 2018-2025 currency-instability period "
                "(1.6%-2.6%) stays below that - no runaway pattern. See ROADMAP.md."),
}

gdelt_df = pd.read_csv("gdelt_country_year_clusters.csv")

volatilities = {c: calculate_volatility(c) for c in test_countries}
_vol_values = list(volatilities.values())
_vol_threshold = (sum(_vol_values) / len(_vol_values)) + 2 * pd.Series(_vol_values).std()
flagged_countries = [c for c in test_countries if volatilities[c] > _vol_threshold]
print(f"Volatility-flagged countries (>{_vol_threshold:.4f}): {flagged_countries}")

# Computed once per flagged country (not re-derived later) so the automated
# pipeline's LLM calls run at most once per country per training run.
cap_info = {c: determine_cap(c, GROWTH_CAPS, NO_CAP_NEEDED, df, gdelt_df) for c in flagged_countries}
for c in flagged_countries:
    cap, meta = cap_info[c]
    if meta["decision"] == "undetermined":
        print(f"  WARNING: '{c}' is flagged but no cap could be determined (hand-set or automated) "
              f"- left uncapped, unresolved: {meta['note']}")
    elif meta["decision"] == "no_cap_needed":
        print(f"  '{c}': no growth cap needed ({meta['source']}) - {meta['note']}")
    else:
        print(f"  '{c}': cap={cap:.2f} ({meta['source']}, confidence={meta['confidence']})")


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
    cap, cap_meta = cap_info.get(country, (None, None))

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
        # yearly_seasonality=False for every Prophet model, capped or not:
        # this data is already one point per year, so Prophet's default
        # yearly seasonality is a meaningless artifact regardless of
        # whether a growth cap is involved - it was originally only fixed
        # here for capped countries (where it visibly pushed forecasts
        # above their own cap), but it was silently live in the yhat of
        # every other Prophet-chosen country too, just masked there by
        # trend dominance rather than absent. See ROADMAP.md.
        if cap is not None:
            prophet_df["cap"] = cap
            final_model = Prophet(changepoint_prior_scale=result["best_prophet_scale"],
                                   growth="logistic", yearly_seasonality=False)
            final_model.fit(prophet_df[["ds", "y", "cap"]])
            growth_caps_used[country] = cap_meta  # already has "cap", "decision", "source", "confidence", "note"
        else:
            final_model = Prophet(changepoint_prior_scale=result["best_prophet_scale"], yearly_seasonality=False)
            final_model.fit(prophet_df[["ds", "y"]])

    final_models[country] = final_model

joblib.dump(final_models, "final_models.pkl")
print(f"Trained and saved {len(final_models)} final models.")

# "capped", "no_cap_needed", and "flagged" are all saved separately, not
# collapsed into "has a cap entry or doesn't" - main.py needs to tell three
# states apart: capped, explicitly resolved as not needing one (safe to
# serve uncapped), and genuinely unresolved (refuse rather than serve an
# unconfirmed-safe forecast). Collapsing "no_cap_needed" into "missing"
# would have reintroduced the exact gap fixed in this change - see
# scenario_pipeline.py's determine_cap() docstring and ROADMAP.md.
no_cap_needed_used = {c: cap_info[c][1] for c in flagged_countries if cap_info[c][1]["decision"] == "no_cap_needed"}
growth_cap_status = {
    "capped": growth_caps_used,
    "no_cap_needed": no_cap_needed_used,
    "flagged": flagged_countries,
}
with open("growth_caps.json", "w") as f:
    json.dump(growth_cap_status, f, indent=2)
resolved = set(growth_caps_used) | set(no_cap_needed_used)
uncapped_flagged = [c for c in flagged_countries if c not in resolved]
print(f"Saved growth cap status to growth_caps.json: {len(growth_caps_used)} capped, "
      f"{len(no_cap_needed_used)} no-cap-needed, {len(uncapped_flagged)} unresolved ({uncapped_flagged})")

# --- Scenario models (ROADMAP.md step 6, 2026-07-24 reversal): Ukraine and
# Türkiye, the two founding validation cases, each get a hand-designed
# scenario template trained for real, replacing the earlier decision to
# ship Türkiye as caveat-only. Each country's own growth-cap decision
# (already resolved above, whether Ukraine's hand-set cap or Türkiye's
# explicit "no cap needed") is reused here rather than re-derived, so the
# scenario models stay consistent with the single-line models' safety
# gating. See scenario_templates.py.
scenario_models = {}
scenario_status = {}
for country, config in SCENARIO_TEMPLATES.items():
    cap, meta = determine_cap(country, GROWTH_CAPS, NO_CAP_NEEDED, df, gdelt_df)
    model, historical_flag = train_scenario_model(country, df, cap_value=cap)
    scenario_models[country] = {"model": model, "historical_flag": historical_flag, "cap": cap}
    scenario_status[country] = {
        "regressor": config["regressor"],
        "scenario_names": list(config["scenarios"].keys()),
        "magnitude_note": config["magnitude_note"],
        "cap": cap,
        "cap_decision": meta["decision"],
        "cap_source": meta.get("source"),
        "cap_confidence": meta.get("confidence"),
        "cap_note": meta.get("note"),
    }
    print(f"  Scenario model trained for '{country}': {len(config['scenarios'])} named scenarios, "
          f"cap={cap} ({meta['decision']})")

joblib.dump(scenario_models, "scenario_models.pkl")
with open("scenario_status.json", "w") as f:
    json.dump(scenario_status, f, indent=2)
print(f"Saved {len(scenario_models)} scenario models to scenario_models.pkl and scenario_status.json")