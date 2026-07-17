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

for country, result in results_log.items():
    if "error" in result:
        continue
    
    country_data = df[df["Country"] == country].dropna()
    
    if result["chosen_model"] == "linear":
        X = country_data[["Year"]]
        y = country_data["Spending"]
        final_model = LinearRegression()
        final_model.fit(X, y)
    else:
        prophet_df = country_data[["Year", "Spending"]].copy()
        prophet_df["ds"] = pd.to_datetime(prophet_df["Year"], format="%Y")
        prophet_df["y"] = prophet_df["Spending"]
        prophet_df = prophet_df[["ds", "y"]]
        final_model = Prophet(changepoint_prior_scale=result["best_prophet_scale"])
        final_model.fit(prophet_df)
    
    final_models[country] = final_model

joblib.dump(final_models, "final_models.pkl")
print(f"Trained and saved {len(final_models)} final models.")