"""
Hand-designed scenario templates for the two founding validation cases
(Ukraine: conflict, Türkiye: currency instability - see docs/ROADMAP_v2_working_notes.md step 6).

Distinct from scenario_pipeline.py's automated GDELT pipeline: these two
countries were individually researched and hand-built (matching how
Ukraine's was proven out in test_scenarios_ukraine.py and Türkiye's in
test_scenarios_turkiye.py), not derived automatically. A future flagged
country without this kind of hand research routes through
scenario_pipeline.py's automated fallback instead, which now builds a full
scenario template too (construct_scenarios(), not just a growth cap value) -
but a hand-built entry here always takes priority over it, and every
automated result is tagged "source": "automated" so it's never presented
with the same confidence as these two hand-researched templates.

Each template's regressor magnitude is a deliberate, documented human
judgment call, not a statistically fitted coefficient - both countries'
backtests are structurally invalid (the regressor's "on" state has zero
training examples before the real event/crisis begins - see
test_regressor_ukraine_v2.py's Result 1 and test_scenarios_turkiye.py's
Question 1), so there is nothing for Prophet to learn an effect size from.
This is the same category as the scenario templates themselves: a
hand-set assumption about what "conflict continues" or "currency
instability continues" should mean quantitatively, not a number the data
determined on its own. Recorded in `magnitude_note` below for direct
surfacing in the API response, not left implicit.
"""
import numpy as np
import pandas as pd
from prophet import Prophet

SCENARIO_TEMPLATES = {
    "Ukraine": {
        "regressor": "conflict_active",
        "changepoint_prior_scale": 0.8,  # Ukraine's originally-selected Prophet scale
        "historical_flag_year": 2022,  # full-scale invasion - the unambiguous regime shift in real spending data
        "scenarios": {
            "Conflict continues at current intensity": lambda n: np.ones(n),
            "Conflict resolves within 3 years": lambda n: np.concatenate(
                [np.ones(min(3, n)), np.zeros(max(0, n - 3))]
            ),
            # Linear decline over 6 years, not a second binary schedule -
            # deliberately longer than "resolves within 3 years" so it
            # represents a genuinely different assumption (a protracted
            # wind-down), not the same shape with a smoothed edge. See
            # test_scenarios_ukraine.py's docstring.
            "Gradual de-escalation": lambda n: np.clip(1 - np.arange(n) / 6, 0, 1),
        },
        "magnitude_note": (
            "conflict_active's magnitude is a hand-set assumption (flag=1 for 2022+, sourced "
            "from known history), not a statistically fitted coefficient - the dataset contains "
            "exactly one real conflict occurrence, insufficient to identify an effect size from "
            "data alone. See docs/ROADMAP_v2_working_notes.md."
        ),
    },
    "Türkiye": {
        "regressor": "currency_active",
        "changepoint_prior_scale": 0.05,  # Türkiye's best-tested Prophet scale per model_selection_log.json
        "historical_flag_year": 2018,  # documented lira depreciation onset
        "scenarios": {
            "Currency instability continues": lambda n: np.ones(n),
            # Linear decline over 4 years (not a sharp cutoff): a currency
            # crisis resolves as a multi-year credibility-rebuilding
            # process, not a discrete event - grounded in real 1992 European
            # ERM crisis recovery times (UK ~3yr, Sweden ~4yr). See
            # test_scenarios_turkiye.py's docstring.
            "Currency stabilizes": lambda n: np.clip(1 - np.arange(n) / 4, 0, 1),
        },
        "magnitude_note": (
            "currency_active's magnitude is a hand-set assumption (flag=1 for 2018+, sourced "
            "from the lira's documented depreciation), not a statistically fitted coefficient - "
            "the backtest is structurally invalid (zero pre-2016 training examples of the flag "
            "being active). See docs/ROADMAP_v2_working_notes.md."
        ),
    },
}


def train_scenario_model(country, spending_df, cap_value=None):
    """
    Fits the country's scenario-specific Prophet model: its own named
    regressor, at its own changepoint_prior_scale, growth-capped if
    cap_value is given (None means this country's growth cap decision was
    "no cap needed" - see train.py's NO_CAP_NEEDED). Returns the fitted
    model and the historical regressor series as a numpy array, needed to
    build each scenario's future dataframe positionally (see
    test_scenarios_ukraine.py's docstring for why: make_future_dataframe's
    appended dates are year-end, not year-start, so a date-based merge
    would silently produce NaN).
    """
    config = SCENARIO_TEMPLATES[country]
    country_data = spending_df[spending_df["Country"] == country][["Year", "Spending"]].dropna()
    country_data = country_data.sort_values("Year").reset_index(drop=True)
    country_data["ds"] = pd.to_datetime(country_data["Year"], format="%Y")
    country_data["y"] = country_data["Spending"]
    country_data[config["regressor"]] = (country_data["Year"] >= config["historical_flag_year"]).astype(int)

    prophet_kwargs = dict(changepoint_prior_scale=config["changepoint_prior_scale"], yearly_seasonality=False)
    cols = ["ds", "y", config["regressor"]]
    if cap_value is not None:
        country_data["cap"] = cap_value
        cols.append("cap")
        prophet_kwargs["growth"] = "logistic"

    model = Prophet(**prophet_kwargs)
    model.add_regressor(config["regressor"])
    model.fit(country_data[cols])
    return model, country_data[config["regressor"]].values


def forecast_scenarios(country, model, historical_flag, years_ahead, cap_value=None):
    """Returns {scenario_name: [forecast values]} for every named scenario in this country's template."""
    config = SCENARIO_TEMPLATES[country]
    results = {}
    for name, flag_fn in config["scenarios"].items():
        future = model.make_future_dataframe(periods=years_ahead, freq="YE")
        future[config["regressor"]] = np.concatenate([historical_flag, flag_fn(years_ahead)])
        if cap_value is not None:
            future["cap"] = cap_value
        forecast = model.predict(future)
        results[name] = forecast["yhat"].tail(years_ahead).tolist()
    return results
