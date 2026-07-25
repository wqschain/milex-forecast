from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
from sklearn.linear_model import LinearRegression
import pandas as pd
from pydantic import BaseModel
import logging
import json
from scenario_templates import SCENARIO_TEMPLATES, forecast_scenarios
from scenario_pipeline import forecast_automated_scenarios

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

models = joblib.load("final_models.pkl")

# Countries whose Prophet model was fit with growth="logistic" (see
# train.py's GROWTH_CAPS / ROADMAP.md step 5 round 2) - predict() requires a
# "cap" column on every dataframe passed in, not just at training time.
# "capped" and "no_cap_needed" are both explicit resolutions (a real cap
# value, or a deliberate "checked, doesn't need one" decision); "flagged"
# (volatility outliers) is tracked separately so a country that's flagged
# but NEITHER resolution applies to yet can be refused explicitly below,
# instead of silently falling through to an unconfirmed-safe forecast. See
# scenario_pipeline.py's determine_cap() docstring and ROADMAP.md,
# 2026-07-24, for why collapsing "no_cap_needed" into "missing" was a real
# gap, not just an implementation detail.
with open("growth_caps.json") as f:
    _growth_cap_status = json.load(f)
growth_caps = _growth_cap_status["capped"]
no_cap_needed = _growth_cap_status["no_cap_needed"]
flagged_countries = set(_growth_cap_status["flagged"])

# Hand-designed scenario templates (Ukraine: conflict, Türkiye: currency
# instability - see scenario_templates.py, ROADMAP.md step 6). Both were
# proven out standalone (test_scenarios_ukraine.py, test_scenarios_turkiye.py)
# before being trained for real in train.py and wired in here.
scenario_models = joblib.load("scenario_models.pkl")
with open("scenario_status.json", encoding="utf-8") as f:
    scenario_status = json.load(f)


class ForecastRequest(BaseModel):
    country: str
    years_ahead: int


def _single_line_forecast(model, years_ahead, cap_entry):
    """Shared by both status="trend" and status="undetermined": the same
    model object and the same cap-application code path every time, so a
    country that's flagged-but-unresearched (undetermined) can never reach
    an unbounded forecast through a second, separate path - see
    ROADMAP.md's two-gate design."""
    if isinstance(model, LinearRegression):
        future_years = pd.DataFrame({"Year": range(2026, 2026 + years_ahead)})
        return model.predict(future_years).tolist()
    future = model.make_future_dataframe(periods=years_ahead, freq="YE")
    if cap_entry is not None:
        future["cap"] = cap_entry["cap"]
    forecast = model.predict(future)
    return forecast["yhat"].tail(years_ahead).tolist()


@app.post("/forecast")
def forecast_spending(request: ForecastRequest):
    country = request.country

    if country not in models and country not in scenario_models:
        raise HTTPException(status_code=404, detail=f"No model available for '{country}'")

    try:
        # status="scenario": a hand-built template exists (currently
        # Ukraine, Türkiye). All named scenarios are returned together, not
        # one per request - the frontend use case (multiple lines on one
        # chart) needs the complete picture in a single round-trip.
        #
        # Checked first, before the flagged_countries/growth_caps gate
        # below - deliberately, not an oversight. That gate protects the
        # generic single-line path, which cross-references growth_caps.json
        # at serving time. A scenario model's cap decision is resolved and
        # baked in once, at training time (train.py always calls
        # determine_cap() before training the scenario model - see the
        # "Scenario models" block there), so there's nothing to re-verify
        # per request: the model itself already reflects a resolved
        # decision, capped or explicitly not needing one. Verified directly
        # (2026-07-24): emptied growth_caps.json's "capped"/"no_cap_needed"
        # entirely and confirmed Ukraine still serves correctly (still
        # scenario status 200) while a genuinely unresolved flagged country
        # with no scenario template still gets refused (503) via the gate
        # below.
        if country in scenario_models:
            entry = scenario_models[country]
            status_meta = scenario_status[country]
            # Two different forecast functions depending on provenance: a
            # hand-built country (Ukraine, Türkiye) has a fixed
            # SCENARIO_TEMPLATES entry to look its flag schedules up from;
            # an automated country has none - its schedules were LLM-
            # constructed at training time and persisted as "scenario_spec"
            # in scenario_models.pkl (not scenario_status.json - it's
            # inference config, not human/API-facing status). The frontend
            # never sees this dispatch - both produce the identical
            # {scenario_name: [...]} response shape.
            if country in SCENARIO_TEMPLATES:
                scenarios = forecast_scenarios(
                    country, entry["model"], entry["historical_flag"], request.years_ahead, cap_value=entry["cap"]
                )
            else:
                scenarios = forecast_automated_scenarios(
                    entry["model"], entry["historical_flag"], entry["scenario_spec"],
                    request.years_ahead, cap_value=entry["cap"]
                )
            response = {
                "country": country,
                "status": "scenario",
                "scenarios": scenarios,
                "magnitude_note": status_meta["magnitude_note"],
                # Never presented with equal certainty: a hand-built
                # scenario (individually researched, matching Ukraine's/
                # Türkiye's conflict_active/currency_active provenance) and
                # an automated one (GDELT-only, not independently
                # cross-checked) must be visibly distinguishable - same
                # principle as cap_source/cap_confidence. See ROADMAP.md.
                "scenario_source": status_meta["scenario_source"],
                "scenario_confidence": status_meta["scenario_confidence"],
            }
            if status_meta["cap_decision"] == "capped":
                response["cap_source"] = status_meta["cap_source"]
                response["cap_confidence"] = status_meta["cap_confidence"]
                response["cap_note"] = status_meta["cap_note"]
            return response

        resolved_cap = growth_caps.get(country) or no_cap_needed.get(country)

        if country in flagged_countries and resolved_cap is None:
            # Fail loudly rather than silently serve an unconfirmed-safe
            # forecast for a country identified as needing resolution
            # (growth cap or an explicit "no cap needed" decision) that
            # doesn't have one yet - same "explicit unresolved state, never
            # a silent default" discipline as the undetermined-by-GDELT
            # branch (see ROADMAP.md).
            logger.error(f"'{country}' is volatility-flagged with no resolved growth-cap decision - refusing "
                         f"to serve a forecast rather than risk an unbounded extrapolation.")
            raise HTTPException(
                status_code=503,
                detail=f"'{country}' is flagged for scenario treatment but has no growth-cap decision resolved "
                       f"yet (neither a cap value nor an explicit 'no cap needed' determination). Refusing to "
                       f"serve a forecast for a country known to need this resolved first."
            )

        model = models[country]
        result = _single_line_forecast(model, request.years_ahead, growth_caps.get(country))

        if country in flagged_countries:
            # Flagged, and growth-cap-safe (capped or no_cap_needed), but
            # with no hand-built scenario template - "undetermined" is
            # reserved specifically for this: a genuine lack-of-research
            # state, not a lack-of-automated-detection state (see
            # ROADMAP.md's 2026-07-23 clarification). No live example
            # exists today (Ukraine, the only flagged country, has a
            # template) - this path is built and verified but has not yet
            # had a real country reach it.
            response = {
                "country": country, "status": "undetermined", "trend_forecast": result,
                "flagged_reason": "volatility outlier (calculate_volatility, >2 std dev above mean)",
                "detection_status": "Flagged and growth-cap-safe, but no scenario template has been built for "
                                     "this country yet - needs individual research, the same way Ukraine and "
                                     "Türkiye originally got (see ROADMAP.md step 4/6).",
            }
        else:
            response = {"country": country, "status": "trend", "forecast": result}

        if country in growth_caps:
            entry = growth_caps[country]
            response["cap_source"] = entry["source"]
            response["cap_confidence"] = entry["confidence"]
            response["cap_note"] = entry["note"]
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.get("/health")
def health_check():
    return {"status": "ok"}
