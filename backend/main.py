from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
from sklearn.linear_model import LinearRegression
import pandas as pd
from pydantic import BaseModel
import logging
import json

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
# "flagged" (all volatility-outlier countries) is tracked separately from
# "capped" (the subset that actually got a growth cap applied) so a
# country that's flagged but not yet capped can be refused explicitly
# below, instead of silently falling through to an uncapped forecast.
with open("growth_caps.json") as f:
    _growth_cap_status = json.load(f)
growth_caps = _growth_cap_status["capped"]
flagged_countries = set(_growth_cap_status["flagged"])

# Countries with a known, researched anomaly whose future magnitude isn't
# reliably estimable from available data - a qualitative caveat attached to
# the country's normal production forecast, not a second model or a set of
# named scenarios. See ROADMAP.md step 6, "Türkiye scenario proof-of-concept":
# building a scenario mechanism for Türkiye would have meant abandoning its
# actual best-fitting model (linear regression, MAE 0.00293) for a worse one
# (Prophet, MAE 0.00727) to gain a regressor whose effect (~0.3pp) turned out
# to be the same order of magnitude as the accuracy cost of doing so - not a
# worthwhile trade, unlike Ukraine's growth cap/conflict flag. Independent of
# the growth-cap mechanism above: Türkiye isn't in the volatility-flagged
# list (it was identified by an earlier, different method - see ROADMAP.md),
# so this doesn't interact with or bypass that fail-safe.
FORECAST_CAVEATS = {
    "Türkiye": (
        "Known currency instability since 2018 (lira depreciation) is not reflected in this "
        "trend forecast - its future magnitude could not be reliably estimated from available "
        "data. See ROADMAP.md, step 6, for the full investigation."
    ),
}


class ForecastRequest(BaseModel):
    country: str
    years_ahead: int


@app.post("/forecast")
def forecast_spending(request: ForecastRequest):
    if request.country not in models:
        raise HTTPException(status_code=404, detail=f"No model available for '{request.country}'")

    if request.country in flagged_countries and request.country not in growth_caps:
        # Fail loudly rather than silently serve an unbounded extrapolation
        # for a country that was identified as needing a growth cap but
        # doesn't have one configured yet - same "explicit unresolved state,
        # never a silent default" discipline as the undetermined-by-GDELT
        # branch (see ROADMAP.md).
        logger.error(f"'{request.country}' is volatility-flagged but has no growth cap configured - refusing "
                     f"to serve a forecast rather than risk an unbounded extrapolation.")
        raise HTTPException(
            status_code=503,
            detail=f"'{request.country}' is flagged for scenario treatment but has no growth cap configured yet. "
                   f"Refusing to serve an uncapped forecast for a country known to need one."
        )

    model = models[request.country]


    try:
        if isinstance(model, LinearRegression):
            future_years = pd.DataFrame({
                "Year": range(2026, 2026 + request.years_ahead)
            })
            predictions = model.predict(future_years)
            result = predictions.tolist()
        else:
            future = model.make_future_dataframe(periods=request.years_ahead, freq="YE")
            if request.country in growth_caps:
                future["cap"] = growth_caps[request.country]
            forecast = model.predict(future)
            result = forecast["yhat"].tail(request.years_ahead).tolist()
        
            
        response = {"country": request.country, "forecast": result}
        if request.country in FORECAST_CAVEATS:
            response["caveat"] = FORECAST_CAVEATS[request.country]
        return response

    except Exception as e: 
                    # catching unexpected errors and handling it cleanly
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail="Prediction failed")



@app.get("/health")
def health_check():
    return {"status": "ok"}