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
with open("growth_caps.json") as f:
    growth_caps = json.load(f)


class ForecastRequest(BaseModel):
    country: str
    years_ahead: int


@app.post("/forecast")
def forecast_spending(request: ForecastRequest):
    if request.country not in models:
        raise HTTPException(status_code=404, detail=f"No model available for '{request.country}'")

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
        
            
        return {"country": request.country, "forecast": result}

    except Exception as e: 
                    # catching unexpected errors and handling it cleanly
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail="Prediction failed")



@app.get("/health")
def health_check():
    return {"status": "ok"}