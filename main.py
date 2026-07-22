from fastapi import FastAPI, HTTPException
import joblib
from sklearn.linear_model import LinearRegression
import pandas as pd
from pydantic import BaseModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
models = joblib.load("final_models.pkl")


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