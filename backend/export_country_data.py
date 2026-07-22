"""
Generates frontend/src/data/countrySpending.json from cleaned_data.csv and
model_selection_log.json: the set of countries with a trained model, each
paired with its most recent known spending value (% of GDP), its earliest
and most recent years of available data, and its full historical series
(the /forecast API only returns forecasted years, not history, so the
frontend's forecast chart draws the observed portion of the line from this
file instead).

Re-run this after retraining (train.py) if the underlying data changes:
    python export_country_data.py
"""
import json
from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BACKEND_DIR.parent / "frontend" / "src" / "data" / "countrySpending.json"

df = pd.read_csv(BACKEND_DIR / "cleaned_data.csv").dropna()

with open(BACKEND_DIR / "model_selection_log.json") as f:
    model_log = json.load(f)

country_spending = {}
for country in model_log:
    country_data = df[df["Country"] == country].sort_values("Year")
    if country_data.empty:
        print(f"WARNING: no spending rows for '{country}', skipping")
        continue
    earliest = country_data.iloc[0]
    latest = country_data.iloc[-1]
    country_spending[country] = {
        "earliestYear": int(earliest["Year"]),
        "latestYear": int(latest["Year"]),
        "latestSpending": round(float(latest["Spending"]), 5),
        "history": [
            {"year": int(row["Year"]), "spending": round(float(row["Spending"]), 5)}
            for _, row in country_data.iterrows()
        ],
    }

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(country_spending, f, indent=2)

print(f"Wrote {len(country_spending)} countries to {OUTPUT_PATH}")
