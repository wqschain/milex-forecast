"""
Generates frontend/src/data/facts.json: short, factual one-line items for the
frontend's fact ticker. Two sources, concatenated:

1. Static findings straight from README.md's "Notable findings" section
   (qualitative - not derivable from the data itself, so they're transcribed
   here by hand and should be kept in sync with the README).
2. Computed 5-year-ahead spending projections, using the exact same
   forecast_spending() logic the live API serves, for the countries with the
   largest projected relative change. Countries whose projection would cross
   into negative spending (a linear-regression extrapolation artifact, not a
   real result) are excluded before ranking.

Re-run this after retraining (train.py) or after editing README.md's
findings section:
    python export_facts.py
"""
import json
from pathlib import Path

from main import ForecastRequest, forecast_spending

BACKEND_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BACKEND_DIR.parent / "frontend" / "src" / "data" / "facts.json"
FORECAST_HORIZON_YEARS = 5
STANDOUT_COUNT = 3

# Transcribed from README.md "Notable findings" - update both places together.
STATIC_FINDINGS = [
    "TÜRKIYE — data volatility ~5x average, traced to the lira losing 80%+ of its value since 2018, not a policy change",
    "NORTH KOREA — excluded from the dataset: no reliable spending data reported for the study period",
    "MODEL SELECTION — no single approach wins universally; Prophet outperformed on 27 of 31 countries",
    "TUNING — at default sensitivity, linear regression beat Prophet on every country tested",
    "UKRAINE 2022 — the spending spike fell in the test period; both models were blind to it",
]


def compute_standout_projections():
    with open(BACKEND_DIR / "model_selection_log.json") as f:
        countries = list(json.load(f).keys())
    with open(BACKEND_DIR.parent / "frontend" / "src" / "data" / "countrySpending.json") as f:
        spending = json.load(f)

    results = []
    for country in countries:
        latest = spending[country]["latestSpending"]
        response = forecast_spending(ForecastRequest(country=country, years_ahead=FORECAST_HORIZON_YEARS))
        projected = response["forecast"][-1]
        if projected < 0:
            # Linear regression extrapolating a decline past zero - a modeling
            # artifact, not a real result. Excluded rather than reported.
            continue
        pct_change = (projected - latest) / latest * 100
        target_year = spending[country]["latestYear"] + FORECAST_HORIZON_YEARS
        results.append((country, latest, projected, pct_change, target_year))

    results.sort(key=lambda r: abs(r[3]), reverse=True)
    return results[:STANDOUT_COUNT]


def format_projection_fact(country, latest, projected, pct_change, target_year):
    direction = "rising" if pct_change > 0 else "falling"
    return (
        f"{country.upper()} — model trend projects spending {direction} from "
        f"{latest * 100:.1f}% to {projected * 100:.1f}% of GDP by {target_year}"
    )


standout = compute_standout_projections()
projection_facts = [format_projection_fact(*row) for row in standout]

facts = STATIC_FINDINGS + projection_facts

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(facts, f, indent=2, ensure_ascii=False)

print(f"Wrote {len(facts)} facts to {OUTPUT_PATH}")
for fact in facts:
    print(" -", fact)
