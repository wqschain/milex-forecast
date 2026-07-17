# milex-forecast

A per-country military spending forecasting pipeline, built on real SIPRI data across 31 countries. This project demonstrates handling messy, real-world data (not a clean tutorial dataset), comparing multiple forecasting approaches with actual evidence, and making transparent, defensible modeling decisions in a real world scenario.

## Why I built this

I built this project to go deeper into MLOps and real-world data handling, as a follow-up to an earlier project (iris-mlops-demo) that used a clean, pre-labeled toy dataset. That project taught me the mechanics of the full pipeline: training, serving, containerizing, and testing a model. This one is about the part that project didn't cover: what real, messy data actually looks like, and how to make and defend real modeling decisions when there isn't one obviously correct answer.

I write and reason through every line of code myself alongside Claude to explain unfamiliar concepts, walk through why something broke, and push back on my assumptions when my reasoning was incomplete, the same way I'd use a mentor or documentation, not as something writing the project for me. Debugging in particular has been mine to work through: tracing an off-by-one header row error, a mixed-up row/column slice, and a duplicate-variable bug that silently ignored a function parameter were all things I had to reason through, not paste in.

I picked military spending data specifically because I wanted a subject with real, current stakes rather than another synthetic or overused dataset. NATO's 2% GDP spending target is an active, reported policy debate, and I wanted to build something that could speak to it honestly: not a tool that claims to predict geopolitics, but one that shows what the actual historical data says, compares reasonable forecasting approaches on their merits, and is upfront about where the data itself is unreliable or misleading.

## What this project does

Forecasts military spending (as % of GDP) for 31 countries spanning every populated region, using SIPRI's historical expenditure data from 1980 to 2025. Rather than applying one model to every country, it evaluates two different forecasting approaches per country and selects whichever performs better, based on measured error against real held-out data.

## Status: data pipeline and modeling complete. API and frontend not yet built.

## What has been done so far

**1. Data cleaning (`sipri_data.xlsx` to `cleaned_data.csv`)**

Loaded SIPRI's real, messy Excel database, which required:
- Correctly identifying the actual header row buried under title and methodology text
- Filtering to 32 selected countries spanning North/South America, Europe, the Middle East, Asia, Africa, and Oceania, resolving real naming inconsistencies (SIPRI uses "Türkiye," "Korea, South," "Viet Nam")
- Converting text placeholder codes ("xxx" = country did not exist, ".." = data unavailable) into proper missing values, without conflating their different meanings
- Reshaping from wide format (one column per year) to long format (one row per country-year)
- Filtering to 1980 onward
- Filling gaps using linear interpolation, applied separately per country and in correct chronological order, so gaps are only bridged between two known real values, never guessed at the edges

**2. Model comparison**

Built and evaluated two forecasting approaches per country, using a chronological train/test split (train on 1980-2015, test against the real, known 2016-2025 values, since a random split would let the model "see the future"):

- **Linear regression** — a simple baseline, fitting one straight-line trend across all historical data
- **Prophet** — a trend-detection model capable of adapting to genuine shifts in a country's spending trajectory, tuned across multiple changepoint sensitivity settings (0.05, 0.3, 0.8)

Initial testing with Prophet at default settings showed linear regression winning on every test. Re-testing with Prophet properly tuned reversed this: Prophet won on 27 of 31 countries, several by a wide margin.

**3. Notable findings**

- **Türkiye's data is an outlier by roughly 5x on year-over-year volatility.** Investigated and traced this to the Turkish lira losing over 80% of its value against the dollar since 2018, driven by chronic high inflation, not a genuine defense-policy shift. Since spending is measured as a share of GDP in dollar terms, currency instability directly distorts the metric. Linear regression's simple averaging handled this noise better than Prophet, which tried to fit it as a real trend.
- **North Korea was excluded from the country list entirely.** It has no reliable, publicly reported military spending data for almost the entire study period, so there is nothing meaningful to forecast from.
- **No single model or setting works best universally.** The right choice depends on whether a country's history has genuine, discrete shifts (a war, a policy change) or a smoother, gradual trend.

**4. Final output**

For each of the 31 countries, the winning model configuration was retrained on the full dataset (not just the pre-2015 training slice) and saved. Outputs:
- `cleaned_data.csv` — the cleaned, long-format dataset
- `model_selection_log.json` — a transparent record of which model won for each country, its error score, and the runner-up's score, so every decision is traceable back to real evidence
- `final_models.pkl` — all 31 final trained models, bundled by country name, ready to be loaded and queried

## What's next

- Build a FastAPI service that loads `final_models.pkl` and serves forecasts through a `/forecast` endpoint, keyed by country
- Add input validation, error handling, and tests, following the same practices used in a prior project (iris-mlops-demo)
- Containerize with Docker and set up CI, same as the prior project
- Build a frontend to visualize spending trends and forecasts across countries, likely with a curated set of real historical events (an embargo, a conflict) marked on the timeline as honest context, not a causal claim
- Deploy publicly

## Tech stack so far

- pandas, openpyxl for data loading and cleaning
- scikit-learn (LinearRegression) and Prophet for forecasting
- joblib for model persistence

## Data source

SIPRI Military Expenditure Database, https://www.sipri.org/databases/milex. Used under SIPRI's fair use policy for non-commercial, educational purposes.