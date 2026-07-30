# milex-forecast

A per-country military spending forecasting pipeline, built on real SIPRI data across 31 countries. This project demonstrates handling messy, real-world data (not a clean tutorial dataset), comparing multiple forecasting approaches with actual evidence, and making transparent, defensible modeling decisions in a real world scenario.

## Why I built this

I built this project to go deeper into MLOps and real-world data handling, as a follow-up to an earlier project (iris-mlops-demo) that used a clean, pre-labeled toy dataset. That project taught me the mechanics of the full pipeline: training, serving, containerizing, and testing a model. This one is about the part that project didn't cover: what real, messy data actually looks like, and how to make and defend real modeling decisions when there isn't one obviously correct answer.

For the data pipeline, model comparison, and API, I write and reason through every line of code myself alongside Claude, which explains unfamiliar concepts, walks through why something broke, and pushes back on my assumptions when my reasoning was incomplete, the same way I'd use a mentor or documentation, not as something writing the project for me. Debugging in particular has been mine to work through: tracing an off-by-one header row error, a mixed-up row/column slice, and a duplicate-variable bug that silently ignored a function parameter were all things I had to reason through, not paste in.

The frontend was built differently, and I want to be upfront about that distinction. I used Claude Code as an agentic coding tool to implement the React app, rather than writing it line by line myself, since frontend development was genuinely new territory for me and this project's priority was the ML/data engineering side. My role there was scoping and directing the work: specifying the interaction design, catching a real gap in the original plan (having it inspect the existing backend and README first, and restructure the repo into backend/ and frontend/ without breaking existing tests or CI), diagnosing real bugs from what I saw on screen (a debounce issue causing overlapping API calls, a layout bug causing the modal to resize erratically) and directing the fixes, and reviewing its output rather than assuming it was correct, including having it walk me through, in detail, why an npm audit-flagged vulnerability in a transitive dependency didn't actually apply to how we use the library.

I picked military spending data specifically because I wanted a subject with real, current stakes rather than another synthetic or overused dataset. NATO's 2% GDP spending target is an active, reported policy debate, and I wanted to build something that could speak to it honestly: not a tool that claims to predict geopolitics, but one that shows what the actual historical data says, compares reasonable forecasting approaches on their merits, and is upfront about where the data itself is unreliable or misleading.

## What this project does

Forecasts military spending (as % of GDP) for 31 countries spanning every populated region, using SIPRI's historical expenditure data from 1980 to 2025. Rather than applying one model to every country, it evaluates two different forecasting approaches per country and selects whichever performs better, based on measured error against real held-out data.

## Status

**v1 (`master`): data pipeline, modeling, API, and frontend complete and deployable.**
**v2 (`scenario-forecasting` branch, summarized below): event-aware scenario forecasting, built and live-verified, not yet merged to `master`.** `master` remains the stable, deployable version throughout — v2 is a second phase built on top of it, not a replacement.

## Project structure

```
backend/   FastAPI service, model training, and the data pipeline
frontend/  React app (Vite) - interactive world map + forecast UI
docs/      Detailed working notes for in-progress project phases
```

The two runtime pieces are independently runnable; `frontend/` talks to `backend/` only over HTTP (`/forecast`, `/health`).

## What has been done so far (v1)

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
- **Tuning mattered more than model choice.** At Prophet's default sensitivity setting, linear regression won on every country tested. Only after properly tuning Prophet's changepoint sensitivity did the result flip, with Prophet winning 27 of 31 — evidence that a poorly configured model can lose to a much simpler one, and that a first comparison shouldn't be trusted without checking both sides were fairly tuned.
- **Forecasting models can't predict genuinely novel events.** Ukraine's 2022 spending spike falls inside the test period, not the training period, so both models were effectively guessing blind for that year. Tuning Prophet's sensitivity made no measurable difference here, which is itself informative: no amount of model flexibility can infer an event with zero precedent in its training history.

**4. Model artifacts**

For each of the 31 countries, the winning model configuration was retrained on the full dataset (not just the pre-2015 training slice) and saved. Outputs:
- `cleaned_data.csv` — the cleaned, long-format dataset
- `model_selection_log.json` — a transparent record of which model won for each country, its error score, and the runner-up's score, so every decision is traceable back to real evidence
- `final_models.pkl` — all 31 final trained models, bundled by country name, ready to be loaded and queried

**5. API**

A FastAPI service (`backend/main.py`) loads `final_models.pkl` and serves forecasts:
- `POST /forecast` — takes a country name and a number of years ahead, looks up that country's chosen model, and returns the forecasted values. Returns a `404` for countries without a trained model, and handles unexpected prediction errors with a clean `500` rather than an unhandled crash.
- `GET /health` — a lightweight uptime check, kept separate from the actual prediction logic.

Covered by a pytest suite (a valid forecast request, an invalid country, and the health check), run automatically by GitHub Actions on every push, before the Docker build. Containerized with a Dockerfile and `.dockerignore` that excludes anything the running service doesn't actually need (the raw and cleaned data files, training script, and evaluation log all stay out of the image, since the API only ever needs the final trained models).

**6. Frontend**

A React app (`frontend/`, built with Vite) renders an interactive world map using `react-simple-maps`, highlighting the 31 countries that have a trained model:

- **Country name mapping.** SIPRI's names don't all match the map library's naming (sourced from Natural Earth via `world-atlas`) — e.g. SIPRI's "Türkiye," "Korea, South," and "Viet Nam." Rather than assume the two datasets line up, `frontend/src/data/countryNameMap.js` holds an explicit translation table, and `frontend/src/data/countryData.js` cross-checks every mapped name against the map's actual topojson data at load time, logging a console warning for anything that doesn't resolve. Currently all 31 countries match confidently: 27 by exact name (including "United States of America," which - despite differing conventions elsewhere - happens to already match), and 3 needing the explicit translation (Türkiye → Turkey, Korea, South → South Korea, Viet Nam → Vietnam).
- **Hover** shows a small popup with the country's most recent known spending value (from `cleaned_data.csv`, exported to the frontend as static JSON via `backend/export_country_data.py`).
- **Click** opens a modal (not a new page) with a slider to pick how many years ahead to forecast, calling the live `POST /forecast` endpoint. Results are shown as a line chart (Recharts) combining the last 15 years of observed history with the forecasted years on one continuous line — solid for observed data, dashed for the forecast, sharing a bridge point at the last observed year so the two segments connect with no visual gap. The chart has a fixed size regardless of how many years are selected, so requesting more years plots more points in the same space rather than growing the modal.
- The slider is debounced (waits for the user to pause dragging before calling the API, and discards any in-flight request superseded by a newer one), fixing an earlier bug where dragging fired overlapping requests that raced each other and caused the results to flicker.
- Countries without a trained model are still rendered (so the map looks complete), but are visually distinct (a hairline hatch pattern) and not interactive.
- The visual design is a light, editorial "printed policy report" aesthetic — a warm cream background, ink-navy for observed data and muted burgundy for forecasted data (same cool/observed vs. warm/projected logic carried through the map, hover, and chart), and a serif typeface for body text with sans reserved for UI labels.
- Built with Claude Code (see "Why I built this" above for how this piece was developed differently from the rest of the project).

## v2: event-aware scenario forecasting (`scenario-forecasting` branch, not yet merged)

v1 forecasts every country as a pure function of time. That breaks down for countries whose recent history is dominated by an exceptional event: Ukraine's 2022 invasion, extrapolated blindly 15 years out, pushes forecast spending above 100% of GDP — a number with no real-world meaning. v2 is a second development phase built on top of the working v1 MVP, not a replacement for it. It lives on a separate branch specifically so `master` stays stable and deployable throughout.

**What it adds:**
- **Event data.** GDELT global event data, pulled via BigQuery and aggregated to yearly per-country summaries, normalized against a real artifact in GDELT's own historical source coverage (which grows ~17x between 2006-2016 independent of any real-world trend) so genuine spikes aren't confounded with that ramp.
- **Growth caps.** A flagged country's forecast gets a stated, evidence-grounded ceiling instead of an unconstrained extrapolation.
- **Named scenarios.** Flagged countries get 2-3 named forecasts, each conditional on an explicit future assumption ("conflict continues," "conflict resolves within 3 years," etc.), instead of one number pretending to be certain — standard practice in real economic and policy forecasting.
- **An automated pipeline** (GDELT signal → LLM categorization → template selection → LLM-drafted magnitude, entirely local/free via Ollama + Mistral 7B) intended to extend scenario treatment to future flagged countries without hand-research each time.
- **Honest provenance labeling.** Every cap and scenario carries a `source` (`"hand-researched"` vs. `"automated"`) and `confidence` field, surfaced directly in the API response and as a visible badge in the frontend UI — an automated estimate is never presented with the same confidence as a hand-verified one.
- **A three-way `status` field** on `/forecast` (`"trend"` / `"scenario"` / `"undetermined"`) that the frontend branches on structurally — by status and by which fields are present in the response, never by hardcoded country name — so any future flagged or unflagged country renders correctly with no frontend changes required.
- **A confidence band** for trend-only countries, surfacing Prophet's own 80% prediction interval (already computed, not newly calculated). Scenario countries deliberately don't get one: the named scenarios already represent that uncertainty, and layering a statistical band on top would mix two different kinds of uncertainty rather than clarify anything.

**Two countries have scenario treatment today, both hand-researched founding cases:**
- **Ukraine** — conflict, capped at 50% of GDP (grounded in the UK's real WWII defense-spending peak, ~46-52% of GDP), with a hand-set `conflict_active` regressor.
- **Türkiye** — currency instability, no cap needed (its historical peak, 4.30% of GDP, already exceeds its current crisis-era spending, 1.6-2.6%). This scenario was initially rejected on a cost/benefit calculation — but that rejection was a genuine analytical error, not a case of conditions changing later: the Prophet accuracy cost it was weighed against was a stale, `yearly_seasonality`-bug-corrupted MAE (0.00727) left over from before a separate bug fix, when the correct number (0.00294) was already sitting in the same test output the whole time. Recomputed correctly, the tradeoff flips from looking like a bad trade to a roughly 300x favorable one, and the decision was reversed. It's kept in here, not just the working notes, because it's the clearest instance in this project of the standard it's built on: catching your own mistake by checking the number you actually have, not just re-running the test.

**Headline finding: Türkiye is a real, structural blind spot for both automated detection methods this project tried — not a bug, and not softened by "known limitation" hand-waving.** Currency instability produced no signal under GDELT event-volume detection (Türkiye's 2018 event-volume ratio was 0.77, actually *below* its own baseline — a currency crisis has no discrete reportable event to generate volume around) *and* no signal under spending-volatility detection (Türkiye ranks 15th of 31 countries on year-over-year volatility, solidly mid-pack — a gradual multi-year currency drift doesn't produce the dispersion signature a sudden shock does). Both checks are correctly implemented; the underlying phenomenon simply has neither check's signature. Practical consequence: as built, this pipeline would never have flagged Türkiye for scenario treatment on its own — it has one today only because it was independently, manually researched. Passing an automated anomaly check is evidence a country is unremarkable *by that check's specific definition*, not evidence it's actually unremarkable.

**What's proven on real data versus what's built-and-simulated-but-unverified:**
- **Proven — real, independently-verified evidence:** the automated pipeline's detection and categorization stages were validated with a genuinely blind test against Ukraine, the one country with real, independently known ground truth to check against. The pipeline was calibrated on the other 30 countries, given no hardcoded dates, no country name, and no value reused from any earlier hand-built script; run blind, it correctly identified Ukraine's real conflict episode (2022-2025) and correctly categorized it as "conflict" — checked against the known answer only after every stage had already run.
- **Built and mechanism-tested, but not a real-world validation — don't read these as equivalent to the above:** the same pipeline was also run against Nigeria and Australia. Neither is actually flagged in production; running the pipeline against them proves the mechanism executes end-to-end and produces self-consistent output, nothing more. There is no ground truth to check either result against, and none is claimed. (Nigeria's first run did surface a real bug — a "current episode" that was actually twenty years stale — which was fixed as a result; that's a genuine finding about the pipeline's correctness, not a validation of Nigeria's output.)
- **Deliberately never fully trusted, by design:** the pipeline's LLM-drafted growth-cap magnitude lands in the right neighborhood on the one case with ground truth to check it against (60% of GDP vs. Ukraine's hand-researched 50% — a real ~10 percentage-point gap, stated plainly rather than rounded away) but is never allowed to override a hand-researched value where one exists, and is always labeled `"automated"` with lower confidence in the API rather than presented as equivalent to a verified one.

Full step-by-step decision log — every finding, correction, and reversed decision — lives in [`docs/ROADMAP_v2_working_notes.md`](docs/ROADMAP_v2_working_notes.md).

## What's next

- Merge v2 to `master` after review
- Extend the automated scenario pipeline as/if additional countries cross the volatility threshold (only Ukraine does today; Türkiye is flagged manually)
- Define a "policy shift" scenario template (currently has no validated real example — see working notes)
- Deploy publicly

## Tech stack so far

**v1:**
- pandas, openpyxl for data loading and cleaning
- scikit-learn (LinearRegression) and Prophet for forecasting
- joblib for model persistence
- FastAPI, Pydantic for the API
- pytest, GitHub Actions for testing and CI
- Docker for containerization
- React, Vite, react-simple-maps, Recharts for the frontend

**v2 additions:**
- Google BigQuery for GDELT event data
- sentence-transformers for embeddings, scikit-learn (KMeans) for clustering
- Ollama + Mistral 7B, run locally, for event categorization and scenario drafting

## Running locally

**Backend** (from `backend/`, with its dependencies installed):
```
uvicorn main:app --reload
```
Serves the API at `http://localhost:8000`.

**Frontend** (from `frontend/`):
```
npm install
npm run dev
```
Serves the app at `http://localhost:5173`, configured (via `.env.example` → `VITE_API_BASE_URL`) to call the backend at `http://localhost:8000`.

## Data source

SIPRI Military Expenditure Database, https://www.sipri.org/databases/milex. Used under SIPRI's fair use policy for non-commercial, educational purposes.
