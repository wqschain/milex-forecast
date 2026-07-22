## Roadmap: Event-Aware Scenario Forecasting (v2, `scenario-forecasting` branch)

### The problem this solves

The current model (`master`) forecasts spending as a pure function of time. For countries with smooth, stable trends, this is sound. For countries whose recent history is dominated by an exceptional event, it isn't: a trend-based model has no concept of "this spike is a war, not a permanent trajectory," so it extrapolates the recent slope indefinitely. Concretely, forecasting Ukraine 15 years out currently produces spending above 100% of GDP, a number with no real-world meaning. This is a known, general limitation of trend-only forecasting, not a bug specific to this implementation, but it's worth solving properly rather than just capping the output.

### Why not just add more variables directly

The intuitive fix, "factor in GDP, conflict, neighboring instability, etc.," runs into two real constraints:

1. Each of the 31 current models is trained on only ~46 yearly data points (one country's own history). Adding several input variables to a dataset that small risks overfitting rather than improving the model, the same failure mode discussed in the original Iris project.
2. Forecasting forward requires knowing *future* values of any added inputs (future GDP, future conflict status), not just future dates. Those are themselves unknown, so this isn't a bolt-on fix; it changes what "forecasting" means.

The resolution is scenario-based forecasting: rather than one blind extrapolation, produce a small number of named, clearly-labeled forecasts, each conditional on an explicit assumption about the future (e.g., "conflict continues" vs. "conflict resolves within 3 years"). This is standard practice in real economic and policy forecasting, and it directly fixes the Ukraine problem: different assumptions produce different, individually sensible numbers, instead of one number pretending to be certain.

### Planned pipeline

**1. Event data — GDELT, via BigQuery**
Pull and aggregate GDELT's event data to yearly, per-country summaries for the existing 31 countries, using BigQuery's SQL interface rather than parsing raw event files directly. Decided date range: **1980-2025**, matching the existing SIPRI window exactly, for a clean join on country + year with no gaps to explain (GDELT itself covers back to 1979, but there's no reason to pull the extra year). GDELT's built-in Goldstein scale (an existing intensity/tone score) is a useful starting signal before any custom modeling is layered on top. Google Cloud's free tier (1 TB of query processing/month) comfortably covers this, since the result is aggregated down to a small summary table, not a raw bulk pull.

**2. Embeddings**
Convert each country-year's aggregated event summary into a vector embedding, so country-years can be compared by underlying similarity rather than exact wording. Using `sentence-transformers` (Hugging Face), run locally — no API cost, and the volume here (roughly one embedding per country-year, ~1,400 total) is small enough that a local model is both sufficient and the more practical choice regardless of budget.

**3. Clustering**
Group country-years by embedding similarity (e.g., k-means, via scikit-learn — already a project dependency), surfacing natural groupings — such as a cluster containing Ukraine 2022+ alongside other historically volatile country-years — without predefining categories.

**4. Interpretation (LLM, once per cluster, not once per row)**
Have an LLM review a sample from each cluster and assign it one of four decided categories: **conflict, currency instability, policy shift, stable**. These map directly to the two cases already investigated by hand (Ukraine = conflict, Türkiye = currency instability), giving two of the four categories a built-in validation example; "policy shift" covers cases like a documented change in defense posture without active conflict; "stable" is the default/majority case. Run locally via Ollama with Mistral 7B (confirmed installed and working, model storage on `D:` to avoid `C:` drive constraints) — free, no API cost, and adequate for this bounded classification task. Labels are checked against the two known cases before being trusted on countries without prior manual research; if local labeling accuracy doesn't hold up, a paid API (Claude or OpenAI) is a fallback worth the small cost, given the low call volume (once per cluster, not per country-year).

**5. Feed labels into the existing models**
Every country-year inherits its cluster's label. This becomes a new regressor, added to Prophet via `add_regressor()` (a supported, standard Prophet feature, not a custom hack), alongside the existing time-based fit.

**6. Scenario forecasting**

Decided: a country is flagged for scenario treatment if its volatility (from the existing `calculate_volatility` function) is more than 2 standard deviations above the mean volatility across all 31 countries — a standard statistical definition of an outlier, not an arbitrary cutoff, and one that already correctly flags Türkiye using data already computed earlier in this project. Countries below this threshold keep the current single-line forecast; not every country needs scenario treatment.

Decided scenarios for the two cases already investigated by hand:

- **Ukraine (conflict-driven)**
  - "Conflict continues at current intensity" — conflict flag held active for the full forecast horizon
  - "Conflict resolves within 3 years" — conflict flag active for years 1-3, inactive afterward, spending gradually reverting toward the pre-2022 trend
  - "Gradual de-escalation" — conflict flag steps down in intensity gradually, spending plateauing rather than fully reverting

- **Türkiye (currency-driven)**
  - "Currency instability continues" — continued high inflation/volatility at recent levels
  - "Currency stabilizes" — inflation/volatility reverts toward Türkiye's own pre-2018 baseline within the forecast horizon

Each scenario is grounded in an already-established real event (the 2022 conflict start, the 2018 onset of currency instability), not an invented assumption. Additional countries flagged by the volatility threshold will need their own scenario definitions, following this same pattern (grounded in a real, researched event, not a guess).

### Integration with the existing project

- Data pipeline: extends `backend/`'s existing pattern (raw source → cleaned CSV → model input), adding a second source (GDELT) alongside SIPRI, joined on country + year.
- Modeling: extends the existing per-country `evaluate_country` function rather than replacing it — the same chronological train/test discipline and MAE-based model selection already in place still applies, now evaluated per scenario as well as per model type.
- API: `/forecast` gains an optional scenario parameter for countries where scenarios are defined; countries without a defined scenario continue to behave exactly as they do now.
- Frontend: the existing forecast chart gains the ability to render multiple scenario lines (for flagged countries) instead of one, using the same observed/forecast color convention already established, extended with a scenario selector.
- This is a genuinely separate development phase from the current MVP: real new data source, new dependency (embeddings/clustering/BigQuery), and a re-evaluation of the model selection logic. It is being developed on a separate branch for exactly that reason, so `master` remains a stable, working, deployable version throughout.

### Setup status

All infrastructure for this phase is in place:
- Google Cloud project created, BigQuery API enabled, `gcloud` authenticated locally
- Docker data relocated off `C:` to avoid storage conflicts with this phase's new tooling
- Ollama installed, Mistral 7B pulled and confirmed working (`ollama list`), model storage redirected to `D:`
- `google-cloud-bigquery` and `sentence-transformers` installed in the project venv
- The four scoping decisions above (cluster categories, volatility threshold, Ukraine/Türkiye scenario definitions, GDELT date range) are settled

Next actual step: a first exploratory BigQuery query against GDELT's public dataset, starting with Ukraine, to confirm real data access before building the full pipeline.

### Tooling and cost

- BigQuery: free tier, Google Cloud account required.
- Embeddings: `sentence-transformers`, local, free.
- Cluster labeling: Ollama + Mistral 7B (or Llama 3 8B), local, free. Paid API (Claude/OpenAI) as a fallback only if local labeling accuracy doesn't hold up against the known Ukraine/Türkiye validation cases — a small cost given the low call volume (once per cluster).

### Scope notes

- This phase pulls GDELT data once, as a static historical enrichment (matching how SIPRI's data is already handled: pulled once, cleaned, saved, worked from). A live, continuously-updating version is a legitimate later extension, not part of this phase.
- Scenario construction is deliberately scoped to countries with demonstrated high volatility, not applied uniformly, to avoid manufacturing false complexity for countries whose current single-line forecast is already reasonable.