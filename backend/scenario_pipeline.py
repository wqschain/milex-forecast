"""
Automated event-aware scenario pipeline (docs/ROADMAP_v2_working_notes.md step 6 rebuild).

Replaces the hand-set `conflict_active` flag (a literal hardcoded date
range that never updates) with an end-to-end automated pipeline: detection
-> GDELT-derived signal -> categorization (embeddings/clustering + LLM) ->
template selection -> LLM-drafted magnitude. Growth cap application
(train.py's GROWTH_CAPS mechanism, the two-gate composition design) is
unchanged - this module produces a *value* a cap could use, it doesn't
touch how caps are applied.

Wired into train.py (as the fallback in determine_cap()/determine_scenario()
for any flagged country with no hand-built entry) and main.py (dispatching
to forecast_automated_scenarios() for automated scenario models) - but only
after passing the end-to-end blind test in backend/blind_test_ukraine.py,
kept as the record of that validation.

Note: `calculate_volatility` and `test_countries` are reimplemented here
identically to train.py rather than imported from it, because train.py has
no `if __name__ == "__main__":` guard - it executes the full training loop
(fits all 31 countries, overwrites model_selection_log.json/final_models.pkl)
as top-level module code, so importing it would silently re-run training
just to get one small function.
"""
import re
import requests
import numpy as np
import pandas as pd
from prophet import Prophet

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral:latest"

test_countries = [
    "United States of America", "Canada", "Mexico", "Brazil", "Argentina", "Colombia",
    "United Kingdom", "Germany", "France", "Italy", "Poland", "Russia", "Ukraine",
    "Israel", "Saudi Arabia", "Türkiye", "Iran", "China", "India", "Japan",
    "Korea, South", "Pakistan",
    "Australia", "Indonesia", "Viet Nam", "South Africa", "Nigeria", "Egypt", "Algeria",
    "Sweden", "Norway"
]


# ---------- Stage 1: detection (already working - reused identically, not reimplemented differently) ----------
def calculate_volatility(spending_df, country_name):
    country_data = spending_df[spending_df["Country"] == country_name].dropna().sort_values("Year")
    return country_data["Spending"].pct_change().std()


def detect_flagged_countries(spending_df):
    volatilities = {c: calculate_volatility(spending_df, c) for c in test_countries}
    vals = pd.Series(list(volatilities.values()))
    threshold = vals.mean() + 2 * vals.std()
    flagged = [c for c in test_countries if volatilities[c] > threshold]
    return flagged, threshold, volatilities


# ---------- Stage 2: GDELT-derived elevated-baseline signal ----------
def compute_elevated_signal(gdelt_df, country):
    """
    baseline_ratio = mention_share / this country's own long-run (median,
    robust) share - a FIXED reference point, not spike_ratio's rolling
    trailing window. That's the deliberate fix: a rolling window gets
    contaminated by the anomaly's own recent years, pulling the "baseline"
    up to match it and making the ratio decay back toward 1.0 even while
    the real anomaly is still ongoing (see docs/ROADMAP_v2_working_notes.md's spike_ratio
    finding). A baseline fixed once, from the country's full history, isn't
    dragged upward by a handful of elevated years the way a rolling window
    is - median is used specifically because it's robust to those years
    distorting the reference point at all.

    Threshold: mean + 2*std of baseline_ratio across the other 30
    countries - the same statistical convention calculate_volatility
    already uses, calibrated on OTHER countries' data before this
    country's own results are computed, specifically so the threshold
    can't be tuned to land on any particular known outcome for this
    country.
    """
    df = gdelt_df.copy()
    long_run_baseline = df.groupby("Country")["mention_share"].median()
    df["long_run_baseline"] = df["Country"].map(long_run_baseline)
    df["baseline_ratio"] = df["mention_share"] / df["long_run_baseline"]

    calib = df[df["Country"] != country]["baseline_ratio"]
    threshold = calib.mean() + 2 * calib.std()

    country_df = df[df["Country"] == country][
        ["Year", "mention_share", "baseline_ratio", "avg_tone", "avg_goldstein"]
    ].sort_values("Year").reset_index(drop=True)
    country_df["elevated_active"] = (country_df["baseline_ratio"] > threshold).astype(int)
    return country_df, threshold


def find_current_episode(signal_df, max_gap_years=2):
    """The most recent contiguous run of elevated_active=1 years is the
    'current' episode driving forward scenario construction; any earlier,
    separate elevated years are historical precedent for stage 5's
    magnitude reasoning, not part of what's being forecast forward.

    Recency check (added 2026-07-24, a fix, not a documented deferral -
    see docs/ROADMAP_v2_working_notes.md): the most recent block's last year must be within
    `max_gap_years` of this signal_df's own latest year, or it isn't
    treated as "current" at all - both current_years and precedent_years
    come back empty, routing the caller to undetermined. Without this,
    the most-recent-ever elevated block was returned regardless of how
    long ago it ended - caught when Nigeria's block (2002-2005, twenty
    years stale) was confidently labeled "current" and used to construct
    forward-looking scenarios as if it were ongoing. Ukraine's validated
    result is unaffected (its block ends in 2025, the dataset's own last
    year, gap=0). max_gap_years=2 is a round, generally-applicable choice
    (allows a one-to-two-year lull without dropping a genuinely ongoing
    situation), not tuned to reproduce any specific country's outcome.
    """
    elevated_years = signal_df.loc[signal_df["elevated_active"] == 1, "Year"].tolist()
    if not elevated_years:
        return [], []
    current = [elevated_years[-1]]
    for y in reversed(elevated_years[:-1]):
        if current[0] - y <= 1:
            current.insert(0, y)
        else:
            break

    latest_year = signal_df["Year"].max()
    if latest_year - current[-1] > max_gap_years:
        return [], []  # most recent elevated block is stale, not ongoing - undetermined, not guessed

    precedent = [y for y in elevated_years if y not in current]
    return current, precedent


# ---------- Ollama call (shared by stages 3 and 5) ----------
def call_ollama(prompt, temperature=0.2):
    resp = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": temperature},
    }, timeout=180)
    resp.raise_for_status()
    return resp.json()["response"]


# ---------- Stage 3: categorization ----------
def categorize_episode(signal_df, current_years):
    """
    Anonymized deliberately: no country name and no real calendar year is
    ever sent to the LLM, only the numeric pattern (relabeled "Year 1",
    "Year 2", ...) - so a correct category has to come from the data
    itself, not the LLM recalling a specific real-world event by name or
    date from its training data. This is what makes the categorization
    step a genuine test of the pipeline rather than the model's memory.

    Deliberately omits avg_goldstein: docs/ROADMAP_v2_working_notes.md's step 1 finding (written
    before this pipeline existed, not derived from this test) already
    established that yearly-averaged Goldstein is diluted by hundreds of
    thousands of routine events per country-year and can be misleading at
    this granularity (e.g. Ukraine's real 2022 avg_goldstein reads as less
    negative than its 2018 value). Share-normalized volume and avg_tone are
    the features that finding identifies as the reliable signal - this
    prompt uses only those two, consistent with a pre-existing, general
    finding rather than anything derived from this specific test's outcome.
    """
    rows = signal_df[signal_df["Year"].isin(current_years)].sort_values("Year")
    lines = []
    for i, (_, row) in enumerate(rows.iterrows(), 1):
        lines.append(
            f"Year {i}: event volume {row['baseline_ratio']:.1f}x this location's own long-run typical level, "
            f"average tone {row['avg_tone']:.2f} (negative = critical/hostile press coverage, positive = "
            f"neutral-to-favorable)."
        )
    data_block = "\n".join(lines)

    prompt = f"""You are analyzing anonymized event-monitoring data for one location over a multi-year period of unusually elevated news/event volume. You are NOT told the location's name or the real calendar years - reason only from the numbers below.

{data_block}

Classify the underlying situation into exactly one category:
- conflict (war, invasion, sustained armed violence)
- currency instability (economic/financial crisis, inflation, currency collapse)
- other (anything else, including civil unrest, political upheaval, natural disaster, or anything this data doesn't clearly support)

Respond in exactly this format:
CATEGORY: <conflict, currency instability, or other>
REASONING: <2-3 sentences citing the specific numbers above>"""

    response = call_ollama(prompt)
    match = re.search(r"CATEGORY:\s*(conflict|currency instability|other)", response, re.IGNORECASE)
    category = match.group(1).lower() if match else None
    return category, response, prompt


# ---------- Stage 4: template selection (mechanical, no LLM) ----------
CONFLICT_TEMPLATE = [
    "Conflict continues at current intensity",
    "Conflict resolves within 3 years",
    "Gradual de-escalation",
]
CURRENCY_TEMPLATE = [
    "Currency instability continues",
    "Currency stabilizes",
]


def select_template(category):
    if category == "conflict":
        return CONFLICT_TEMPLATE
    if category == "currency instability":
        return CURRENCY_TEMPLATE
    return None  # "other" (or unparsed) routes to undetermined - no template exists for it


# ---------- Stage 5: LLM-drafted magnitude ----------
def draft_magnitude(signal_df, current_years, precedent_years, category, last_observed_spending_pct):
    """
    last_observed_spending_pct: the flagged country's own most recent
    observed military spending, as % of GDP - real SIPRI data, structurally
    available for any flagged country (it's what calculate_volatility
    itself already reads in stage 1), not Ukraine-specific knowledge.
    Included after an initial run with no numeric anchor produced an
    incoherent result (the LLM conflated event-volume ratios with GDP
    percentages) - a ceiling estimate needs a real reference point for what
    "current" spending looks like, for any country, not just this one.

    A drafted ceiling that doesn't exceed the country's own current
    spending is definitionally incoherent (a "growth ceiling" below where
    spending already sits isn't a ceiling), so it's checked automatically
    below rather than trusted at face value.
    """
    current_rows = signal_df[signal_df["Year"].isin(current_years)].sort_values("Year")
    precedent_rows = signal_df[signal_df["Year"].isin(precedent_years)].sort_values("Year")

    current_desc = "; ".join(f"{r['baseline_ratio']:.1f}x baseline, tone {r['avg_tone']:.1f}"
                              for _, r in current_rows.iterrows())
    if len(precedent_rows) > 0:
        precedent_desc = "; ".join(f"{r['baseline_ratio']:.1f}x baseline, tone {r['avg_tone']:.1f}"
                                    for _, r in precedent_rows.iterrows())
        precedent_line = (f"This location has {len(precedent_rows)} earlier, separate historical precedent "
                           f"episode(s) of similarly elevated event volume: {precedent_desc}.")
    else:
        precedent_line = "This location has no earlier historical precedent episodes of elevated event volume."

    prompt = f"""You are drafting a magnitude estimate for a {category} scenario, to use as a growth ceiling (the maximum plausible military spending as a share of GDP) in a forecasting model.

This location's military spending is currently {last_observed_spending_pct:.1f}% of GDP (most recent observed year).

Current elevated episode (event-monitoring data, not spending data): {current_desc}
{precedent_line}

Draft a specific numeric ceiling (percent of GDP) for how high military spending could plausibly rise if this {category} situation continues. The ceiling must be higher than the current {last_observed_spending_pct:.1f}% of GDP figure above - it is a maximum, not a current estimate. Also give a confidence level.

Respond in exactly this format:
CEILING_PERCENT_GDP: <a single number, e.g. 35>
CONFIDENCE: <low, moderate, or high>
REASONING: <2-3 sentences citing how many precedent episodes were available and what in the data drove the number; state explicitly if confidence is low due to limited historical precedent>"""

    response = call_ollama(prompt)
    ceiling_match = re.search(r"CEILING_PERCENT_GDP:\s*([\d.]+)", response)
    confidence_match = re.search(r"CONFIDENCE:\s*(low|moderate|high)", response, re.IGNORECASE)
    ceiling = float(ceiling_match.group(1)) if ceiling_match else None
    confidence = confidence_match.group(1).lower() if confidence_match else None

    # Automatic sanity gate, not a manual eyeball check: a ceiling that
    # fails to parse, or doesn't exceed current spending, is incoherent by
    # definition and routes to "undetermined" rather than being trusted.
    if ceiling is None or confidence is None or ceiling <= last_observed_spending_pct:
        return None, None, response, prompt, "undetermined - LLM output failed the automatic coherence check"
    return ceiling, confidence, response, prompt, "ok"


# ---------- Stage 6: scenario construction (automated, docs/ROADMAP_v2_working_notes.md 2026-07-24) ----------
def construct_scenarios(signal_df, current_years, precedent_years, category):
    """
    LLM constructs three named scenarios directly from this country's real
    GDELT evidence, rather than filling in Ukraine's/Türkiye's rigid
    pre-written templates - the same reasoning quality already validated
    in categorize_episode(), applied to scenario construction instead of
    category labeling.

    Structurally always three scenarios (continues indefinitely / resolves
    relatively quickly / gradually de-escalates over a longer period),
    matching the conceptual shape of the hand-built templates - but the
    NAMES and DURATIONS are LLM-derived from the real evidence, not
    hardcoded. Deliberately constrained to simple structured fields (two
    names, two small integers), not raw numeric flag arrays: draft_magnitude()'s
    first attempt at unconstrained LLM numeric output produced an
    incoherent result (see docs/ROADMAP_v2_working_notes.md, 2026-07-24) - the lesson carried
    forward here is to constrain what the LLM has to get exactly right,
    not ask it to invent an arbitrary numeric sequence.

    Returns (scenario_spec, response, prompt) where scenario_spec is None
    if parsing failed or the result failed its coherence check (a
    "gradual, longer-process" scenario must actually take longer to
    resolve than the "quick resolution" one, and both must be positive
    integers) - never trusted at face value, same principle as
    draft_magnitude()'s ceiling check.
    """
    current_rows = signal_df[signal_df["Year"].isin(current_years)].sort_values("Year")
    precedent_rows = signal_df[signal_df["Year"].isin(precedent_years)].sort_values("Year")

    current_desc = "; ".join(f"{r['baseline_ratio']:.1f}x baseline, tone {r['avg_tone']:.1f}"
                              for _, r in current_rows.iterrows())
    if len(precedent_rows) > 0:
        precedent_desc = "; ".join(f"{r['baseline_ratio']:.1f}x baseline, tone {r['avg_tone']:.1f}"
                                    for _, r in precedent_rows.iterrows())
        precedent_line = f"This location has {len(precedent_rows)} earlier precedent episode(s): {precedent_desc}."
    else:
        precedent_line = "This location has no earlier precedent episodes."

    prompt = f"""You are naming three future scenarios for a {category} situation detected in event-monitoring data, to drive a forecasting model. A regressor flag is held at 1 while the situation is assumed active and transitions to 0 as it resolves.

Current elevated episode: {current_desc}
{precedent_line}

Name three scenarios:
1. The situation continues at its current intensity indefinitely.
2. The situation resolves relatively quickly - give a specific number of years until fully resolved.
3. The situation gradually de-escalates over a longer period than scenario 2 - give a specific number of years until fully resolved, greater than scenario 2's.

Respond in exactly this format:
SCENARIO_1_NAME: <short descriptive name for the "continues" scenario>
SCENARIO_2_NAME: <short descriptive name for the "resolves quickly" scenario>
SCENARIO_2_YEARS: <integer number of years>
SCENARIO_3_NAME: <short descriptive name for the "gradual" scenario>
SCENARIO_3_YEARS: <integer number of years, greater than SCENARIO_2_YEARS>"""

    response = call_ollama(prompt)
    name1 = re.search(r"SCENARIO_1_NAME:\s*(.+)", response)
    name2 = re.search(r"SCENARIO_2_NAME:\s*(.+)", response)
    years2 = re.search(r"SCENARIO_2_YEARS:\s*(\d+)", response)
    name3 = re.search(r"SCENARIO_3_NAME:\s*(.+)", response)
    years3 = re.search(r"SCENARIO_3_YEARS:\s*(\d+)", response)

    if not all([name1, name2, years2, name3, years3]):
        return None, response, prompt

    y2, y3 = int(years2.group(1)), int(years3.group(1))
    # Coherence gate: a "gradual, longer-process" scenario has to actually
    # take longer than "resolves quickly", and both must be positive - not
    # trusted at face value, same principle as draft_magnitude()'s check.
    if y2 <= 0 or y3 <= 0 or y3 <= y2:
        return None, response, prompt

    spec = {
        "continues_name": name1.group(1).strip(),
        "resolves_name": name2.group(1).strip(),
        "resolves_years": y2,
        "gradual_name": name3.group(1).strip(),
        "gradual_years": y3,
    }
    return spec, response, prompt


def build_flag_schedules(scenario_spec):
    """Deterministic numeric flag arrays built from construct_scenarios()'s
    structured output - the LLM never generates raw numbers here, only the
    names and the two integer durations that parameterize these three
    fixed shapes (matching scenario_templates.py's hand-built shapes)."""
    d2, d3 = scenario_spec["resolves_years"], scenario_spec["gradual_years"]
    return {
        scenario_spec["continues_name"]: lambda n: np.ones(n),
        scenario_spec["resolves_name"]: lambda n, d=d2: np.concatenate([np.ones(min(d, n)), np.zeros(max(0, n - d))]),
        scenario_spec["gradual_name"]: lambda n, d=d3: np.clip(1 - np.arange(n) / d, 0, 1),
    }


def train_automated_scenario_model(country, spending_df, signal_df, cap_value, changepoint_prior_scale):
    """
    Fits an automated scenario Prophet model using the elevated_active
    signal (stage 2) as the regressor directly - unlike scenario_templates.py's
    hand-built models, which use a hardcoded historical date, this reuses
    the exact binary series stage 2 already computed for this country, so
    there's no second, separately-tuned flag definition to keep in sync.
    """
    country_data = spending_df[spending_df["Country"] == country][["Year", "Spending"]].dropna()
    country_data = country_data.sort_values("Year").reset_index(drop=True)
    country_data["ds"] = pd.to_datetime(country_data["Year"], format="%Y")
    country_data["y"] = country_data["Spending"]
    flag_lookup = signal_df.set_index("Year")["elevated_active"]
    country_data["elevated_active"] = country_data["Year"].map(flag_lookup).fillna(0).astype(int)

    prophet_kwargs = dict(changepoint_prior_scale=changepoint_prior_scale, yearly_seasonality=False)
    cols = ["ds", "y", "elevated_active"]
    if cap_value is not None:
        country_data["cap"] = cap_value
        cols.append("cap")
        prophet_kwargs["growth"] = "logistic"

    model = Prophet(**prophet_kwargs)
    model.add_regressor("elevated_active")
    model.fit(country_data[cols])
    return model, country_data["elevated_active"].values


def forecast_automated_scenarios(model, historical_flag, scenario_spec, years_ahead, cap_value=None):
    """Returns {scenario_name: [forecast values]} for the three constructed scenarios."""
    flag_schedules = build_flag_schedules(scenario_spec)
    results = {}
    for name, flag_fn in flag_schedules.items():
        future = model.make_future_dataframe(periods=years_ahead, freq="YE")
        future["elevated_active"] = np.concatenate([historical_flag, flag_fn(years_ahead)])
        if cap_value is not None:
            future["cap"] = cap_value
        forecast = model.predict(future)
        results[name] = forecast["yhat"].tail(years_ahead).tolist()
    return results


# ---------- Orchestration: determine a growth cap for a flagged country ----------
def determine_cap(country, hand_set_caps, no_cap_needed, spending_df, gdelt_df):
    """
    Returns (cap_fraction_or_None, metadata) for a flagged country.
    `metadata` is always a dict with a "decision" key - never conflate
    "explicitly resolved with no cap value" with "unresolved" by checking
    `cap is None` alone, since both "no_cap_needed" and "undetermined"
    return `cap=None`. Check `metadata["decision"]` instead:
      - "capped": a real cap value exists (metadata["cap"] duplicates it).
      - "no_cap_needed": explicitly evaluated and found not to need one -
        safe to serve uncapped, not an unresolved gap.
      - "undetermined": genuinely unresolved - the caller (train.py) routes
        this to the flagged-but-uncapped fail-safe (main.py refuses rather
        than serving a forecast that was never confirmed safe).

    This three-way split was added after a real gap was found (2026-07-24):
    Türkiye's "no growth cap needed" decision (checked directly - historical
    max 4.30% of GDP, 2018-2025 crisis years stay below that, no runaway
    pattern) was never encoded anywhere. It worked only by accident, because
    Türkiye isn't in `calculate_volatility`'s flagged list at all (it was
    identified by an earlier, different method) - so it bypasses the
    fail-safe gate rather than passing it. If a country's volatility were
    ever to newly cross the threshold after already having a "no cap
    needed" decision on record, that decision needs to be encoded
    somewhere the gate can see it, or it would silently turn into an
    unexplained refusal instead of correctly recognizing the question was
    already answered. See docs/ROADMAP_v2_working_notes.md.

    Priority order, per the 2026-07-24 decision (docs/ROADMAP_v2_working_notes.md): a
    hand-researched entry in `hand_set_caps` or `no_cap_needed` always wins
    over the automated pipeline below. A value with a real, externally-
    checkable anchor (or an explicit "checked, not needed" determination)
    is preferred over an LLM estimate reasoning from GDELT event magnitude
    alone, even at "moderate" stated confidence - the LLM estimate has no
    outside verification at all, unlike e.g. Ukraine's UK WW2 precedent.

    Only for a flagged country with no hand-set entry does this fall back
    to the automated pipeline (detection already done by the caller ->
    GDELT signal -> categorization -> template selection -> LLM-drafted
    magnitude), validated end-to-end via a blind test on Ukraine
    (backend/blind_test_ukraine.py) before being wired in here. The
    resulting metadata is tagged "source": "automated" specifically so
    main.py/the API can present it distinctly from a hand-researched
    value - an automated, GDELT-only estimate must never look equally
    confident as an externally-verified one.

    Kept in this module (not train.py) so it's safely importable and
    directly testable - train.py has no `if __name__ == "__main__":` guard
    and runs its full training loop as top-level code on import.
    """
    if country in hand_set_caps:
        return hand_set_caps[country], {
            "decision": "capped",
            "cap": hand_set_caps[country],
            "source": "hand-researched",
            "confidence": "high",
            "note": "Externally verified against independent historical precedent - see docs/ROADMAP_v2_working_notes.md.",
        }

    if country in no_cap_needed:
        return None, {
            "decision": "no_cap_needed",
            "source": "hand-researched",
            "confidence": "high",
            "note": no_cap_needed[country],
        }

    signal_df, _ = compute_elevated_signal(gdelt_df, country)
    current_years, precedent_years = find_current_episode(signal_df)
    if not current_years:
        return None, {"decision": "undetermined",
                       "note": "flagged by SIPRI volatility but GDELT shows no corresponding elevation"}

    category, _, _ = categorize_episode(signal_df, current_years)
    template = select_template(category)
    if template is None:
        return None, {"decision": "undetermined", "note": "category 'other' (or unparsed) - no template exists"}

    last_observed = spending_df[spending_df["Country"] == country].dropna().sort_values("Year")["Spending"].iloc[-1] * 100
    ceiling, confidence, _, _, status = draft_magnitude(
        signal_df, current_years, precedent_years, category, last_observed
    )
    if status != "ok":
        return None, {"decision": "undetermined", "note": "LLM output failed the automatic coherence check"}

    return ceiling / 100, {
        "decision": "capped",
        "cap": ceiling / 100,
        "source": "automated",
        "confidence": confidence,
        "note": f"GDELT-only estimate (category: {category}), not independently cross-checked - see docs/ROADMAP_v2_working_notes.md.",
    }


# ---------- Orchestration: determine a scenario for a flagged country ----------
def determine_scenario(country, hand_built_countries, spending_df, gdelt_df):
    """
    Returns a dict describing an automated scenario configuration for a
    flagged country, or None if hand-built, undetermined, or the category
    has no scenario template - never a guess. Mirrors determine_cap()'s
    exact pattern (2026-07-24): a hand-built entry always wins and is
    checked first; the automated pipeline only runs for a flagged country
    with neither.

    `hand_built_countries` is the caller's registry of hand-built templates
    (e.g. scenario_templates.SCENARIO_TEMPLATES) - a country in it returns
    None here, since the caller already trains that country's model
    entirely separately via scenario_templates.py, not this module. This
    function has nothing to add for those two countries; it exists purely
    for the automated fallback.

    On success, returns {"category", "cap", "confidence", "scenario_spec",
    "signal_df"} - everything train.py needs to train and label an
    automated scenario model without recomputing stages 2/3/5 a second
    time. Every automated result is tagged with the same "source":
    "automated" / confidence-level pattern already established for
    determine_cap()'s automated growth caps - it must never be presented
    with the same certainty as Ukraine's or Türkiye's hand-built scenarios.

    Undetermined at any stage - no elevated episode, category not conflict
    or currency instability (no template exists for "other" yet, see
    docs/ROADMAP_v2_working_notes.md step 4), the magnitude coherence check failing, or the
    scenario-construction coherence check failing - returns None. The
    caller routes that to the existing "undetermined" status, the same as
    a country determine_cap() also couldn't resolve.
    """
    if country in hand_built_countries:
        return None

    signal_df, _ = compute_elevated_signal(gdelt_df, country)
    current_years, precedent_years = find_current_episode(signal_df)
    if not current_years:
        return None

    category, _, _ = categorize_episode(signal_df, current_years)
    if category not in ("conflict", "currency instability"):
        return None  # "other"/unparsed - no scenario template exists for this category yet (see docs/ROADMAP_v2_working_notes.md)

    last_observed = spending_df[spending_df["Country"] == country].dropna().sort_values("Year")["Spending"].iloc[-1] * 100
    ceiling, mag_confidence, _, _, mag_status = draft_magnitude(
        signal_df, current_years, precedent_years, category, last_observed
    )
    if mag_status != "ok":
        return None

    scenario_spec, _, _ = construct_scenarios(signal_df, current_years, precedent_years, category)
    if scenario_spec is None:
        return None

    return {
        "category": category,
        "cap": ceiling / 100,
        "confidence": mag_confidence,
        "scenario_spec": scenario_spec,
        "signal_df": signal_df,
    }
