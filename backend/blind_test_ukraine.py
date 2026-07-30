"""
Blind test of backend/scenario_pipeline.py on Ukraine.

The point: Ukraine is the ONE country calculate_volatility's 2-std-dev
threshold organically flags today (re-confirmed: Nigeria, the next-highest,
is at 0.2935 against a 0.6122-0.6199 threshold - nowhere close), and it's
also the one country with well-established ground truth to check the
pipeline's output against. That makes it possible to run a genuinely blind
test without needing to borrow a country (which would only test
categorization/magnitude in isolation) or loosen the threshold to
manufacture a second candidate (rejected - an arbitrary threshold change
with no independent justification, the same failure mode already avoided
elsewhere in this project).

"Blind" means concretely: this script consults NOTHING already known about
Ukraine from prior manual work in this project.
    - No hardcoded 2022 date anywhere (stage 2 derives its own signal).
    - No "conflict" category asserted anywhere (stage 3's LLM call is
      anonymized - no country name or real calendar year is ever sent to
      it, only relabeled "Year 1/2/3..." and the numeric pattern).
    - No reuse of conflict_active, GROWTH_CAPS, or any value from
      test_regressor_ukraine*.py / test_scenarios_ukraine.py.
    - Stage 2's elevated-signal threshold is calibrated on the OTHER 30
      countries' data before Ukraine's own results are ever computed, so
      it can't have been tuned to land on any particular expected answer.

Ground truth is compared ONLY at the very end, after every stage has
already run and printed its output - never fed back into an earlier stage.
"""
import pandas as pd
from scenario_pipeline import (
    detect_flagged_countries, compute_elevated_signal, find_current_episode,
    categorize_episode, select_template, draft_magnitude,
)

TEST_COUNTRY = "Ukraine"

print("=" * 70)
print("STAGE 1: Detection (calculate_volatility, >2 std dev above mean)")
print("=" * 70)
spending_df = pd.read_csv("cleaned_data.csv")
flagged, threshold, volatilities = detect_flagged_countries(spending_df)
print(f"Threshold: {threshold:.4f}")
print(f"Flagged countries: {flagged}")
assert TEST_COUNTRY in flagged, f"{TEST_COUNTRY} not flagged - blind test requires organic stage-1 detection"
print(f"-> '{TEST_COUNTRY}' flagged organically, no manual selection.")
print()

print("=" * 70)
print("STAGE 2: GDELT-derived elevated-baseline signal")
print("=" * 70)
gdelt_df = pd.read_csv("gdelt_country_year_clusters.csv")  # has mention_share already computed
signal_df, elevated_threshold = compute_elevated_signal(gdelt_df, TEST_COUNTRY)
print(f"Threshold (calibrated on the other 30 countries): {elevated_threshold:.4f}")
print()
print(signal_df.to_string(index=False))
print()

current_years, precedent_years = find_current_episode(signal_df)
print(f"Current (most recent contiguous) elevated episode: {current_years}")
print(f"Earlier historical precedent episode years: {precedent_years}")
print()

print("=" * 70)
print("STAGE 3: Categorization (anonymized LLM call, no country/year revealed)")
print("=" * 70)
category, raw_response, prompt_used = categorize_episode(signal_df, current_years)
print("--- prompt sent to the LLM (anonymized) ---")
print(prompt_used)
print("--- raw LLM response ---")
print(raw_response)
print(f"--- parsed category: {category} ---")
print()

print("=" * 70)
print("STAGE 4: Template selection (mechanical, category -> template)")
print("=" * 70)
template = select_template(category)
print(f"Category '{category}' -> template: {template}")
print()

print("=" * 70)
print("STAGE 5: LLM-drafted magnitude (growth-cap ceiling estimate)")
print("=" * 70)
if template is not None:
    last_observed = spending_df[spending_df["Country"] == TEST_COUNTRY].dropna().sort_values("Year")["Spending"].iloc[-1] * 100
    print(f"Grounding anchor (last observed spending, real SIPRI data): {last_observed:.1f}% of GDP")
    ceiling, confidence, raw_mag_response, mag_prompt, status = draft_magnitude(
        signal_df, current_years, precedent_years, category, last_observed
    )
    print("--- prompt sent to the LLM ---")
    print(mag_prompt)
    print("--- raw LLM response ---")
    print(raw_mag_response)
    print(f"--- status: {status} ---")
    print(f"--- parsed: ceiling={ceiling}% of GDP, confidence={confidence} ---")
else:
    print("No template selected (category = 'other' or unparsed) - routes to undetermined, no magnitude drafted.")
    ceiling, confidence = None, None
print()

print("=" * 70)
print("COMPARISON AGAINST KNOWN GROUND TRUTH (consulted only now, at the end)")
print("=" * 70)
print("Known: Ukraine's full-scale invasion began Feb 2022; category = conflict.")
print("Known: hand-researched production growth cap = 50% of GDP (docs/ROADMAP_v2_working_notes.md,")
print("       grounded in UK WW2 defense spending, verified 46-52% of GDP).")
print("Known: actual observed 2025 spending = 39.6% of GDP, still rising.")
print()
print(f"Pipeline's independently-detected current episode: {current_years}")
print(f"Pipeline's independently-derived category: {category}")
print(f"Pipeline's independently-drafted ceiling: {ceiling}% of GDP (confidence: {confidence})")
