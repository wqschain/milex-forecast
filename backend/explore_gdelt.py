"""
Exploratory script for the scenario-forecasting phase (see docs/ROADMAP_v2_working_notes.md).
Confirms BigQuery access to GDELT's public dataset, discovers the real
per-country FIPS codes GDELT uses (empirically, not assumed from ISO codes,
which do NOT match - see the Australia/Sweden note below), and estimates
the cost of the real 31-country aggregation query before it's run for real.

None of this is billed unless it actually queries `gdelt-bq.full.events`;
dry runs (job_config=bigquery.QueryJobConfig(dry_run=True)) cost nothing.

Phases 2, 3, 3B, and 3C, and the aggregation query, already ran once
(2026-07-22) and are confirmed - their results are recorded in comments
below, and they all default to OFF so re-running this script doesn't
silently re-bill ~445 GB to reconfirm something already known. Flip the
relevant RUN_* flag back to True only if you want to re-verify one of them
(e.g. after a schema change). The aggregation query's dry-run estimate is
still printed unconditionally (costs nothing) so its cost stays visible
even with RUN_AGGREGATION off.
"""
import json
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

client = bigquery.Client(project="milex-forecast")

BACKEND_DIR = Path(__file__).resolve().parent
COUNTRY_LEVEL_CACHE = BACKEND_DIR / "gdelt_country_level_raw.json"
AGGREGATION_OUTPUT = BACKEND_DIR / "gdelt_country_year.csv"

RUN_DATE_CHECK = False
RUN_UKRAINE_SAMPLE = False
RUN_UNREST_VERIFICATION = False
RUN_POLICY_SHIFT_VERIFICATION = False
RUN_AGGREGATION = False

# --- PHASE 1: table metadata (free - no data scanned) ---
for table_id in ["gdelt-bq.full.events", "gdelt-bq.gdeltv2.events"]:
    t = client.get_table(table_id)
    print(f"{table_id}: {t.num_rows:,} rows, {t.num_bytes / 1e9:.1f} GB, "
          f"partitioned={t.time_partitioning is not None}, clustered={bool(t.clustering_fields)}")
# Neither table is partitioned or clustered, which matters a lot here: a
# WHERE filter on Year or country does NOT reduce bytes scanned. Cost is
# driven purely by which columns are selected, not by filter selectivity -
# every query below is dry-run first for exactly that reason.
# gdeltv2.events only starts in 2015, so full.events (1920-2026, confirmed
# below) is the table that actually covers the needed 1980-2025 range.

# --- PHASE 2: confirm date coverage ---
# Confirmed (2026-07-22, 7.062 GB billed): min_year=1920, max_year=2026.
# Comfortably spans the needed 1980-2025 range.
if RUN_DATE_CHECK:
    job = client.query("SELECT MIN(Year) as min_year, MAX(Year) as max_year FROM `gdelt-bq.full.events`")
    for row in job:
        print(dict(row))
    print(f"bytes billed: {job.total_bytes_billed / 1e9:.3f} GB")

# --- PHASE 3: Ukraine sample - confirm real data access + shape ---
# FIPS 10-4 country code for Ukraine, confirmed via
# `SELECT DISTINCT ActionGeo_CountryCode, ActionGeo_FullName ... WHERE
# ActionGeo_FullName LIKE "%Ukraine%"` (26.7 GB) before trusting it here.
# Confirmed (2026-07-22, 99.615 GB billed): events cluster around Feb-Mar
# 2022 (invasion) and Apr-May 2022 (Donbas/Azovstal), GoldsteinScale
# bottoms out at -10.0 on EventRootCode "19" (CAMEO FIGHT), AvgTone
# negative almost everywhere - consistent with real invasion-period
# coverage, not noise. Confirmed looking right before building anything
# further on top of it.
if RUN_UKRAINE_SAMPLE:
    sql_ukraine_sample = """
    SELECT SQLDATE, Year, Actor1Name, Actor2Name, EventCode, EventRootCode, QuadClass,
           GoldsteinScale, AvgTone, NumMentions, NumArticles, ActionGeo_FullName
    FROM `gdelt-bq.full.events`
    WHERE ActionGeo_CountryCode = "UP" AND Year = 2022
    ORDER BY NumMentions DESC
    LIMIT 20
    """
    job = client.query(sql_ukraine_sample)
    rows = [dict(row) for row in job]
    print(f"Ukraine 2022 sample: {len(rows)} rows, {job.total_bytes_billed / 1e9:.3f} GB billed")
    for r in rows[:5]:
        print(json.dumps(r, default=str))

# --- PHASE 3B: verify the "positive-tone spike" embedding cluster wasn't
# coincidental (see docs/ROADMAP_v2_working_notes.md embeddings follow-up finding) ---
# Confirmed (2026-07-22, 99.615 GB billed): Korea 1987's top events by
# mention volume are Seoul student protesters vs. riot police (EventCode
# 190/"FIGHT", GoldsteinScale repeatedly -10.0) clustered Apr-Sep 1987 - the
# real June Democratic Uprising. India 1984's top events are Sikh actors vs.
# paramilitary police in Amritsar/Punjab (the Operation Blue Star era) and
# Hindu-Muslim communal violence in Bombay, with Zail Singh/Indira Gandhi/
# Rajiv Gandhi as real actors - one of the most violent years in modern
# Indian history. Both real, both violent - and in both, GoldsteinScale
# -10.0 events still carry *positive* AvgTone (1.9-11.7), a real divergence
# between GDELT's conflict-intensity score and its lexical sentiment score,
# not an error. This cluster is domestic civil unrest / protest / communal
# violence (positive-toned coverage despite real violence), categorically
# different from Ukraine's sustained interstate war (negative-toned
# coverage) - not "elections/diplomatic events" as first guessed.
if RUN_UNREST_VERIFICATION:
    sql_unrest_verification = """
    SELECT * EXCEPT(rn) FROM (
      SELECT ActionGeo_CountryCode, Year, SQLDATE, Actor1Name, Actor2Name, EventCode, EventRootCode,
             QuadClass, GoldsteinScale, AvgTone, NumMentions, NumArticles, ActionGeo_FullName,
             ROW_NUMBER() OVER (PARTITION BY ActionGeo_CountryCode ORDER BY NumMentions DESC) AS rn
      FROM `gdelt-bq.full.events`
      WHERE (ActionGeo_CountryCode = "IN" AND Year = 1984)
         OR (ActionGeo_CountryCode = "KS" AND Year = 1987)
    )
    WHERE rn <= 20
    ORDER BY ActionGeo_CountryCode, NumMentions DESC
    """
    job = client.query(sql_unrest_verification)
    rows = [dict(row) for row in job]
    print(f"India 1984 / Korea 1987 sample: {len(rows)} rows, {job.total_bytes_billed / 1e9:.3f} GB billed")
    for r in rows[:5]:
        print(json.dumps(r, default=str))

# --- PHASE 3C: verify whether "policy shift" has a real, findable GDELT
# signal - Germany's 2022 Zeitenwende defense-policy reversal is real,
# documented, and unlike Ukraine/Korea/India has no aggregate spike_ratio
# signal at all (Germany's spike_ratio never exceeds ~1.05 across
# 2018-2025). Does that mean nothing is there, like Türkiye, or does it
# mean the aggregate feature is diluting a real signal, like the original
# avg_goldstein finding? ---
# Confirmed (2026-07-22, 99.615 GB billed): real, identifiable Zeitenwende
# content IS present - NAVY/FRIGATE deployment events and Germany
# "providing aid" (EventCode 070, GoldsteinScale +7.0) cluster right at
# Feb 25-26 2022 (the Zeitenwende speech was Feb 27), and repeated
# cooperation events located at Ramstein appear Apr 25 / May 22 / Jun 25 -
# the real Ukraine Defense Contact Group ("Ramstein format") that started
# convening there in April 2022. This is a THIRD, distinct outcome from
# the other two: unlike Türkiye (no distinguishable content at all - a
# true data absence), Germany's signal genuinely exists in the raw events,
# it just doesn't clear the yearly-aggregate spike_ratio threshold -
# likely because Germany's baseline event volume is already enormous (see
# the Canada/Australia English-language-media-bias note from the earlier
# volatility check), and because diplomatic/aid-provision events (Goldstein
# positive - cooperation, not conflict) generate lower relative volume than
# active fighting does, even when the underlying policy shift is major.
# This is a detection-methodology gap (current features aren't sensitive
# enough), not a data-absence gap - worth documenting as a real limitation
# of the *current* pipeline, but distinct from currency instability's more
# fundamental absence. See docs/ROADMAP_v2_working_notes.md for how this is handled for now.
if RUN_POLICY_SHIFT_VERIFICATION:
    sql_policy_shift_verification = """
    SELECT SQLDATE, Year, Actor1Name, Actor2Name, EventCode, EventRootCode, QuadClass,
           GoldsteinScale, AvgTone, NumMentions, NumArticles, ActionGeo_FullName
    FROM `gdelt-bq.full.events`
    WHERE ActionGeo_CountryCode = "GM" AND Year = 2022
    ORDER BY NumMentions DESC
    LIMIT 30
    """
    job = client.query(sql_policy_shift_verification)
    rows = [dict(row) for row in job]
    print(f"Germany 2022 sample: {len(rows)} rows, {job.total_bytes_billed / 1e9:.3f} GB billed")
    for r in rows[:5]:
        print(json.dumps(r, default=str))

# --- PHASE 4: discover every country's real FIPS code in one pass ---
# ActionGeo_Type == 1 is GDELT's documented "country-level" geocoding type,
# where ActionGeo_FullName is just the bare country name - one query gets
# every country's code at once (33.7 GB) instead of 31 separate per-country
# LIKE queries (~30 GB each => ~930 GB, nearly the entire monthly free
# tier). Cached to a file so re-running this script doesn't re-bill it.
if COUNTRY_LEVEL_CACHE.exists():
    with open(COUNTRY_LEVEL_CACHE, encoding="utf-8") as f:
        country_level_rows = json.load(f)
    print(f"loaded {len(country_level_rows)} country-level rows from cache")
else:
    sql_country_codes = """
    SELECT DISTINCT ActionGeo_CountryCode, ActionGeo_FullName
    FROM `gdelt-bq.full.events`
    WHERE ActionGeo_Type = 1
    """
    job = client.query(sql_country_codes)
    country_level_rows = [dict(row) for row in job]
    print(f"discovered {len(country_level_rows)} country-level rows, {job.total_bytes_billed / 1e9:.3f} GB billed")
    with open(COUNTRY_LEVEL_CACHE, "w", encoding="utf-8") as f:
        json.dump(country_level_rows, f, indent=2, ensure_ascii=False)

# --- PHASE 5: SIPRI name -> GDELT FIPS code, verified against phase 4 ---
# GDELT uses FIPS 10-4, not ISO 3166 - these are NOT interchangeable.
# Confirmed traps found in the real data: Australia is "AS" (FIPS "AU" is
# Austria); Sweden is "SW" (FIPS "SE" is Seychelles); China is "CH" (not
# ISO "CN"); Japan is "JA" (not ISO "JP"); Germany is "GM" (not ISO "DE").
# A naive ISO-code assumption would have silently pulled the wrong
# country's events for at least two of the 31.
SIPRI_TO_GDELT_NAME = {
    "United States of America": "United States",
    "Canada": "Canada",
    "Mexico": "Mexico",
    "Brazil": "Brazil",
    "Argentina": "Argentina",
    "Colombia": "Colombia",
    "United Kingdom": "United Kingdom",
    "Germany": "Germany",
    "France": "France",
    "Italy": "Italy",
    "Poland": "Poland",
    "Russia": "Russia",
    "Ukraine": "Ukraine",
    "Israel": "Israel",
    "Saudi Arabia": "Saudi Arabia",
    "Türkiye": "Turkey",
    "Iran": "Iran",
    "China": "China",
    "India": "India",
    "Japan": "Japan",
    "Korea, South": "South Korea",
    "Pakistan": "Pakistan",
    "Australia": "Australia",
    "Indonesia": "Indonesia",
    "Viet Nam": "Vietnam, Republic Of",
    "South Africa": "South Africa",
    "Nigeria": "Nigeria",
    "Egypt": "Egypt",
    "Algeria": "Algeria",
    "Sweden": "Sweden",
    "Norway": "Norway",
}

gdelt_code_by_name = {
    (row["ActionGeo_FullName"] or "").lower(): row["ActionGeo_CountryCode"] for row in country_level_rows
}

SIPRI_TO_GDELT_CODE = {}
unmatched = []
for sipri_name, gdelt_name in SIPRI_TO_GDELT_NAME.items():
    code = gdelt_code_by_name.get(gdelt_name.lower())
    if code is None:
        unmatched.append(sipri_name)
    else:
        SIPRI_TO_GDELT_CODE[sipri_name] = code

print(f"matched {len(SIPRI_TO_GDELT_CODE)}/{len(SIPRI_TO_GDELT_NAME)} countries to a GDELT code")
if unmatched:
    print("UNMATCHED (needs manual investigation before use):", unmatched)
for sipri_name, code in SIPRI_TO_GDELT_CODE.items():
    print(f"  {sipri_name:30s} -> {code}")

# --- PHASE 6: the real 31-country aggregation query ---
# Confirmed (2026-07-22, 38.749 GB billed, matching its dry-run estimate
# exactly): 1,426 rows (31 countries x 46 years, complete, no gaps), saved
# to gdelt_country_year.csv. Defaults to RUN_AGGREGATION=False since it's
# already run and the output file already exists - the dry-run estimate
# below still prints unconditionally (free) so the cost stays visible if
# you ever do need to re-run it (e.g. after a schema change).
codes_sql = ", ".join(f'"{c}"' for c in SIPRI_TO_GDELT_CODE.values())
sql_aggregation = f"""
SELECT
  Year,
  ActionGeo_CountryCode,
  AVG(GoldsteinScale) AS avg_goldstein,
  AVG(AvgTone) AS avg_tone,
  SUM(NumMentions) AS total_mentions,
  SUM(NumArticles) AS total_articles,
  COUNT(*) AS event_count
FROM `gdelt-bq.full.events`
WHERE ActionGeo_CountryCode IN ({codes_sql})
  AND Year BETWEEN 1980 AND 2025
GROUP BY Year, ActionGeo_CountryCode
ORDER BY Year, ActionGeo_CountryCode
"""

dry_run_job = client.query(sql_aggregation, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False))
print(f"aggregation query dry-run estimate: {dry_run_job.total_bytes_processed / 1e9:.3f} GB "
      f"({dry_run_job.total_bytes_processed / 1e12 * 100:.2f}% of 1TB free tier)")

if RUN_AGGREGATION:
    job = client.query(sql_aggregation)
    agg_rows = [dict(row) for row in job]
    print(f"aggregation: {len(agg_rows)} country-year rows, {job.total_bytes_billed / 1e9:.3f} GB billed")

    # Saved once, like cleaned_data.csv - this ~39 GB query never needs to
    # run again. GDELT code is mapped back to the SIPRI country name (via
    # the same verified mapping from phase 5) so this joins directly onto
    # cleaned_data.csv on Country + Year; the raw code is kept alongside for
    # traceability back to the source query.
    gdelt_code_to_sipri_name = {code: name for name, code in SIPRI_TO_GDELT_CODE.items()}
    df = pd.DataFrame(agg_rows)
    df["Country"] = df["ActionGeo_CountryCode"].map(gdelt_code_to_sipri_name)
    df = df[["Country", "Year", "avg_goldstein", "avg_tone", "total_mentions",
             "total_articles", "event_count", "ActionGeo_CountryCode"]]
    df = df.sort_values(["Country", "Year"])
    df.to_csv(AGGREGATION_OUTPUT, index=False)
    print(f"saved {len(df)} rows to {AGGREGATION_OUTPUT}")
else:
    print("RUN_AGGREGATION is False - aggregation query not executed, no bytes billed for it.")
