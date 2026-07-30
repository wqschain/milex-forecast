"""
Embeddings step of the scenario-forecasting pipeline (see docs/ROADMAP_v2_working_notes.md step 2).

Converts each of the 1,426 country-years in gdelt_country_year.csv into a
short natural-language description and embeds it with sentence-transformers,
so country-years can later be clustered by underlying similarity (step 3).

Per the finding recorded in docs/ROADMAP_v2_working_notes.md, the description leads with
share-normalized event volume relative to that country's own trailing
baseline (the strong signal), not raw Goldstein/tone averages alone (the
weak one) - avg_goldstein and avg_tone are still included, just not as the
primary signal.

Local model (all-MiniLM-L6-v2, the standard general-purpose sentence-
transformers default), no API cost. Runs in seconds for ~1,400 rows.
"""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

TEXT_OUTPUT = "gdelt_country_year_descriptions.csv"
EMBEDDINGS_OUTPUT = "gdelt_embeddings.npy"

TRAILING_WINDOW_YEARS = 5

df = pd.read_csv("gdelt_country_year.csv").sort_values(["Country", "Year"]).reset_index(drop=True)

# Share of each year's combined mention total across all 31 countries -
# cancels out GDELT's own ~17x source-coverage growth from 2006-2016, which
# otherwise swamps any same-country comparison (see docs/ROADMAP_v2_working_notes.md finding).
yearly_total = df.groupby("Year")["total_mentions"].transform("sum")
df["mention_share"] = df["total_mentions"] / yearly_total


def trailing_ratio(group):
    # min_periods=1 (not 3): using whatever trailing history exists, even
    # just one prior year, instead of requiring three - min_periods=3 left
    # each country's first ~3 years with no ratio at all (93 rows total),
    # which caused a downstream problem (see the note on missing_baseline
    # below). Only the true first year of each country's series (where
    # shift(1) has literally nothing before it, regardless of min_periods)
    # still ends up with no ratio - an unavoidable 31 rows, one per country.
    trailing_avg = group["mention_share"].shift(1).rolling(TRAILING_WINDOW_YEARS, min_periods=1).mean()
    return group["mention_share"] / trailing_avg


df["spike_ratio"] = df.groupby("Country", group_keys=False).apply(trailing_ratio)

# For that unavoidable first-year-per-country remainder, default to 1.0
# ("assume typical baseline") rather than a placeholder phrase or omitting
# the volume clause. An earlier version did the latter and it backfired:
# even without identical wording, rows missing the volume clause were
# structurally distinct (shorter, starting with "Average event tone...")
# and k-means clustered all ~93 of them together by that shared shape, not
# by any real event characteristic. Defaulting to 1.0 keeps every row's
# sentence the same shape, and it only affects 31/1426 rows (~2%).
df["missing_baseline"] = df["spike_ratio"].isna()
df["spike_ratio"] = df["spike_ratio"].fillna(1.0)
print(f"{df['missing_baseline'].sum()} rows had no trailing history at all (each country's first year) "
      f"- defaulted to spike_ratio=1.0")


def volume_phrase(ratio):
    if ratio < 1.5:
        return f"event volume around its typical baseline ({ratio:.1f}x)"
    if ratio < 2.5:
        return f"event volume somewhat elevated above its typical baseline ({ratio:.1f}x)"
    if ratio < 5:
        return f"a notable spike in event volume above its typical baseline ({ratio:.1f}x)"
    return f"an extreme spike in event volume far above its typical baseline ({ratio:.1f}x)"


def tone_phrase(avg_tone):
    return "negative" if avg_tone < 0 else "positive"


def describe(row):
    # Deliberately excludes Country/Year: an early sanity check (comparing
    # nearest neighbors for Ukraine 2022, Norway 2022, Egypt 2011) showed
    # that including the country name as a leading token made same-country
    # years dominate similarity almost entirely - Ukraine 2022's nearest
    # neighbors were other Ukraine years, not other volatile country-years
    # elsewhere. That defeats the actual clustering goal (grouping by event
    # character across countries, not rediscovering "31 country clusters").
    # Country/Year are kept as separate metadata columns instead.
    return (
        f"{volume_phrase(row['spike_ratio']).capitalize()}. "
        f"Average event tone was {tone_phrase(row['avg_tone'])} ({row['avg_tone']:.2f}). "
        f"Average Goldstein score was {row['avg_goldstein']:.2f}."
    )


df["description"] = df.apply(describe, axis=1)

df[["Country", "Year", "avg_goldstein", "avg_tone", "mention_share", "spike_ratio",
    "missing_baseline", "description"]].to_csv(TEXT_OUTPUT, index=False)
print(f"wrote {len(df)} descriptions to {TEXT_OUTPUT}")
print()
print("--- sample descriptions ---")
for _, row in df[df["Country"].isin(["Ukraine", "Türkiye", "Norway"])].iterrows():
    if row["Year"] in (2016, 2018, 2021, 2022, 2023):
        print(row["description"])

print()
print("embedding with sentence-transformers (all-MiniLM-L6-v2, local)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["description"].tolist(), show_progress_bar=True, normalize_embeddings=True)
np.save(EMBEDDINGS_OUTPUT, embeddings)
print(f"saved {embeddings.shape} embeddings to {EMBEDDINGS_OUTPUT}")
