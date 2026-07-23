"""
Clustering step of the scenario-forecasting pipeline (see ROADMAP.md step 3).

Groups the 1,426 country-year embeddings into natural clusters via k-means
(scikit-learn), without predefining categories - the LLM interpretation
step (step 4) assigns each resulting cluster one of the four decided
category labels afterward, potentially many-to-one.

k is chosen by silhouette score across a range, not guessed, since the
whole point of this step is to surface whatever structure is actually
there rather than force a predetermined count.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

CLUSTER_OUTPUT = "gdelt_country_year_clusters.csv"
RANDOM_STATE = 42

df = pd.read_csv("gdelt_country_year_descriptions.csv")
embeddings = np.load("gdelt_embeddings.npy")
assert len(df) == len(embeddings), "descriptions and embeddings are out of sync"

print("--- choosing k by silhouette score ---")
k_range = range(4, 16)
scores = {}
for k in k_range:
    labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(embeddings)
    score = silhouette_score(embeddings, labels)
    scores[k] = score
    print(f"k={k:2d}  silhouette={score:.4f}")

best_k = max(scores, key=scores.get)
print(f"\nbest k by silhouette score: {best_k}")

kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
df["cluster"] = kmeans.fit_predict(embeddings)

print("\n--- cluster sizes ---")
print(df["cluster"].value_counts().sort_index().to_string())

print("\n--- representative examples per cluster (closest to centroid) ---")
for cluster_id in sorted(df["cluster"].unique()):
    mask = df["cluster"] == cluster_id
    cluster_embeddings = embeddings[mask.values]
    centroid = kmeans.cluster_centers_[cluster_id]
    dists = np.linalg.norm(cluster_embeddings - centroid, axis=1)
    closest_idx = np.argsort(dists)[:5]
    cluster_rows = df[mask].reset_index(drop=True)
    print(f"\nCluster {cluster_id} (n={mask.sum()}):")
    for i in closest_idx:
        row = cluster_rows.iloc[i]
        print(f"  {row['Country']:25s} {int(row['Year'])}  {row['description']}")

df.to_csv(CLUSTER_OUTPUT, index=False)
print(f"\nsaved {len(df)} rows with cluster assignments to {CLUSTER_OUTPUT}")

print("\n--- which cluster is Ukraine 2022 in? ---")
ukraine_2022 = df[(df["Country"] == "Ukraine") & (df["Year"] == 2022)].iloc[0]
print(f"Ukraine 2022 -> cluster {ukraine_2022['cluster']}")
same_cluster = df[df["cluster"] == ukraine_2022["cluster"]]
print(f"Other members of that cluster ({len(same_cluster) - 1} others):")
for _, row in same_cluster.iterrows():
    if not (row["Country"] == "Ukraine" and row["Year"] == 2022):
        print(f"  {row['Country']:25s} {int(row['Year'])}  {row['description']}")
