import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

DATA = "data/customer_data.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA)

features = [
    "TotalSpending",
    "NumberOfPurchases",
    "PurchaseFrequency",
    "AverageOrderValue",
    "Recency"
]

X = df[features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compare K values and select the best silhouette score.
scores = {}
for k in range(2, 7):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    scores[k] = silhouette_score(X_scaled, labels)

best_k = max(scores, key=scores.get)

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Second clustering approach for the required comparison.
agg = AgglomerativeClustering(n_clusters=best_k)
agg_labels = agg.fit_predict(X_scaled)
agg_score = silhouette_score(X_scaled, agg_labels)
kmeans_score = silhouette_score(X_scaled, df["Cluster"])

joblib.dump(kmeans, f"{MODEL_DIR}/kmeans_model.pkl")
joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
joblib.dump(features, f"{MODEL_DIR}/features.pkl")
df.to_csv(f"{MODEL_DIR}/clustered_customers.csv", index=False)

# Segment profile
profile = df.groupby("Cluster")[features].mean().round(2)
profile.to_csv(f"{MODEL_DIR}/segment_profile.csv")

# Save metrics
with open(f"{MODEL_DIR}/metrics.txt", "w") as f:
    f.write(f"Best K: {best_k}\n")
    f.write(f"K-Means silhouette score: {kmeans_score:.4f}\n")
    f.write(f"Agglomerative silhouette score: {agg_score:.4f}\n")
    f.write("K-Means K scores:\n")
    for k, score in scores.items():
        f.write(f"k={k}: {score:.4f}\n")

# Simple cluster plot
plt.figure(figsize=(8, 5))
plt.scatter(
    df["TotalSpending"], df["NumberOfPurchases"],
    c=df["Cluster"], alpha=0.65
)
plt.xlabel("Total Spending")
plt.ylabel("Number of Purchases")
plt.title("Customer Segments")
plt.tight_layout()
plt.savefig(f"{MODEL_DIR}/customer_segments.png", dpi=150)
plt.close()

print("Training completed.")
print(f"Best number of clusters: {best_k}")
print(f"K-Means silhouette score: {kmeans_score:.4f}")
print(f"Agglomerative silhouette score: {agg_score:.4f}")
