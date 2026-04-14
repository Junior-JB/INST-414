# Housing Affordability Clustering using Zillow Data (Improved Version)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


df = pd.read_csv("C:/Users/junior/OneDrive/Desktop/datasets/Metro_new_homeowner_income_needed_downpayment_0.20_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv")

time_cols = df.columns[6:]


df = df[df["RegionName"] != "United States"]


df = df.dropna(subset=time_cols[-12:])


latest_income = df[time_cols[-1]]
income_growth = df[time_cols[-1]] - df[time_cols[-12]]
volatility = df[time_cols].std(axis=1)

features = pd.DataFrame({
    "RegionName": df["RegionName"],
    "latest_income": latest_income,
    "income_growth": income_growth,
    "volatility": volatility
}).dropna()


features = features[
    (features["latest_income"] < features["latest_income"].quantile(0.99)) &
    (features["volatility"] < features["volatility"].quantile(0.99))
    ]


X = features[["latest_income", "income_growth", "volatility"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


k = 3

kmeans = KMeans(n_clusters=k, random_state=42)
features["cluster"] = kmeans.fit_predict(X_scaled)


print("\nCluster Summary (Mean Values):")
cluster_summary = features.groupby("cluster")[["latest_income", "income_growth", "volatility"]].mean()
print(cluster_summary)


print("\nCluster Sizes:")
print(features["cluster"].value_counts())


for i in range(k):
    print(f"\nCluster {i} example metros:")
    print(features[features["cluster"] == i]["RegionName"].head(5))


print("\nTop 10 Most Expensive Markets:")
print(features.sort_values("latest_income", ascending=False).head(10))


plt.figure()
plt.scatter(features["latest_income"], features["income_growth"], c=features["cluster"])
plt.xlabel("Latest Income Needed")
plt.ylabel("Income Growth (Last Year)")
plt.title("Housing Affordability Clusters")
plt.show()