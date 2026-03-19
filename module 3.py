import os
import pandas as pd
import kagglehub

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

#dowload dataset
path = kagglehub.dataset_download(
    "kamilpytlak/personal-key-indicators-of-heart-disease/versions/2"
)

files = os.listdir(path)
csv_file = [f for f in files if f.endswith(".csv")][0]
csv_path = os.path.join(path, csv_file)

df = pd.read_csv(csv_path)

print("Original shape:", df.shape)


df = df.drop_duplicates().dropna().copy()

print("Shape after cleaning:", df.shape)
print("\nHeartDisease distribution:")
print(df["HeartDisease"].value_counts())

# only looking at heart dieses portion of data set
hd_df = df[df["HeartDisease"] == "Yes"].copy()

print("\nHeart disease subset shape:", hd_df.shape)


feature_cols = [
    "BMI",
    "Smoking",
    "AlcoholDrinking",
    "Stroke",
    "PhysicalHealth",
    "MentalHealth",
    "DiffWalking",
    "Sex",
    "AgeCategory",
    "Race",
    "Diabetic",
    "PhysicalActivity",
    "GenHealth",
    "SleepTime",
    "Asthma",
    "KidneyDisease"
]

categorical_cols = [
    "Smoking",
    "AlcoholDrinking",
    "Stroke",
    "DiffWalking",
    "Sex",
    "AgeCategory",
    "Race",
    "Diabetic",
    "PhysicalActivity",
    "GenHealth",
    "Asthma",
    "KidneyDisease"
]

X = hd_df[feature_cols].copy()
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=False)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

#cosine similarity
similarity_matrix = cosine_similarity(X_scaled)

similarity_df = pd.DataFrame(
    similarity_matrix,
    index=hd_df.index,
    columns=hd_df.index
)

# top 10 query for similarty
def get_top_similar(query_index, top_n=10):
    sims = similarity_df.loc[query_index].drop(query_index)
    top_matches = sims.sort_values(ascending=False).head(top_n)

    result = hd_df.loc[top_matches.index, feature_cols].copy()
    result.insert(0, "similarity_score", top_matches.values)
    result.insert(0, "matched_index", top_matches.index)

    return result


#can be modified to make different query
query_indices = [5, 10, 35]

print("\nChosen query indices:", query_indices)


for q_idx in query_indices:
    print("\n" + "=" * 90)
    print(f"QUERY ITEM: {q_idx}")
    print("=" * 90)

    print("\nQuery profile:")
    print(hd_df.loc[q_idx, feature_cols])

    print("\nTop 10 most similar items:")
    top_10 = get_top_similar(q_idx, top_n=10)
    print(top_10)


    top_10.to_csv(f"top10_similar_query_{q_idx}.csv", index=False)

print("\nDone.")