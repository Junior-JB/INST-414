import kagglehub

# Download latest version
# path = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset")
# print("Path to dataset files:", path)

import pandas as pd
import networkx as nx
from itertools import combinations
from networkx.algorithms.community import greedy_modularity_communities


df = pd.read_csv("C:/dataset files/dataset.csv")

keep_cols = [
    "track_id", "artists", "track_name", "album_name", "popularity",
    "track_genre", "danceability", "energy", "valence", "tempo"
]
df = df[[col for col in keep_cols if col in df.columns]].copy()

df = df.dropna(subset=["artists"])

df["artist_list"] = df["artists"].apply(
    lambda x: [artist.strip() for artist in str(x).split(";") if artist.strip()]
)

df = df[df["artist_list"].apply(len) > 0].copy()

#build network
G = nx.Graph()

for _, row in df.iterrows():
    artists = row["artist_list"]
    popularity = row["popularity"] if pd.notna(row["popularity"]) else None
    genre = row["track_genre"] if pd.notna(row["track_genre"]) else None
    track_name = row["track_name"] if pd.notna(row["track_name"]) else None

    for artist in artists:
        if artist not in G:
            G.add_node(
                artist,
                tracks=0,
                popularity_sum=0,
                popularity_count=0,
                genres=set()
            )

        G.nodes[artist]["tracks"] += 1

        if popularity is not None:
            G.nodes[artist]["popularity_sum"] += popularity
            G.nodes[artist]["popularity_count"] += 1

        if genre is not None:
            G.nodes[artist]["genres"].add(genre)

    if len(artists) > 1:
        for a1, a2 in combinations(sorted(set(artists)), 2):
            if G.has_edge(a1, a2):
                G[a1][a2]["weight"] += 1
                if track_name is not None:
                    G[a1][a2]["tracks"].append(track_name)
            else:
                G.add_edge(
                    a1, a2,
                    weight=1,
                    tracks=[track_name] if track_name is not None else []
                )

for artist in G.nodes:
    pop_count = G.nodes[artist]["popularity_count"]
    G.nodes[artist]["avg_popularity"] = (
        G.nodes[artist]["popularity_sum"] / pop_count if pop_count > 0 else None
    )
    G.nodes[artist]["genre_count"] = len(G.nodes[artist]["genres"])


# node importance determined by these mettrics

degree_centrality = nx.degree_centrality(G)
weighted_degree = dict(G.degree(weight="weight"))
betweenness_centrality = nx.betweenness_centrality(G, weight=None)
pagerank = nx.pagerank(G, weight="weight")


results = pd.DataFrame({
    "artist": list(G.nodes),
    "num_tracks": [G.nodes[a]["tracks"] for a in G.nodes],
    "avg_popularity": [G.nodes[a]["avg_popularity"] for a in G.nodes],
    "genre_count": [G.nodes[a]["genre_count"] for a in G.nodes],
    "degree_centrality": [degree_centrality[a] for a in G.nodes],
    "weighted_degree": [weighted_degree[a] for a in G.nodes],
    "betweenness_centrality": [betweenness_centrality[a] for a in G.nodes],
    "pagerank": [pagerank[a] for a in G.nodes]
})

for col in ["degree_centrality", "weighted_degree", "betweenness_centrality", "pagerank"]:
    min_val = results[col].min()
    max_val = results[col].max()
    if max_val > min_val:
        results[col + "_scaled"] = (results[col] - min_val) / (max_val - min_val)
    else:
        results[col + "_scaled"] = 0

results["importance_score"] = (
        0.25 * results["pagerank_scaled"] +
        0.25 * results["degree_centrality_scaled"] +
        0.25 * results["weighted_degree_scaled"] +
        0.25 * results["betweenness_centrality_scaled"]
)

results = results.sort_values("importance_score", ascending=False)


print("\nTop 15 most important artists in the collaboration network:\n")
print(
    results[
        [
            "artist", "importance_score", "pagerank", "degree_centrality",
            "weighted_degree", "betweenness_centrality",
            "num_tracks", "avg_popularity", "genre_count"
        ]
    ].head(15).to_string(index=False)
)

bridge_artists = results.sort_values(
    ["betweenness_centrality", "genre_count"],
    ascending=[False, False]
)

print("\nTop 10 bridge artists:\n")
print(
    bridge_artists[
        ["artist", "betweenness_centrality", "genre_count", "avg_popularity"]
    ].head(10).to_string(index=False)
)

hub_artists = results.sort_values(
    ["weighted_degree", "degree_centrality"],
    ascending=[False, False]
)

print("\nTop 10 hub artists:\n")
print(
    hub_artists[
        ["artist", "weighted_degree", "degree_centrality", "num_tracks", "avg_popularity"]
    ].head(10).to_string(index=False)
)



#detecting different communities
communities = greedy_modularity_communities(G)

community_map = {}
for i, community in enumerate(communities):
    for artist in community:
        community_map[artist] = i

results["community"] = results["artist"].map(community_map)

print("\nTop communities by size:\n")
print(results["community"].value_counts().head(10))




num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
density = nx.density(G)

connected_components = list(nx.connected_components(G))
largest_component = max(connected_components, key=len)
largest_component_size = len(largest_component)

print("\nNetwork structure summary:\n")
print(f"Number of artists (nodes): {num_nodes}")
print(f"Number of collaborations (edges): {num_edges}")
print(f"Network density: {density:.8f}")
print(f"Number of connected components: {len(connected_components)}")
print(f"Largest connected component size: {largest_component_size}")





isolates = list(nx.isolates(G))
print("\nNumber of isolated artists (no collaborations):")
print(len(isolates))

if len(isolates) > 0:
    print("\nSample isolated artists:")
    print(isolates[:20])




# cross genre artist
cross_genre_artists = results.sort_values(
    ["genre_count", "betweenness_centrality", "weighted_degree"],
    ascending=[False, False, False]
)

print("\nTop 10 cross-genre artists:\n")
print(
    cross_genre_artists[
        ["artist", "genre_count", "betweenness_centrality", "weighted_degree", "avg_popularity"]
    ].head(10).to_string(index=False)
)



community_summary = results.groupby("community").agg(
    community_size=("artist", "count"),
    avg_artist_popularity=("avg_popularity", "mean"),
    avg_weighted_degree=("weighted_degree", "mean"),
    max_importance_score=("importance_score", "max")
).sort_values("community_size", ascending=False)

print("\nCommunity summary:\n")
print(community_summary.head(10).to_string())


# top artist within communities

top_artists_by_community = (
    results.sort_values(["community", "importance_score"], ascending=[True, False])
    .groupby("community")
    .head(1)
    .sort_values("importance_score", ascending=False)
)

print("\nTop artist from each community:\n")
print(
    top_artists_by_community[
        ["community", "artist", "importance_score", "weighted_degree", "avg_popularity", "genre_count"]
    ].head(15).to_string(index=False)
)





#save outputs
results.to_csv("spotify_artist_network_results.csv", index=False)
community_summary.to_csv("spotify_community_summary.csv")

print("\nSaved:")
print("- spotify_artist_network_results.csv")
print("- spotify_community_summary.csv")


