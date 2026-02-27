import kagglehub
from kagglehub import KaggleDatasetAdapter
import kagglehub
import pandas as pd


# Download latest version
#path = kagglehub.dataset_download("kabhishm/best-selling-nintendo-switch-video-games")
#print("Path to dataset files:", path)





# Load the latest version
file_path = "/Users/juniorbenitez/Desktop/INST414/best_selling_switch_games.csv"
df = pd.read_csv(file_path)

print(df.head())

file_path = "/Users/juniorbenitez/Desktop/INST414/best_selling_switch_games.csv"
df = pd.read_csv(file_path)

# Sort by copies_sold descending and take top 5
top_7_games = df.sort_values(by="copies_sold", ascending=False).head(7)

# Display the title and number of copies sold
print("Top 5 Best-Selling Nintendo Switch Games (Copies Sold in Millions):")
print(top_7_games[["title", "copies_sold"]])



sales_by_genre = df.groupby("genre")["copies_sold"].sum()

# Sort descending to get top 5 genres
top_5_genres = sales_by_genre.sort_values(ascending=False).head(5)

# Display top 5 genres with total copies sold
print("Top 5 Selling Genres with Total Copies Sold:")
print(top_5_genres)


############################
sales_by_developer = df.groupby("developer")["copies_sold"].sum()

# Sort descending and take top 5
top_5_developers = sales_by_developer.sort_values(ascending=False).head(5)

print("Top 5 Developers by Total Copies Sold (Millions):")
print(top_5_developers)











###########################
# Group by publisher and sum total copies sold
sales_by_publisher = df.groupby("publisher")["copies_sold"].sum()

# Sort descending and take top 5
top_5_publishers = sales_by_publisher.sort_values(ascending=False).head(5)

print("Top 5 Publishers by Total Copies Sold (Millions):")
print(top_5_publishers)



########################## good

# Standardize and manually combine messy genres
genre_cleaning = {
    "Exergamerole-playing": "Exergame",
    "Exergamerhythm": "Exergame",
    "Platformercompilation": "Platformer",
    "PlatformerLevel editor": "Platformer",
    "Role-playingaction-adventure": "Role-playing",
    "Tactical role-playing": "Role-playing",
    "Action role-playing": "Role-playing",
    "Hack and slashRole-playing": "Role-playing",
    "Simulationrole-playing": "Role-playing",
    "Sandboxsurvival": "Survival",
    "Partysocial deduction": "Party",
    "Real-time strategypuzzle": "Puzzle",
    "Kart racingaugmented reality": "Kart racing",
    "Action-adventure, Hack and Slash": "Action-adventure"
}

# Replace messy genres
df["genre"] = df["genre"].replace(genre_cleaning)


sales_by_genre = (
    df.groupby("genre")["copies_sold"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

print("Cleaned Genres and Total Copies Sold (Millions):")
print(sales_by_genre)


# Clean publisher inconsistencies
publisher_cleaning = {
    "The Pokémon CompanyNintendo": "Nintendo",
    "JP: The Pokémon CompanyNA/PAL: Nintendo": "Nintendo",
    "The Pokémon Company": "Nintendo"
}

df["publisher"] = df["publisher"].replace(publisher_cleaning)


top_3_publishers = (
    df.groupby("publisher")["copies_sold"]
    .sum()
    .sort_values(ascending=False)
    .head(3)
)

print("Top 3 Publishers:")
print(top_3_publishers)


# Filter dataset for only top 3 publishers
top_publishers_df = df[df["publisher"].isin(top_3_publishers.index)]

# Group by publisher AND genre
publisher_genre_sales = (
    top_publishers_df
    .groupby(["publisher", "genre"])["copies_sold"]
    .sum()
    .reset_index()
)




# For each publisher, get the genre with max sales
print("max sells of each publisher by genre")
top_genre_per_publisher = (
    publisher_genre_sales
    .sort_values(["publisher", "copies_sold"], ascending=[True, False])
    .groupby("publisher")
    .head(1)
)

print("\nTop Genre for Each of the Top 3 Publishers:")
print(top_genre_per_publisher)

developer_genre_sales = (
    df.groupby(["developer", "genre"])["copies_sold"]
    .sum()
    .reset_index()
)

top_genre_per_dev = (
    developer_genre_sales
    .loc[developer_genre_sales.groupby("developer")["copies_sold"].idxmax()]
    .sort_values("copies_sold", ascending=False)
)

print(top_genre_per_dev.head(10))


#################################



avg_sales_per_genre = (
    df.groupby("genre")["copies_sold"]
    .agg(
        avg_copies_sold="mean",
        total_copies_sold="sum",
        number_of_games="count"
    )
    .sort_values("avg_copies_sold", ascending=False)
    .reset_index()
)


avg_sales_per_genre["avg_copies_sold"] = avg_sales_per_genre["avg_copies_sold"].map("{:,.0f}".format)
avg_sales_per_genre["total_copies_sold"] = avg_sales_per_genre["total_copies_sold"].map("{:,.0f}".format)

print(avg_sales_per_genre)