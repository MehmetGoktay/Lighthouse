import numpy as np
import pandas as pd
from pandas.core.config_init import pc_max_info_rows_doc
from scipy.stats import ttest_ind, mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def load_filter_impute_data(file_path):
    # Read the CSV file and store it in a DataFrame
    df = pd.read_csv(file_path)
    # print(df.describe())
    # print(df.isna().sum())
    # print(df.info())
    # Fill empty values with NaN and drop hotels with missing review_scores
    df["review_score"] = df["review_score"].replace(r"^\s*$", np.nan, regex=True)
    df.loc[df["review_score"] < 0, "review_score"] = np.nan
    df = df.dropna(subset=["review_score"])

    # # Impute stars with mode
    df["stars"] = df["stars"].replace(r"^\s*$", np.nan, regex=True)
    df["stars"] = df["stars"].fillna(df["stars"].mode()[0])

    # # Impute room_count with median
    df["room_count"] = df["room_count"].replace(r"^\s*$", np.nan, regex=True)
    df["room_count"] = df["room_count"].fillna(df["room_count"].median())
    tweede = df.copy()
    df["country"] = df["latitude"].apply(assign_country)
    print(df.info())

    return df

# Assign country name based on latitude and longitude
def assign_country(latitude):
    if latitude > 51.5:
        return "Netherlands"
    else:
        return "Belgium"

# Calculate k-means clustering
def kmeans_clustering(dataframe):

    df = dataframe.copy()
    features = ["review_score", "stars", "room_count", "latitude", "longitude"]
    X = df[features].copy()
    X_scaled = StandardScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=2, random_state=42)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    centroids = df.groupby("cluster")[["review_score", "stars", "room_count", "latitude", "longitude"]].mean()
    print(centroids)
# Plot PCA
def plot_pca(dataframe):
    df = dataframe.copy()
    features = ["review_score", "stars", "room_count", "latitude", "longitude"]
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    pca_components = pca.fit_transform(X_scaled)

    df["PC1"] = pca_components[:, 0]
    df["PC2"] = pca_components[:, 1]

    plt.figure(figsize=(8, 6))
    plt.scatter(df["PC1"], df["PC2"], c=df["country"].map({"Belgium": 0, "Netherlands": 1}), cmap="viridis", alpha=0.8)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig("pca.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    # Load the data, perform data cleaning and imputation
    df_hotels = load_filter_impute_data('hotels_information.csv')

    # Create two dataframes for Belgium and The Netherlands
    belgium_hotels = df_hotels[df_hotels["country"] == "Belgium"]
    netherlands_hotels = df_hotels[df_hotels["country"] == "Netherlands"]

    # T-test for review_score
    t_stat, p_value = ttest_ind(belgium_hotels["review_score"], netherlands_hotels["review_score"], equal_var=False)
    print('t-statistics', t_stat)
    print('p-value', p_value)

    # Mann-Whitney U test for stars
    u_stat, p_value = mannwhitneyu(belgium_hotels["stars"], netherlands_hotels["stars"], alternative="two-sided")
    print('u-statistics', u_stat)
    print('p-value', p_value)

    # Mann-Whitney U test for room count
    u_stat, p_value = mannwhitneyu(belgium_hotels["room_count"], netherlands_hotels["room_count"], alternative="two-sided")
    print('u-statistics', u_stat)
    print('p-value', p_value)

    # Performing correlation (Spearman) for Belgium and The Netherlands
    corr_spearman = df_hotels.groupby("country")[["review_score", "stars", "room_count"]].corr(method="spearman")
    print(corr_spearman)

    # Kmeans clustering
    kmeans_clustering(df_hotels)

    # PCA plotting
    plot_pca(df_hotels)

    # ### What are the distributions of hotels when it comes to stars and review scores?

    # Plot distributions of star ratings per country
    plt.figure(figsize=(10,5))
    sns.countplot(data=df_hotels, x="stars", hue="country", palette="Set2")
    plt.title("Distribution of Star Ratings per Country")
    plt.xlabel("Star Rating")
    plt.ylabel("Number of Hotels")
    plt.savefig("star_ratings_per_country.png", dpi=300, bbox_inches="tight")

    # Plot review score distribution per country
    plt.figure(figsize=(10,5))
    sns.histplot(data=df_hotels, x="review_score", kde=True, hue="country", bins=15, palette="Set2", alpha=0.5)
    plt.title("Review Score Distribution per Country")
    plt.xlabel("Review Score")
    plt.ylabel("Number of Hotels")
    plt.savefig("review_scores_per_country.png", dpi=300, bbox_inches="tight")

    # Plot distribution of room counts
    plt.figure(figsize=(10,5))
    sns.histplot(data=df_hotels, x="room_count", kde=True, bins=15, color="teal", alpha=0.6)
    plt.title("Distribution of Room Counts")
    plt.xlabel("Room Count")
    plt.ylabel("Number of Hotels")
    plt.savefig("distribution_of_room_counts_all.png", dpi=300, bbox_inches="tight")

    # Plot room count distribution per country
    plt.figure(figsize=(10,5))
    sns.histplot(data=df_hotels, x="room_count",kde=True,bins=15,hue="country",alpha=0.5,palette="Set2")
    plt.title("Room Count Distribution per Country")
    plt.xlabel("Room Count")
    plt.ylabel("Number of Hotels")
    plt.savefig("distribution_of_room_counts_per_country.png", dpi=300, bbox_inches="tight")

