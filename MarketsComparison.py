import pandas as pd
import numpy as np
from DataDiscovery import assign_country
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from DataDiscovery import load_filter_impute_data
import seaborn as sns

############################ How many hotels are available per day per market? #############################

# Loading pricing info
bookings = pd.read_csv('/Users/megoktay/Desktop/Projects/Project-LH/pricing_data.csv')
bookings["arrival_date"] = pd.to_datetime(bookings["arrival_date"])

# Load hotel info
hotels_raw = pd.read_csv('/Users/megoktay/Desktop/Projects/Project-LH/hotels_information.csv')

# Assign each hotel to Belgium or the Netherlands
hotels_raw["country"] = hotels_raw["latitude"].apply(assign_country)

# Keep only columns needed for merging
hotels = hotels_raw[["our_hotel_id", "country"]]

# 3. Merge hotel metadata into bookings dataset
df = bookings.merge(hotels, on="our_hotel_id")

# 4. Compute daily hotel availability per market
availability = (
    df[df["is_sold_out"] == False]
    .groupby(["arrival_date", "country"])["our_hotel_id"]
    .nunique()
    .reset_index(name="available_hotels")
)

availability["arrival_date"] = pd.to_datetime(availability["arrival_date"])

# Plot
plt.figure(figsize=(12, 6))

for country in availability["country"].unique():
    subset = availability[availability["country"] == country]
    plt.plot(
        subset["arrival_date"],
        subset["available_hotels"],
        label=country,
        linewidth=2
    )

# Set a 7 day step to avoid overlapping xlabs
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

plt.xticks(rotation=45)
plt.xlabel("Arrival Date")
plt.ylabel("Available Hotels")
plt.title("Hotel Availability Per Day Per Market")
plt.legend(title="Country")
plt.grid(alpha=0.3)
plt.tight_layout()

# Save
plt.savefig("hotel_availability_per_market.png", dpi=300, bbox_inches="tight")

################# What are the overall pricing behaviors? Are there different patterns to be observed? #################
def clean_pricing(df):
    # Replace empty strings with NaN
    df["price_value_ref"] = df["price_value_ref"].replace(r"^\s*$", np.nan, regex=True)
    df["price_value_non_ref"] = df["price_value_non_ref"].replace(r"^\s*$", np.nan, regex=True)

    # Convert to numeric (invalid entries → NaN)
    df["price_value_ref"] = pd.to_numeric(df["price_value_ref"], errors="coerce")
    df["price_value_non_ref"] = pd.to_numeric(df["price_value_non_ref"], errors="coerce")

    # Remove rows where BOTH prices are missing
    df = df.dropna(subset=["price_value_ref", "price_value_non_ref"], how="all")

    return df

df_hotels = load_filter_impute_data('/Users/megoktay/Desktop/Projects/Project-LH/hotels_information.csv')
df_bookings = pd.read_csv('/Users/megoktay/Desktop/Projects/Project-LH/pricing_data.csv')

df_bookings = clean_pricing(df_bookings)

df_merged = df_bookings.merge(
    df_hotels[["our_hotel_id", "country"]],
    on="our_hotel_id",
    how="left"
)

# Convert refundable + non-refundable columns into long format
price_df = df_merged.melt(
    id_vars=["country"],                      # keep country
    value_vars=["price_value_ref", "price_value_non_ref"],
    var_name="price_type",
    value_name="price"
)

plt.figure(figsize=(10,6))
sns.boxplot(
    data=price_df,
    x="country",
    y="price",
    hue="price_type",
    palette="Set2"
)

plt.title("Price Distribution per Market (Refundable vs Non-refundable)", fontsize=14)
plt.xlabel("Country")
plt.ylabel("Price")
plt.grid(axis="y", alpha=0.3)
plt.legend(title="Price type")
plt.tight_layout()
plt.savefig("price_distribution_per_market.png", dpi=300, bbox_inches="tight")
