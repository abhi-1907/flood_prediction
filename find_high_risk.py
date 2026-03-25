import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df = pd.read_csv("india_pakistan_flood_balancednew.csv")
print(f"Dataset Shape: {df.shape}")
print(f"Class Distribution:\n{df['flood_occurred'].value_counts(normalize=True)}")

# Filter for actual floods
floods = df[df['flood_occurred'] == 1]
high_rain = floods.sort_values(by=['rain_mm_weekly'], ascending=False).head(5)

print("\nTop 5 High Weekly Rainfall Flood Events:")
print(high_rain[['name', 'rain_mm_weekly', 'rain_mm_monthly', 'elevation_m', 'dist_major_river_km', 'waterbody_nearby', 'flood_occurred']])

# Low Weekly Rainfall but still a flood?
low_rain_flood = floods.sort_values(by=['rain_mm_weekly'], ascending=True).head(5)
print("\nTop 5 LOW Weekly Rainfall Flood Events (Anomalies?):")
print(low_rain_flood[['name', 'rain_mm_weekly', 'rain_mm_monthly', 'elevation_m', 'dist_major_river_km', 'waterbody_nearby', 'flood_occurred']])
