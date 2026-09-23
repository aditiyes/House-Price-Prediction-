import numpy as np
import pandas as pd
import os

# Set random seed for reproducibility
np.random.seed(42)

n_samples = 1000

# Feature distributions
square_feet = np.random.normal(loc=1600, scale=600, size=n_samples).astype(int)
square_feet = np.clip(square_feet, 500, 4500)

bedrooms = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.1, 0.3, 0.4, 0.15, 0.05])
bathrooms = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.2, 0.5, 0.25, 0.05])

property_age_years = np.random.randint(0, 35, size=n_samples)
location_score = np.random.randint(1, 11, size=n_samples) # 1 to 10 rating
distance_to_city_center_km = np.round(np.random.exponential(scale=8.0, size=n_samples) + 1.0, 1)
distance_to_city_center_km = np.clip(distance_to_city_center_km, 1.0, 35.0)

garage_spaces = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.25, 0.45, 0.25, 0.05])

# Target variable formulation (price_inr in Lakhs, e.g. 50 Lakhs = 5,000,000 INR)
# Base price formula + noise
base_price = (
    square_feet * 3500 +                       # ₹3,500 per sq ft
    bedrooms * 250000 +                        # ₹2.5 Lakh per bedroom
    bathrooms * 300000 +                       # ₹3 Lakh per bathroom
    location_score * 400000 -                  # ₹4 Lakh per location score point
    property_age_years * 50000 -              # ₹50k depreciation per year
    distance_to_city_center_km * 120000 +     # ₹1.2 Lakh discount per km distance
    garage_spaces * 200000 +                  # ₹2 Lakh per garage space
    1500000                                    # Base constant ₹15 Lakhs
)

# Add normal noise (standard deviation ₹5 Lakhs)
noise = np.random.normal(loc=0, scale=500000, size=n_samples)
price_inr = np.round(base_price + noise, -3) # round to thousands
price_inr = np.clip(price_inr, 1500000, 25000000) # clip between 15 Lakhs and 2.5 Crores

df = pd.DataFrame({
    'square_feet': square_feet,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'property_age_years': property_age_years,
    'location_score': location_score,
    'distance_to_city_center_km': distance_to_city_center_km,
    'garage_spaces': garage_spaces,
    'price_inr': price_inr
})

os.makedirs('data', exist_ok=True)
df.to_csv('data/house_prices.csv', index=False)
print(f"Dataset generated successfully with {len(df)} records.")
print(df.head())
print(df.describe())
