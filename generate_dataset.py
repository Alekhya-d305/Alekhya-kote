import numpy as np
import pandas as pd
np.random.seed(42)
N = 3000
annual_rainfall = np.random.normal(2200, 650, N).clip(400, 4500)
seasonal_rainfall = (annual_rainfall * np.random.uniform(0.55, 0.75, N)
                      + np.random.normal(0, 120, N)).clip(200, 3200)
cloud_visibility = (15 - (seasonal_rainfall / 3200) * 12
                     + np.random.normal(0, 1.5, N)).clip(0.5, 15)
humidity = (55 + (seasonal_rainfall / 3200) * 35
            + np.random.normal(0, 5, N)).clip(20, 100)
temperature = np.random.normal(27, 4, N).clip(10, 42)
river_discharge = (500 + (seasonal_rainfall / 3200) * 4000
                    + np.random.normal(0, 300, N)).clip(50, 6000)
soil_moisture = (30 + (seasonal_rainfall / 3200) * 60
                  + np.random.normal(0, 6, N)).clip(5, 100)
risk_score = (
    0.35 * (seasonal_rainfall / 3200) +
    0.20 * (annual_rainfall / 4500) +
    0.15 * (1 - cloud_visibility / 15) +
    0.15 * (humidity / 100) +
    0.10 * (river_discharge / 6000) +
    0.05 * (soil_moisture / 100)
)
noise = np.random.normal(0, 0.06, N)
flood_prob = risk_score + noise
threshold = np.quantile(flood_prob, 0.55)
flood = (flood_prob > threshold).astype(int)
df = pd.DataFrame({
    "annual_rainfall_mm": annual_rainfall.round(1),
    "seasonal_rainfall_mm": seasonal_rainfall.round(1),
    "cloud_visibility_km": cloud_visibility.round(2),
    "humidity_percent": humidity.round(1),
    "temperature_celsius": temperature.round(1),
    "river_discharge_cusecs": river_discharge.round(1),
    "soil_moisture_percent": soil_moisture.round(1),
    "flood": flood
})
df.to_csv("data/flood_data.csv", index=False)
print("Dataset generated:", df.shape)
print(df["flood"].value_counts())
print(df.head())
