import pandas as pd
import numpy as np

# read data using df
bos = pd.read_csv("boston_weather_data.csv")

# inspect dataset, the summary stats and data type
print(f"{bos.shape[0]} rows × {bos.shape[1]} columns")
print(f"First 10 columns:\n{bos.head()}")
print(f'Info: \n{bos.info()}')
print(f'Summary stats{bos.describe()}')
print(f"Num of missing value of each column:\n{bos.isna().sum()}")

# convert the time col string into datetime, and create year and month col
bos["time"] = pd.to_datetime(bos["time"])
bos["Year"]  = bos["time"].dt.year
bos["Month"] = bos["time"].dt.month

# drop time col
bos = bos.drop('time', axis=1)

# group by each year and month, and take mean of columns except date data
monthly_mean = (
    bos.groupby(["Year", "Month"], as_index=False)
      .mean(numeric_only=True)    
      .sort_values(["Year", "Month"])
)

print(monthly_mean.head(12))
