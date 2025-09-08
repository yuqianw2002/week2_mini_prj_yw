import pandas as pd

# read data using df
bos = pd.read_csv("boston_weather_data.csv")

# inspect dataset, the summary stats and data type
print(f"{bos.shape[0]} rows × {bos.shape[1]} columns")
print(f"First 10 columns:\n{bos.head()}")
print(f'Info: \n{bos.info()}')
print(f'Summary stats{bos.describe()}')
print(f"Missing value of each column:\n{bos.isna().sum()}")

# 