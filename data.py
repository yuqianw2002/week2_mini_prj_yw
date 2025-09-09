import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

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
    bos.groupby(["Year", "Month"], as_index=False).mean()  
      .sort_values(["Year", "Month"])
)
print(monthly_mean.head())

# filtering Winter month (12-2), get mean value of winter month
winter_df = monthly_mean[monthly_mean["Month"].isin([12, 1, 2])]
winter_mean = (winter_df.groupby(["Year"], as_index=False).mean().sort_values(["Year"]))
print(winter_mean.drop('Month', axis=1))

# linear regression on the data
mdl = LinearRegression()

# impute missing value with mean of the col
num_cols = bos.select_dtypes(include='number').columns
bos[num_cols] = bos[num_cols].fillna(bos[num_cols].mean())
print(bos.isnull().sum())

# select features 
# features = ['prcp', 'wdir', 'wspd', 'pres']
features = ['wspd', 'pres', 'wdir']
target = ['tavg']
X = bos[features]
y = bos[target]

# split data into 20% test size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# fit the mdl
mdl.fit(X_train, y_train)

# predict test value
y_pred = mdl.predict(X_test)
# y_pred = pd.DataFrame(y_pred)

# evaluate model
mse = mean_squared_error(y_pred, y_test)
print("The mean squared error:", mse)

# plot the true value and pred value
plt.scatter(X_test.iloc[:,0], y_test, color='blue', label='Actual')
plt.scatter(X_test.iloc[:,0], y_pred, color='red', label='Prediction')
plt.title("Actual temperature vs. Prediction temperaturte Boston winter")
plt.xlabel('Wind speed')
plt.ylabel('Temperature')
plt.legend()
plt.savefig('test_pred.png')
plt.show()
