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



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os


def load_inspect_data(filepath):
    """Load and inspect dataset from CSV file. 
    Inspect dataset and print summary information.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    try:
        data = pd.read_csv(filepath)
        return data
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


def inspect_dataset(data):
    """
    Inspect dataset and print summary information.
    
    Args:
        data: DataFrame to inspect
    """
    print(f"{data.shape[0]} rows × {data.shape[1]} columns")
    print(f"First 10 columns:\n{data.head()}")
    print(f'Info: \n{data.info()}')
    print(f'Summary stats:\n{data.describe()}')
    print(f"Number of missing values per column:\n{data.isna().sum()}")


def preprocess_datetime(data, time_col='time'):
    """
    Convert time column to datetime and extract year/month.
    
    Args:
        data: DataFrame containing time column
        time_col: Name of the time column
        
    Returns:
        DataFrame with processed datetime columns
    """
    if time_col not in data.columns:
        raise KeyError(f"Column '{time_col}' not found in data")
    
    processed_data = data.copy()
    
    try:
        processed_data[time_col] = pd.to_datetime(processed_data[time_col])
        processed_data["Year"] = processed_data[time_col].dt.year
        processed_data["Month"] = processed_data[time_col].dt.month
        processed_data = processed_data.drop(time_col, axis=1)
        return processed_data
    except Exception as e:
        raise ValueError(f"Error converting time column: {str(e)}")


def calculate_monthly_mean(data):
    """
    Group data by year and month, calculate mean values.
    
    Args:
        data: DataFrame with Year and Month columns
        
    Returns:
        DataFrame with monthly mean values
    """
    required_cols = ['Year', 'Month']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")
    
    monthly_mean = (
        data.groupby(["Year", "Month"], as_index=False).mean()
            .sort_values(["Year", "Month"])
    )
    return monthly_mean


def filter_winter_months(data, winter_months=[12, 1, 2]):
    """
    Filter data for winter months.
    
    Args:
        data: DataFrame with Month column
        winter_months: List of month numbers considered as winter
        
    Returns:
        DataFrame filtered for winter months
    """
    if 'Month' not in data.columns:
        raise KeyError("Month column not found in data")
    
    winter_df = data[data["Month"].isin(winter_months)]
    return winter_df


def calculate_winter_mean(winter_data):
    """
    Calculate yearly mean for winter data.
    
    Args:
        winter_data: DataFrame with winter month data
        
    Returns:
        DataFrame with winter yearly means
    """
    if 'Year' not in winter_data.columns:
        raise KeyError("Year column not found in data")
    
    winter_mean = (
        winter_data.groupby(["Year"], as_index=False).mean()
                   .sort_values(["Year"])
    )
    # Drop Month column if it exists (since it's meaningless after grouping)
    if 'Month' in winter_mean.columns:
        winter_mean = winter_mean.drop('Month', axis=1)
    return winter_mean


def impute_missing_values(data, strategy='mean'):
    """
    Impute missing values in numeric columns.
    
    Args:
        data: DataFrame with potential missing values
        strategy: Imputation strategy ('mean', 'median')
        
    Returns:
        DataFrame with imputed values
    """
    if strategy not in ['mean', 'median']:
        raise ValueError("Strategy must be 'mean' or 'median'")
    
    data_imputed = data.copy()
    num_cols = data_imputed.select_dtypes(include='number').columns
    
    if strategy == 'mean':
        data_imputed[num_cols] = data_imputed[num_cols].fillna(data_imputed[num_cols].mean())
    elif strategy == 'median':
        data_imputed[num_cols] = data_imputed[num_cols].fillna(data_imputed[num_cols].median())
    
    return data_imputed


def prepare_model_data(data, features, target, test_size=0.2, random_state=None):
    """
    Prepare data for machine learning model.
    
    Args:
        data: DataFrame containing features and target
        features: List of feature column names
        target: Target column name
        test_size: Proportion of test data
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        raise KeyError(f"Missing feature columns: {missing_features}")
    
    if target not in data.columns:
        raise KeyError(f"Target column '{target}' not found")
    
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    
    X = data[features]
    y = data[target]
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_model(X_train, y_train):
    """
    Train linear regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        Trained LinearRegression model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary containing evaluation metrics and predictions
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'predictions': y_pred
    }


def create_prediction_plot(X_test, y_test, y_pred, save_path='test_pred.png'):
    """
    Create scatter plot comparing actual vs predicted values.
    
    Args:
        X_test: Test features
        y_test: Actual test values
        y_pred: Predicted values
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test.iloc[:, 0], y_test, color='blue', label='Actual', alpha=0.6)
    plt.scatter(X_test.iloc[:, 0], y_pred, color='red', label='Prediction', alpha=0.6)
    plt.title("Actual temperature vs. Prediction temperature Boston winter")
    plt.xlabel('Wind speed')
    plt.ylabel('Temperature')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.show()


def main():
    """Main function to run the complete analysis pipeline."""
    try:
        # Load data
        print("Loading data...")
        bos = load_data("boston_weather_data.csv")
        
        # Inspect dataset
        inspect_dataset(bos)
        
        # Preprocess datetime
        print("\nPreprocessing datetime...")
        bos = preprocess_datetime(bos)
        
        # Calculate monthly means
        print("Calculating monthly means...")
        monthly_mean = calculate_monthly_mean(bos)
        print(monthly_mean.head())
        
        # Filter winter months and calculate winter means
        print("\nFiltering winter months...")
        winter_df = filter_winter_months(monthly_mean)
        winter_mean = calculate_winter_mean(winter_df)
        print(winter_mean)
        
        # Impute missing values
        print("\nImputing missing values...")
        bos = impute_missing_values(bos)
        print("Missing values after imputation:")
        print(bos.isnull().sum())
        
        # Prepare data for modeling
        features = ['wspd', 'pres', 'wdir']
        target = 'tavg'
        
        print(f"\nPreparing model data with features: {features}")
        X_train, X_test, y_train, y_test = prepare_model_data(
            bos, features, target, test_size=0.2, random_state=42
        )
        
        # Train model
        print("Training model...")
        mdl = train_model(X_train, y_train)
        
        # Evaluate model
        print("Evaluating model...")
        results = evaluate_model(mdl, X_test, y_test)
        print(f"Mean Squared Error: {results['mse']:.4f}")
        
        # Create plot
        print("Creating prediction plot...")
        create_prediction_plot(X_test, y_test, results['predictions'])
        
        print("Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        raise


if __name__ == "__main__":
    main()