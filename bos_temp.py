import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_inspect_data(file_path):
    """ Load and inspect the Boston weather dataset."""
    # Read data
    df = pd.read_csv(file_path)

    # Print inspection results
    print(f"{df.shape[0]} rows × {df.shape[1]} columns")
    print(f"First 10 columns:\n{df.head()}")
    print(f'Info: \n{df.info()}')
    print(f'Summary stats:\n{df.describe()}')
    print(f"Number of missing values in each column:\n{df.isna().sum()}")

    return df


def process_clean_data(df):
    """Process the weather data by converting time column, creating date features,
    and calculating monthly/winter means. Also handle missing values."""
    # Make a copy to avoid modifying the original
    processed_df = df.copy()

    # Convert time column to datetime and extract year/month
    processed_df["time"] = pd.to_datetime(processed_df["time"])
    processed_df["Year"] = processed_df["time"].dt.year
    processed_df["Month"] = processed_df["time"].dt.month

    # Drop time column
    processed_df = processed_df.drop('time', axis=1)

    # Group by year and month, calculate mean
    monthly_mean = (
        processed_df.groupby(["Year", "Month"], as_index=False).mean()
        .sort_values(["Year", "Month"])
    )

    # Filter winter months (12, 1, 2) and calculate winter means
    winter_df = monthly_mean[monthly_mean["Month"].isin([12, 1, 2])]
    winter_mean = (
        winter_df.groupby(["Year"], as_index=False).mean()
        .sort_values(["Year"])
        .drop('Month', axis=1)
    )

    # Impute missing values with column means for numeric columns
    num_cols = processed_df.select_dtypes(include='number').columns
    processed_df[num_cols] = processed_df[num_cols].fillna(processed_df[num_cols].mean())

    return processed_df, monthly_mean, winter_mean


def train_model(df, features=['wspd', 'pres', 'wdir'], target='tavg', test_size=0.2, random_state=42):
    """Train a linear regression model on the weather data."""
    # Select features and target
    X = df[features]
    y = df[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Initialize and train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    return model, X_train, X_test, y_train, y_test, y_pred


def evaluate_model(y_test, y_pred):
    """Evaluate the model performance.
    """
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")
    return mse


def plot_predictions(X_test, y_test, y_pred, feature_name='wspd', target_name='tavg', save_path='test_pred.png'):
    """Create a scatter plot comparing actual vs predicted values."""
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test[feature_name], y_test, color='blue', label='Actual', alpha=0.6)
    plt.scatter(X_test[feature_name], y_pred, color='red', label='Prediction', alpha=0.6)
    plt.title(f"Actual {target_name} vs. Predicted {target_name} - Boston Weather")
    plt.xlabel(feature_name.title())
    plt.ylabel(target_name.title())
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')


def main():
    # Load and inspect data
    df = load_and_inspect_data("boston_weather_data.csv")

    # Process data
    processed_df, monthly_mean, winter_mean = process_data(df)

    print("Monthly mean (first 5 rows):")
    print(monthly_mean.head())
    print("\nWinter mean by year:")
    print(winter_mean)

    # Train model
    model, X_train, X_test, y_train, y_test, y_pred = train_model(processed_df)

    # Evaluate model
    mse = evaluate_model(y_test, y_pred)

    # Plot results
    plot_predictions(X_test, y_test, y_pred)

    return {
        'model': model,
        'mse': mse,
        'processed_df': processed_df,
        'monthly_mean': monthly_mean,
        'winter_mean': winter_mean
    }


if __name__ == "__main__":
    main()
