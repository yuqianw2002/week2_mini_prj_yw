import unittest
import pandas as pd
import numpy as np
import tempfile
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Import functions from the main module
from bos_temp import (
    load_and_inspect_data,
    process_data,
    train_model,
    evaluate_model,
    plot_predictions
)

class TestBostonWeatherAnalysis(unittest.TestCase):
    """Test suite for Boston weather analysis functions."""

    def setUp(self):
        """Set up test data before each test method."""
        # Create sample test data
        self.sample_data = pd.DataFrame({
            'time': ['2020-01-01', '2020-01-02', '2020-02-01', '2020-12-01'],
            'tavg': [10.0, 12.0, 15.0, 5.0],
            'wspd': [5.0, 7.0, 3.0, 10.0],
            'pres': [1013.0, 1015.0, 1010.0, 1020.0],
            'wdir': [180.0, 200.0, 150.0, 220.0],
            'prcp': [0.0, 2.5, 0.5, 1.0]
        })
        
        # Create a temporary CSV file for testing
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
        self.sample_data.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()
    
    def test_load_and_inspect_data_success(self):
        """Test successful data loading."""
        df = load_and_inspect_data(self.temp_file.name)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 4)
        self.assertIn('tavg', df.columns)
        self.assertIn('wspd', df.columns)
        self.assertIn('time', df.columns)
    
    def test_process_data_(self):
        """Test basic data processing functionality."""
        processed_df, monthly_mean, winter_mean = process_data(self.sample_data)
        
        # Check that time column is converted and dropped
        self.assertNotIn('time', processed_df.columns)
        self.assertIn('Year', processed_df.columns)
        self.assertIn('Month', processed_df.columns)
        
        # Check data types
        self.assertTrue(pd.api.types.is_integer_dtype(processed_df['Year']))
        self.assertTrue(pd.api.types.is_integer_dtype(processed_df['Month']))
        
        # Check monthly mean structure
        self.assertIsInstance(monthly_mean, pd.DataFrame)
        self.assertIn('Year', monthly_mean.columns)
        self.assertIn('Month', monthly_mean.columns)
        
        # Check winter mean structure
        self.assertIsInstance(winter_mean, pd.DataFrame)
        self.assertIn('Year', winter_mean.columns)
        self.assertNotIn('Month', winter_mean.columns)
    
    def test_process_data_winter_months_filtering(self):
        """Test that winter months are correctly filtered."""
        # Create data with specific months
        winter_data = pd.DataFrame({
            'time': ['2020-12-01', '2020-01-01', '2020-02-01', '2020-06-01'],
            'tavg': [5.0, 10.0, 8.0, 25.0],
            'wspd': [10.0, 8.0, 6.0, 3.0],
            'pres': [1020.0, 1015.0, 1010.0, 1005.0],
            'wdir': [220.0, 180.0, 150.0, 100.0]
        })
        
        _, monthly_mean, winter_mean = process_data(winter_data)
        
        # Winter mean should only include months 12, 1, 2
        # The June data (month 6) should be excluded from winter_mean 
        # winter_mean should have only one entry for year 2020
        self.assertEqual(len(winter_mean), 1) 
        
        # Check that winter mean values are correct
        expected_winter_tavg = (5.0 + 10.0 + 8.0) / 3  # Average of winter months
        self.assertAlmostEqual(winter_mean.iloc[0]['tavg'], expected_winter_tavg, places=1)
    
    def test_train_model_(self):
        """Test model training."""
        processed_df, _, _ = process_data(self.sample_data)
        
        model, X_train, X_test, y_train, y_test, y_pred = train_model(
            processed_df, random_state=42
        )
        
        # Check model type
        self.assertIsInstance(model, LinearRegression)
        
        # Check data shapes
        self.assertEqual(len(X_train) + len(X_test), len(processed_df))
        self.assertEqual(len(y_train), len(X_train))
        self.assertEqual(len(y_test), len(X_test))
        self.assertEqual(len(y_pred), len(y_test))
        
        # Check feature columns
        expected_features = ['wspd', 'pres', 'wdir']
        self.assertListEqual(list(X_train.columns), expected_features)
    
    def test_evaluate_model(self):
        """Test model evaluation function."""
        # Create simple test data
        y_test = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])
        
        # Evaluate model
        mse = evaluate_model(y_test, y_pred) 
        
        # Calculate expected MSE
        expected_mse = np.mean((y_test - y_pred) ** 2)
        self.assertAlmostEqual(mse, expected_mse, places=5)
        self.assertGreaterEqual(mse, 0)  
if __name__ == '__main__':
    # run all the tests
    unittest.main() 
