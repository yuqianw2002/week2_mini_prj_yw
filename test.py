import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from bos_temp_refactored import BostonWeatherAnalyzer


class TestBostonWeatherSimple(unittest.TestCase):
    """Simple test cases for Boston Weather Analysis"""

    def setUp(self):
        """Set up test data before each test"""
        self.analyzer = BostonWeatherAnalyzer()

        # Create simple test data
        self.test_data = pd.DataFrame({
            'time': ['2020-01-01', '2020-01-02', '2020-02-01', '2020-12-01'],
            'tavg': [25.0, 30.0, 35.0, 20.0],
            'wspd': [10.0, 15.0, 12.0, 8.0],
            'pres': [1013.0, 1015.0, 1012.0, 1010.0],
            'wdir': [180.0, 190.0, 200.0, 170.0]
        })

    def test_load_data(self):
        """Test data loading functionality"""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.test_data.to_csv(f.name, index=False)
            temp_file = f.name

        try:
            # Test loading
            result = self.analyzer.load_data(temp_file)
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), 4)
            self.assertIn('time', result.columns)
        finally:
            os.unlink(temp_file)

    def test_load_data_file_not_found(self):
        """Test error handling when file doesn't exist"""
        with self.assertRaises(FileNotFoundError):
            self.analyzer.load_data("nonexistent_file.csv")

    def test_preprocess_datetime(self):
        """Test datetime preprocessing"""
        result = self.analyzer.preprocess_datetime(self.test_data)

        # Check new columns exist
        self.assertIn('Year', result.columns)
        self.assertIn('Month', result.columns)
        self.assertNotIn('time', result.columns)

        # Check values
        self.assertEqual(result['Year'].iloc[0], 2020)
        self.assertEqual(result['Month'].iloc[0], 1)

    def test_calculate_monthly_mean(self):
        """Test monthly mean calculation"""
        processed_data = self.analyzer.preprocess_datetime(self.test_data)
        result = self.analyzer.calculate_monthly_mean(processed_data)

        # Should have 3 months (Jan, Feb, Dec)
        self.assertEqual(len(result), 3)

        # Check January average (first two rows: 25 + 30 = 55, avg = 27.5)
        jan_data = result[result['Month'] == 1]
        self.assertAlmostEqual(jan_data['tavg'].iloc[0], 27.5)

    def test_filter_winter_months(self):
        """Test winter month filtering"""
        processed_data = self.analyzer.preprocess_datetime(self.test_data)
        monthly_data = self.analyzer.calculate_monthly_mean(processed_data)

        winter_data = self.analyzer.filter_winter_months(monthly_data)

        # Should have January and December
        winter_months = winter_data['Month'].tolist()
        self.assertIn(1, winter_months)
        self.assertIn(12, winter_months)
        self.assertNotIn(2, winter_months)  # February not in default winter

    def test_impute_missing_values(self):
        """Test missing value imputation"""
        # Add missing values
        data_with_missing = self.test_data.copy()
        data_with_missing.loc[0, 'tavg'] = np.nan

        result = self.analyzer.impute_missing_values(data_with_missing)

        # Check no missing values remain
        self.assertEqual(result.isnull().sum().sum(), 0)

    def test_prepare_model_data(self):
        """Test model data preparation"""
        features = ['wspd', 'pres', 'wdir']
        target = 'tavg'

        X_train, X_test, y_train, y_test = self.analyzer.prepare_model_data(
            self.test_data, features, target, test_size=0.5
        )

        # Check shapes
        self.assertEqual(len(X_train) + len(X_test), 4)
        self.assertEqual(X_train.shape[1], 3)  # Three features
        self.assertEqual(list(X_train.columns), features)

    def test_train_and_evaluate_model(self):
        """Test model training and evaluation"""
        features = ['wspd', 'pres', 'wdir']
        target = 'tavg'

        X_train, X_test, y_train, y_test = self.analyzer.prepare_model_data(
            self.test_data, features, target, test_size=0.5, random_state=42
        )

        # Train model
        model = self.analyzer.train_model(X_train, y_train)
        self.assertIsNotNone(model)

        # Evaluate model
        results = self.analyzer.evaluate_model(model, X_test, y_test)

        self.assertIn('mse', results)
        self.assertIn('rmse', results)
        self.assertGreaterEqual(results['mse'], 0)  # MSE should be non-negative


if __name__ == '__main__':
    unittest.main(verbosity=2)