"""
Tests for data validation utilities.
"""

import unittest
import pandas as pd
import numpy as np
from src.utils.validators import DataValidator, ModelValidator, validate_business_rules


class TestDataValidator(unittest.TestCase):
    """Test data validation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample price data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        self.valid_price_data = pd.DataFrame({
            'Open': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(150, 250, 100),
            'Low': np.random.uniform(50, 150, 100),
            'Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # Ensure OHLC logic is correct
        for i in range(len(self.valid_price_data)):
            row = self.valid_price_data.iloc[i]
            high = max(row['Open'], row['Close']) + np.random.uniform(0, 10)
            low = min(row['Open'], row['Close']) - np.random.uniform(0, 10)
            self.valid_price_data.iloc[i, self.valid_price_data.columns.get_loc('High')] = high
            self.valid_price_data.iloc[i, self.valid_price_data.columns.get_loc('Low')] = low
    
    def test_validate_price_data_valid(self):
        """Test validation of valid price data."""
        result = DataValidator.validate_price_data(self.valid_price_data)
        self.assertTrue(result)
    
    def test_validate_price_data_empty(self):
        """Test validation fails for empty data."""
        empty_data = pd.DataFrame()
        with self.assertRaises(ValueError):
            DataValidator.validate_price_data(empty_data)
    
    def test_validate_price_data_missing_columns(self):
        """Test validation fails for missing columns."""
        incomplete_data = self.valid_price_data.drop('Close', axis=1)
        with self.assertRaises(ValueError):
            DataValidator.validate_price_data(incomplete_data)
    
    def test_validate_price_data_negative_prices(self):
        """Test validation fails for negative prices."""
        invalid_data = self.valid_price_data.copy()
        invalid_data.iloc[0, invalid_data.columns.get_loc('Close')] = -10
        with self.assertRaises(ValueError):
            DataValidator.validate_price_data(invalid_data)
    
    def test_validate_returns_valid(self):
        """Test validation of valid returns."""
        returns = pd.Series(np.random.normal(0, 0.02, 100))
        result = DataValidator.validate_returns(returns)
        self.assertTrue(result)
    
    def test_validate_returns_empty(self):
        """Test validation fails for empty returns."""
        empty_returns = pd.Series(dtype=float)
        with self.assertRaises(ValueError):
            DataValidator.validate_returns(empty_returns)
    
    def test_validate_returns_infinite(self):
        """Test validation fails for infinite returns."""
        invalid_returns = pd.Series([0.1, np.inf, 0.05])
        with self.assertRaises(ValueError):
            DataValidator.validate_returns(invalid_returns)
    
    def test_validate_portfolio_weights_valid(self):
        """Test validation of valid portfolio weights."""
        weights = {'TSLA': 0.3, 'SPY': 0.5, 'BND': 0.2}
        result = DataValidator.validate_portfolio_weights(weights)
        self.assertTrue(result)
    
    def test_validate_portfolio_weights_negative(self):
        """Test validation fails for negative weights."""
        weights = {'TSLA': -0.1, 'SPY': 0.6, 'BND': 0.5}
        with self.assertRaises(ValueError):
            DataValidator.validate_portfolio_weights(weights)
    
    def test_validate_portfolio_weights_not_sum_to_one(self):
        """Test validation fails when weights don't sum to 1."""
        weights = {'TSLA': 0.3, 'SPY': 0.5, 'BND': 0.3}  # Sum = 1.1
        with self.assertRaises(ValueError):
            DataValidator.validate_portfolio_weights(weights)
    
    def test_validate_covariance_matrix_valid(self):
        """Test validation of valid covariance matrix."""
        # Create a valid covariance matrix
        data = np.random.multivariate_normal([0, 0, 0], [[1, 0.5, 0.3], [0.5, 1, 0.2], [0.3, 0.2, 1]], 100)
        cov_matrix = pd.DataFrame(np.cov(data.T), columns=['A', 'B', 'C'], index=['A', 'B', 'C'])
        result = DataValidator.validate_covariance_matrix(cov_matrix)
        self.assertTrue(result)
    
    def test_validate_covariance_matrix_not_square(self):
        """Test validation fails for non-square matrix."""
        invalid_cov = pd.DataFrame(np.random.rand(3, 2))
        with self.assertRaises(ValueError):
            DataValidator.validate_covariance_matrix(invalid_cov)


class TestModelValidator(unittest.TestCase):
    """Test model validation functionality."""
    
    def test_validate_forecast_inputs_valid(self):
        """Test validation of valid forecast inputs."""
        data = pd.Series(np.random.randn(200))
        result = ModelValidator.validate_forecast_inputs(data)
        self.assertTrue(result)
    
    def test_validate_forecast_inputs_insufficient_data(self):
        """Test validation fails for insufficient data."""
        data = pd.Series(np.random.randn(50))
        with self.assertRaises(ValueError):
            ModelValidator.validate_forecast_inputs(data)
    
    def test_validate_forecast_inputs_zero_variance(self):
        """Test validation fails for zero variance data."""
        data = pd.Series([1.0] * 200)
        with self.assertRaises(ValueError):
            ModelValidator.validate_forecast_inputs(data)
    
    def test_validate_forecast_outputs_valid(self):
        """Test validation of valid forecast outputs."""
        predictions = np.random.randn(30)
        ci = np.column_stack([predictions - 0.1, predictions + 0.1])
        result = ModelValidator.validate_forecast_outputs(predictions, ci)
        self.assertTrue(result)
    
    def test_validate_forecast_outputs_empty(self):
        """Test validation fails for empty predictions."""
        predictions = np.array([])
        with self.assertRaises(ValueError):
            ModelValidator.validate_forecast_outputs(predictions)
    
    def test_validate_forecast_outputs_invalid_ci(self):
        """Test validation fails for invalid confidence intervals."""
        predictions = np.random.randn(30)
        ci = np.column_stack([predictions + 0.1, predictions - 0.1])  # Lower > Upper
        with self.assertRaises(ValueError):
            ModelValidator.validate_forecast_outputs(predictions, ci)


class TestBusinessRules(unittest.TestCase):
    """Test business rule validation."""
    
    def test_validate_business_rules_valid(self):
        """Test validation of valid business rules."""
        weights = {'TSLA': 0.3, 'SPY': 0.4, 'BND': 0.3}
        config = {'max_single_asset_weight': 0.5, 'min_assets': 2}
        result = validate_business_rules(weights, config)
        self.assertTrue(result)
    
    def test_validate_business_rules_concentration_limit(self):
        """Test validation fails for concentration limit violation."""
        weights = {'TSLA': 0.8, 'SPY': 0.2}
        config = {'max_single_asset_weight': 0.5, 'min_assets': 2}
        with self.assertRaises(ValueError):
            validate_business_rules(weights, config)
    
    def test_validate_business_rules_min_diversification(self):
        """Test validation fails for insufficient diversification."""
        weights = {'TSLA': 1.0}
        config = {'max_single_asset_weight': 1.0, 'min_assets': 2}
        with self.assertRaises(ValueError):
            validate_business_rules(weights, config)


if __name__ == '__main__':
    unittest.main()
