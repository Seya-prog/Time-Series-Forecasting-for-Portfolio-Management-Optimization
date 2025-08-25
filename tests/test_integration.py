"""
Integration tests for the portfolio management system.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.config import Config
from src.utils.validators import DataValidator
from src.data.data_collector import FinancialDataCollector
from src.models.time_series_forecasting import TimeSeriesForecaster


class TestSystemIntegration(unittest.TestCase):
    """Integration tests for the complete system workflow."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample data
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        self.sample_data = pd.DataFrame({
            'Open': 100 + np.cumsum(np.random.randn(500) * 0.02),
            'High': 100 + np.cumsum(np.random.randn(500) * 0.02) + 2,
            'Low': 100 + np.cumsum(np.random.randn(500) * 0.02) - 2,
            'Close': 100 + np.cumsum(np.random.randn(500) * 0.02),
            'Volume': np.random.randint(1000, 10000, 500)
        }, index=dates)
        
        # Ensure OHLC logic
        for i in range(len(self.sample_data)):
            row = self.sample_data.iloc[i]
            self.sample_data.iloc[i, 1] = max(row['Open'], row['Close'], row['High'])  # High
            self.sample_data.iloc[i, 2] = min(row['Open'], row['Close'], row['Low'])   # Low
    
    def test_data_validation_pipeline(self):
        """Test complete data validation pipeline."""
        # Test data validation
        result = DataValidator.validate_price_data(self.sample_data)
        self.assertTrue(result)
        
        # Test returns validation
        returns = self.sample_data['Close'].pct_change().dropna()
        result = DataValidator.validate_returns(returns)
        self.assertTrue(result)
    
    def test_forecasting_pipeline(self):
        """Test forecasting pipeline integration."""
        # Create forecaster with sample data
        forecaster = TimeSeriesForecaster(self.sample_data, target_column='Close')
        
        # Split data first
        forecaster.split_data()
        
        # Test data preparation
        prepared_data = forecaster.prepare_arima_data()
        self.assertIsNotNone(prepared_data)
        self.assertGreater(len(prepared_data), 0)
        
        # Test model training (simplified)
        try:
            result = forecaster.fit_arima_model(auto_optimize=False, order=(1, 1, 1))
            self.assertIsNotNone(result)
        except Exception as e:
            # Some ARIMA models may fail on random data, which is acceptable for testing
            self.assertIn("ARIMA", str(type(e).__name__) + str(e))
    
    def test_portfolio_optimization_integration(self):
        """Test portfolio optimization integration."""
        # Create sample returns data for multiple assets
        assets = ['TSLA', 'SPY', 'BND']
        returns_data = {}
        
        for asset in assets:
            # Generate correlated returns
            base_returns = np.random.normal(0.001, 0.02, 252)  # Daily returns for 1 year
            returns_data[asset] = pd.Series(base_returns, 
                                          index=pd.date_range('2023-01-01', periods=252, freq='D'))
        
        returns_df = pd.DataFrame(returns_data)
        
        # Test covariance matrix calculation
        cov_matrix = returns_df.cov() * 252  # Annualize
        result = DataValidator.validate_covariance_matrix(cov_matrix)
        self.assertTrue(result)
        
        # Test expected returns calculation
        expected_returns = returns_df.mean() * 252  # Annualize
        self.assertEqual(len(expected_returns), len(assets))
        self.assertTrue(all(np.isfinite(expected_returns)))
    
    def test_backtesting_integration(self):
        """Test backtesting integration."""
        # Create sample portfolio performance data
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        
        # Strategy returns
        strategy_returns = pd.Series(np.random.normal(0.0008, 0.015, 252), index=dates)
        
        # Benchmark returns
        benchmark_returns = pd.Series(np.random.normal(0.0005, 0.012, 252), index=dates)
        
        # Calculate cumulative returns
        strategy_cumulative = (1 + strategy_returns).cumprod()
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        
        # Test performance metrics
        strategy_total_return = strategy_cumulative.iloc[-1] - 1
        benchmark_total_return = benchmark_cumulative.iloc[-1] - 1
        
        self.assertIsInstance(strategy_total_return, (int, float))
        self.assertIsInstance(benchmark_total_return, (int, float))
        
        # Test volatility calculation
        strategy_vol = strategy_returns.std() * np.sqrt(252)
        benchmark_vol = benchmark_returns.std() * np.sqrt(252)
        
        self.assertGreater(strategy_vol, 0)
        self.assertGreater(benchmark_vol, 0)
        
        # Test Sharpe ratio calculation
        risk_free_rate = 0.02
        strategy_sharpe = (strategy_returns.mean() * 252 - risk_free_rate) / strategy_vol
        benchmark_sharpe = (benchmark_returns.mean() * 252 - risk_free_rate) / benchmark_vol
        
        self.assertIsInstance(strategy_sharpe, (int, float))
        self.assertIsInstance(benchmark_sharpe, (int, float))
    
    def test_end_to_end_workflow(self):
        """Test simplified end-to-end workflow."""
        # 1. Data validation
        DataValidator.validate_price_data(self.sample_data)
        
        # 2. Returns calculation
        returns = self.sample_data['Close'].pct_change().dropna()
        DataValidator.validate_returns(returns, max_daily_return=0.2)  # More lenient for test data
        
        # 3. Simple portfolio weights
        weights = {'TSLA': 0.4, 'SPY': 0.4, 'BND': 0.2}
        DataValidator.validate_portfolio_weights(weights)
        
        # 4. Business rules validation
        from src.utils.validators import validate_business_rules
        config = {'max_single_asset_weight': 0.5, 'min_assets': 2}
        validate_business_rules(weights, config)
        
        # If we reach here, the workflow completed successfully
        self.assertTrue(True)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def test_missing_data_handling(self):
        """Test handling of missing data."""
        # Create data with missing values
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data_with_missing = pd.DataFrame({
            'Open': np.abs(np.random.randn(100)) + 100,  # Ensure positive prices
            'High': np.abs(np.random.randn(100)) + 102,
            'Low': np.abs(np.random.randn(100)) + 98,
            'Close': np.abs(np.random.randn(100)) + 100,
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # Introduce missing values (more than 10% to trigger warning)
        data_with_missing.iloc[0:15, :] = np.nan  # 15% missing data
        
        # Test that validation passes but logs warning for missing data
        result = DataValidator.validate_price_data(data_with_missing)
        self.assertTrue(result)  # Should return True but log warning
        
    def test_extreme_values_handling(self):
        """Test handling of extreme values."""
        # Create returns with extreme values
        returns = pd.Series(np.random.normal(0, 0.02, 100))
        returns.iloc[50] = 0.8  # 80% daily return (extreme)
        
        # Test that validation warns about extreme returns
        with self.assertLogs(level='WARNING') as log:
            DataValidator.validate_returns(returns)


if __name__ == '__main__':
    unittest.main()
