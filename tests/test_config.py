"""
Tests for configuration module.
"""

import unittest
import tempfile
import os
from pathlib import Path
from src.config import Config, DataConfig, ModelConfig, PortfolioConfig


class TestConfig(unittest.TestCase):
    """Test configuration classes."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
    
    def test_data_config_defaults(self):
        """Test DataConfig default values."""
        data_config = DataConfig()
        self.assertEqual(data_config.assets, ["TSLA", "BND", "SPY"])
        self.assertEqual(data_config.target_asset, "TSLA")
        self.assertEqual(data_config.benchmark_assets, ["SPY", "BND"])
    
    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        model_config = ModelConfig()
        self.assertEqual(model_config.arima_max_p, 5)
        self.assertEqual(model_config.lstm_lookback, 60)
        self.assertEqual(model_config.forecast_horizon_months, 12)
        self.assertEqual(model_config.confidence_level, 0.95)
    
    def test_portfolio_config_defaults(self):
        """Test PortfolioConfig default values."""
        portfolio_config = PortfolioConfig()
        self.assertEqual(portfolio_config.risk_free_rate, 0.02)
        self.assertEqual(portfolio_config.benchmark_weights["SPY"], 0.6)
        self.assertEqual(portfolio_config.benchmark_weights["BND"], 0.4)
    
    def test_config_from_env(self):
        """Test configuration from environment variables."""
        # Set environment variables
        os.environ['RISK_FREE_RATE'] = '0.03'
        os.environ['FORECAST_HORIZON'] = '6'
        os.environ['RANDOM_SEED'] = '123'
        
        config = Config.from_env()
        
        self.assertEqual(config.portfolio.risk_free_rate, 0.03)
        self.assertEqual(config.model.forecast_horizon_months, 6)
        self.assertEqual(config.system.random_seed, 123)
        
        # Clean up
        del os.environ['RISK_FREE_RATE']
        del os.environ['FORECAST_HORIZON']
        del os.environ['RANDOM_SEED']
    
    def test_directory_creation(self):
        """Test that necessary directories are created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config with temporary directory
            config = Config()
            config.data.raw_data_path = Path(temp_dir) / "raw"
            config.data.processed_data_path = Path(temp_dir) / "processed"
            config._create_directories()
            
            self.assertTrue(config.data.raw_data_path.exists())
            self.assertTrue(config.data.processed_data_path.exists())


if __name__ == '__main__':
    unittest.main()
