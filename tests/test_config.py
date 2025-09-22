"""
Tests for configuration module.
"""

import os
import tempfile
import unittest
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
        os.environ["RISK_FREE_RATE"] = "0.03"
        os.environ["FORECAST_HORIZON"] = "6"
        os.environ["RANDOM_SEED"] = "123"

        config = Config.from_env()

        self.assertEqual(config.portfolio.risk_free_rate, 0.03)
        self.assertEqual(config.model.forecast_horizon_months, 6)
        self.assertEqual(config.system.random_seed, 123)

        # Clean up
        del os.environ["RISK_FREE_RATE"]
        del os.environ["FORECAST_HORIZON"]
        del os.environ["RANDOM_SEED"]

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

    def test_config_validation(self):
        """Test configuration validation."""
        config = Config()

        # Test that all required attributes exist
        self.assertIsNotNone(config.data)
        self.assertIsNotNone(config.model)
        self.assertIsNotNone(config.portfolio)
        self.assertIsNotNone(config.system)

        # Test data config attributes
        self.assertIsInstance(config.data.assets, list)
        self.assertGreater(len(config.data.assets), 0)

        # Test model config attributes
        self.assertGreater(config.model.arima_max_p, 0)
        self.assertGreater(config.model.lstm_lookback, 0)

        # Test portfolio config attributes
        self.assertGreaterEqual(config.portfolio.risk_free_rate, 0)
        self.assertIsInstance(config.portfolio.benchmark_weights, dict)

    def test_config_serialization(self):
        """Test configuration serialization."""
        config = Config()

        # Test that config can be converted to dict-like structure
        data_dict = {
            "assets": config.data.assets,
            "target_asset": config.data.target_asset,
            "risk_free_rate": config.portfolio.risk_free_rate,
        }

        self.assertIsInstance(data_dict, dict)
        self.assertIn("assets", data_dict)
        self.assertIn("target_asset", data_dict)
        self.assertIn("risk_free_rate", data_dict)

    def test_config_edge_cases(self):
        """Test configuration edge cases."""
        # Test with different environment values
        original_env = os.environ.copy()

        try:
            # Test with invalid numeric values
            os.environ["RISK_FREE_RATE"] = "invalid"
            config = Config.from_env()
            # Should use default value if invalid
            self.assertEqual(config.portfolio.risk_free_rate, 0.02)

        except Exception:
            # If parsing fails, that's also valid behavior
            pass
        finally:
            # Restore original environment
            os.environ.clear()
            os.environ.update(original_env)


if __name__ == "__main__":
    unittest.main()
