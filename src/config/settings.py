"""
Configuration settings for the portfolio management system.
Centralized configuration following best practices.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path
import os


@dataclass
class DataConfig:
    """Data-related configuration settings."""
    raw_data_path: Path = Path("data/raw")
    processed_data_path: Path = Path("data/processed")
    results_path: Path = Path("results")
    figures_path: Path = Path("results/figures")
    
    # Asset symbols
    assets: List[str] = None
    target_asset: str = "TSLA"
    benchmark_assets: List[str] = None
    
    def __post_init__(self):
        if self.assets is None:
            self.assets = ["TSLA", "BND", "SPY"]
        if self.benchmark_assets is None:
            self.benchmark_assets = ["SPY", "BND"]


@dataclass
class ModelConfig:
    """Model-related configuration settings."""
    # ARIMA parameters
    arima_max_p: int = 5
    arima_max_d: int = 2
    arima_max_q: int = 5
    seasonal: bool = True
    
    # LSTM parameters
    lstm_lookback: int = 60
    lstm_epochs: int = 100
    lstm_batch_size: int = 32
    lstm_units: int = 50
    
    # Forecasting
    forecast_horizon_months: int = 12
    confidence_level: float = 0.95
    
    # Model persistence
    model_save_path: Path = Path("models")


@dataclass
class PortfolioConfig:
    """Portfolio optimization configuration."""
    risk_free_rate: float = 0.02
    rebalancing_frequency: str = "monthly"  # daily, weekly, monthly, quarterly
    
    # Optimization constraints
    min_weight: float = 0.0
    max_weight: float = 1.0
    max_single_asset_weight: float = 0.5
    
    # Risk management
    max_volatility: float = 0.35
    max_drawdown: float = 0.25
    
    # Benchmark weights
    benchmark_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.benchmark_weights is None:
            self.benchmark_weights = {"SPY": 0.6, "BND": 0.4, "TSLA": 0.0}


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    backtest_period_months: int = 12
    initial_capital: float = 100000.0
    transaction_cost: float = 0.001  # 0.1% transaction cost
    
    # Performance metrics
    benchmark_name: str = "60/40 Portfolio"
    risk_free_rate: float = 0.02


@dataclass
class SystemConfig:
    """System-wide configuration."""
    # Logging
    log_level: str = "INFO"
    log_file: Path = Path("logs/portfolio_system.log")
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    # Parallel processing
    n_jobs: int = -1  # Use all available cores
    
    # API settings
    max_retries: int = 3
    timeout: int = 30


class Config:
    """Main configuration class that combines all config sections."""
    
    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.portfolio = PortfolioConfig()
        self.backtest = BacktestConfig()
        self.system = SystemConfig()
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.data.raw_data_path,
            self.data.processed_data_path,
            self.data.results_path,
            self.data.figures_path,
            self.model.model_save_path,
            self.system.log_file.parent
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables."""
        config = cls()
        
        # Override with environment variables if present
        if os.getenv('RISK_FREE_RATE'):
            config.portfolio.risk_free_rate = float(os.getenv('RISK_FREE_RATE'))
        
        if os.getenv('FORECAST_HORIZON'):
            config.model.forecast_horizon_months = int(os.getenv('FORECAST_HORIZON'))
        
        if os.getenv('RANDOM_SEED'):
            config.system.random_seed = int(os.getenv('RANDOM_SEED'))
        
        return config


# Global configuration instance
config = Config.from_env()
