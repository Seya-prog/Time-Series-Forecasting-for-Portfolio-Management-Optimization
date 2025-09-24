"""
Data validation utilities for portfolio management system.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("portfolio_system")


class DataValidator:
    """Comprehensive data validation for financial data."""

    @staticmethod
    def validate_price_data(
        data: pd.DataFrame, required_columns: Optional[List[str]] = None
    ) -> bool:
        """Validate price data structure and content."""
        if required_columns is None:
            required_columns = ["Open", "High", "Low", "Close", "Volume"]

        # Check if DataFrame is not empty
        if data.empty:
            raise ValueError("Price data is empty")

        # Check required columns
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Check for negative prices
        price_columns = ["Open", "High", "Low", "Close"]
        for col in price_columns:
            if col in data.columns and (data[col] <= 0).any():
                raise ValueError(f"Negative or zero prices found in {col}")

        # Check OHLC logic
        if all(col in data.columns for col in price_columns):
            invalid_ohlc = (
                (data["High"] < data["Low"])
                | (data["High"] < data["Open"])
                | (data["High"] < data["Close"])
                | (data["Low"] > data["Open"])
                | (data["Low"] > data["Close"])
            )
            if invalid_ohlc.any():
                logger.warning(
                    f"Found {invalid_ohlc.sum()} rows with invalid OHLC relationships"
                )

        # Check for excessive missing values
        missing_pct = data.isnull().sum() / len(data)
        high_missing = missing_pct[missing_pct > 0.1]
        if not high_missing.empty:
            logger.warning(
                f"Columns with >10% missing values: {high_missing.to_dict()}"
            )

        return True

    @staticmethod
    def validate_returns(returns: pd.Series, max_daily_return: float = 0.5) -> bool:
        """Validate return series for outliers and anomalies."""
        if returns.empty:
            raise ValueError("Returns series is empty")

        # Check for extreme returns
        extreme_returns = abs(returns) > max_daily_return
        if extreme_returns.any():
            logger.warning(
                f"Found {extreme_returns.sum()} extreme returns "
                f"(>{max_daily_return * 100}%)"
            )

        # Check for infinite or NaN values
        if not np.isfinite(returns).all():
            raise ValueError("Returns contain infinite or NaN values")

        return True

    @staticmethod
    def validate_portfolio_weights(
        weights: Dict[str, float], tolerance: float = 1e-6
    ) -> bool:
        """Validate portfolio weights sum to 1 and are non-negative."""
        if not weights:
            raise ValueError("Portfolio weights dictionary is empty")

        # Check non-negative weights
        negative_weights = {k: v for k, v in weights.items() if v < 0}
        if negative_weights:
            raise ValueError(f"Negative weights found: {negative_weights}")

        # Check weights sum to 1
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > tolerance:
            raise ValueError(
                f"Weights sum to {total_weight}, not 1.0 (tolerance: {tolerance})"
            )

        return True

    @staticmethod
    def validate_covariance_matrix(cov_matrix: pd.DataFrame) -> bool:
        """Validate covariance matrix properties."""
        if cov_matrix.empty:
            raise ValueError("Covariance matrix is empty")

        # Check if square
        if cov_matrix.shape[0] != cov_matrix.shape[1]:
            raise ValueError("Covariance matrix is not square")

        # Check if symmetric
        if not np.allclose(cov_matrix.values, cov_matrix.values.T):
            raise ValueError("Covariance matrix is not symmetric")

        # Check if positive semi-definite
        eigenvalues = np.linalg.eigvals(cov_matrix.values)
        if (eigenvalues < -1e-8).any():
            raise ValueError("Covariance matrix is not positive semi-definite")

        return True


class ModelValidator:
    """Validation utilities for model inputs and outputs."""

    @staticmethod
    def validate_forecast_inputs(data: pd.Series, min_observations: int = 100) -> bool:
        """Validate time series data for forecasting."""
        if len(data) < min_observations * 2:
            raise ValueError(
                f"Insufficient data: {len(data)} < {min_observations} observations"
            )

        # Check for stationarity issues (basic check)
        if data.std() == 0:
            raise ValueError("Time series has zero variance")

        # Check for excessive missing values
        missing_pct = data.isnull().sum() / len(data)
        if missing_pct > 0.05:
            raise ValueError(f"Too many missing values: {missing_pct:.2%}")

        return True

    @staticmethod
    def validate_forecast_outputs(
        predictions: np.ndarray, confidence_intervals: Optional[np.ndarray] = None
    ) -> bool:
        """Validate forecast outputs."""
        if len(predictions) == 0:
            raise ValueError("Predictions array is empty")

        if not np.isfinite(predictions).all():
            raise ValueError("Predictions contain infinite or NaN values")

        if confidence_intervals is not None:
            if confidence_intervals.shape[0] != len(predictions):
                raise ValueError(
                    "Confidence intervals length doesn't match predictions"
                )

            # Check CI bounds are reasonable
            lower_bounds = confidence_intervals[:, 0]
            upper_bounds = confidence_intervals[:, 1]

            if (lower_bounds > upper_bounds).any():
                raise ValueError("Lower confidence bounds exceed upper bounds")

        return True


def validate_business_rules(
    portfolio_weights: Dict[str, float], config: Dict[str, Any]
) -> bool:
    """Validate business rules and constraints."""

    # Maximum single asset concentration
    max_weight = max(portfolio_weights.values())
    if max_weight > config.get("max_single_asset_weight", 0.5):
        raise ValueError(f"Single asset weight {max_weight:.2%} exceeds limit")

    # Minimum diversification
    min_assets = config.get("min_assets", 2)
    active_assets = sum(1 for w in portfolio_weights.values() if w > 0.01)
    if active_assets < min_assets:
        raise ValueError(
            f"Portfolio has only {active_assets} active positions, minimum {min_assets}"
        )

    return True
