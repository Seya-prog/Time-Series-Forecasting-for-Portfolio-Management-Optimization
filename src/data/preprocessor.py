"""
Data preprocessing and cleaning module for financial time series data.
Handles missing values, data validation, and feature engineering.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import warnings
import logging
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')
# Configure logging - suppress verbose output for user-friendly experience
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class FinancialDataPreprocessor:
    """Preprocesses and cleans financial time series data."""
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.data = {}
        self.processed_data = {}
        
    def load_data(self, data_path: str = None, data_dict: Dict = None) -> Dict[str, pd.DataFrame]:
        """
        Load data from file or dictionary.
        
        Args:
            data_path: Path to CSV file with combined data
            data_dict: Dictionary with symbol as key and DataFrame as value
            
        Returns:
            Dictionary with loaded data
        """
        if data_dict:
            self.data = data_dict.copy()
        elif data_path:
            combined_data = pd.read_csv(data_path)
            # Split by symbol
            for symbol in combined_data['Symbol'].unique():
                self.data[symbol] = combined_data[combined_data['Symbol'] == symbol].copy()
        
        logger.info(f"Loaded data for symbols: {list(self.data.keys())}")
        return self.data
    
    def check_data_quality(self) -> Dict[str, Dict]:
        """
        Check data quality and identify issues.
        
        Returns:
            Dictionary with data quality metrics for each symbol
        """
        quality_report = {}
        
        for symbol, data in self.data.items():
            logger.info(f"Checking data quality for {symbol}...")
            
            # Convert Date column to datetime if it's not already
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
            
            report = {
                'total_records': len(data),
                'date_range': (data['Date'].min(), data['Date'].max()),
                'missing_values': data.isnull().sum().to_dict(),
                'duplicate_dates': data['Date'].duplicated().sum(),
                'data_types': data.dtypes.to_dict(),
                'negative_prices': (data[['Open', 'High', 'Low', 'Close']] < 0).sum().to_dict(),
                'zero_volume': (data['Volume'] == 0).sum(),
                'price_consistency': self._check_price_consistency(data)
            }
            
            quality_report[symbol] = report
            
        return quality_report
    
    def _check_price_consistency(self, data: pd.DataFrame) -> Dict:
        """Check if High >= Low >= 0 and other price consistency rules."""
        consistency_issues = {
            'high_less_than_low': (data['High'] < data['Low']).sum(),
            'high_less_than_open': (data['High'] < data['Open']).sum(),
            'high_less_than_close': (data['High'] < data['Close']).sum(),
            'low_greater_than_open': (data['Low'] > data['Open']).sum(),
            'low_greater_than_close': (data['Low'] > data['Close']).sum()
        }
        return consistency_issues
    
    def clean_data(self) -> Dict[str, pd.DataFrame]:
        """
        Clean and preprocess the data.
        
        Returns:
            Dictionary with cleaned data
        """
        for symbol, data in self.data.items():
            logger.info(f"Cleaning data for {symbol}...")
            
            # Make a copy for processing
            cleaned_data = data.copy()
            
            # Convert Date to datetime and set as index
            cleaned_data['Date'] = pd.to_datetime(cleaned_data['Date'])
            cleaned_data.set_index('Date', inplace=True)
            
            # Sort by date
            cleaned_data.sort_index(inplace=True)
            
            # Remove duplicates
            cleaned_data = cleaned_data[~cleaned_data.index.duplicated(keep='first')]
            
            # Handle missing values
            cleaned_data = self._handle_missing_values(cleaned_data, symbol)
            
            # Validate and fix price inconsistencies
            cleaned_data = self._fix_price_inconsistencies(cleaned_data)
            
            # Add derived features
            cleaned_data = self._add_features(cleaned_data)
            
            self.processed_data[symbol] = cleaned_data
            
        logger.info("Data cleaning completed for all symbols")
        return self.processed_data
    
    def _handle_missing_values(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        missing_counts = data.isnull().sum()
        
        if missing_counts.sum() > 0:
            logger.warning(f"Found missing values in {symbol}: {missing_counts[missing_counts > 0].to_dict()}")
            
            # Forward fill for price data (carry last observation forward)
            price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
            for col in price_columns:
                if col in data.columns:
                    data[col].fillna(method='ffill', inplace=True)
            
            # For volume, use median of surrounding values
            if 'Volume' in data.columns:
                data['Volume'].fillna(data['Volume'].rolling(window=5, center=True).median(), inplace=True)
                data['Volume'].fillna(data['Volume'].median(), inplace=True)
        
        return data
    
    def _fix_price_inconsistencies(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fix basic price inconsistencies."""
        # Ensure High is the maximum of Open, High, Low, Close
        data['High'] = data[['Open', 'High', 'Low', 'Close']].max(axis=1)
        
        # Ensure Low is the minimum of Open, High, Low, Close
        data['Low'] = data[['Open', 'High', 'Low', 'Close']].min(axis=1)
        
        return data
    
    def _add_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add derived features for analysis."""
        # Check for column names and use appropriate one
        adj_close_col = None
        if 'Adj Close' in data.columns:
            adj_close_col = 'Adj Close'
        elif 'Adj_Close' in data.columns:
            adj_close_col = 'Adj_Close'
        elif 'Close' in data.columns:
            adj_close_col = 'Close'
        else:
            logger.warning("No suitable close price column found")
            return data
            
        # Daily returns
        data['Daily_Return'] = data[adj_close_col].pct_change()
        
        # Log returns
        data['Log_Return'] = np.log(data[adj_close_col] / data[adj_close_col].shift(1))
        
        # Volatility (rolling standard deviation)
        data['Volatility_5d'] = data['Daily_Return'].rolling(window=5).std()
        data['Volatility_20d'] = data['Daily_Return'].rolling(window=20).std()
        
        # Moving averages
        data['MA_5'] = data[adj_close_col].rolling(window=5).mean()
        data['MA_20'] = data[adj_close_col].rolling(window=20).mean()
        data['MA_50'] = data[adj_close_col].rolling(window=50).mean()
        
        # Price range
        data['Price_Range'] = data['High'] - data['Low']
        data['Price_Range_Pct'] = (data['High'] - data['Low']) / data['Close']
        
        # Store the adjusted close column name for later use
        data['Adj_Close_Col'] = adj_close_col
        
        return data
    
    def get_basic_statistics(self) -> pd.DataFrame:
        """Get basic statistics for all processed data."""
        stats_list = []
        
        for symbol, data in self.processed_data.items():
            # Determine the close price column
            close_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
            
            stats_dict = {
                'Symbol': symbol,
                'Count': len(data),
                'Mean_Close': data[close_col].mean(),
                'Std_Close': data[close_col].std(),
                'Min_Close': data[close_col].min(),
                'Max_Close': data[close_col].max(),
                'Mean_Volume': data['Volume'].mean(),
                'Mean_Daily_Return': data['Daily_Return'].mean(),
                'Std_Daily_Return': data['Daily_Return'].std(),
                'Skewness': data['Daily_Return'].skew(),
                'Kurtosis': data['Daily_Return'].kurtosis()
            }
            stats_list.append(stats_dict)
        
        return pd.DataFrame(stats_list)
    
    def test_stationarity(self) -> Dict[str, Dict]:
        """Perform Augmented Dickey-Fuller test for stationarity."""
        stationarity_results = {}
        
        for symbol, data in self.processed_data.items():
            # Determine the close price column
            close_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
            
            # Test on closing prices
            adf_prices = adfuller(data[close_col].dropna())
            
            # Test on returns
            adf_returns = adfuller(data['Daily_Return'].dropna())
            
            stationarity_results[symbol] = {
                'prices': {
                    'adf_statistic': adf_prices[0],
                    'p_value': adf_prices[1],
                    'critical_values': adf_prices[4],
                    'is_stationary': adf_prices[1] < 0.05
                },
                'returns': {
                    'adf_statistic': adf_returns[0],
                    'p_value': adf_returns[1],
                    'critical_values': adf_returns[4],
                    'is_stationary': adf_returns[1] < 0.05
                }
            }
        
        return stationarity_results
    
    def save_processed_data(self, output_dir: str = "data/processed") -> None:
        """Save processed data to files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for symbol, data in self.processed_data.items():
            filename = f"{symbol}_processed.csv"
            filepath = os.path.join(output_dir, filename)
            data.to_csv(filepath)
            logger.info(f"Saved processed {symbol} data to {filepath}")


def main():
    """Main function for data preprocessing."""
    from .data_collector import FinancialDataCollector
    
    # Collect data first
    collector = FinancialDataCollector()
    raw_data = collector.fetch_data()
    
    # Preprocess data
    preprocessor = FinancialDataPreprocessor()
    preprocessor.load_data(data_dict=raw_data)
    
    # Check data quality
    quality_report = preprocessor.check_data_quality()
    print("Data Quality Report:")
    for symbol, report in quality_report.items():
        print(f"\n{symbol}:")
        print(f"  Total records: {report['total_records']}")
        print(f"  Missing values: {sum(report['missing_values'].values())}")
        print(f"  Duplicate dates: {report['duplicate_dates']}")
    
    # Clean data
    processed_data = preprocessor.clean_data()
    
    # Get statistics
    stats = preprocessor.get_basic_statistics()
    print("\nBasic Statistics:")
    print(stats.to_string(index=False))
    
    # Test stationarity
    stationarity = preprocessor.test_stationarity()
    print("\nStationarity Test Results:")
    for symbol, results in stationarity.items():
        print(f"\n{symbol}:")
        print(f"  Prices - ADF p-value: {results['prices']['p_value']:.4f}, Stationary: {results['prices']['is_stationary']}")
        print(f"  Returns - ADF p-value: {results['returns']['p_value']:.4f}, Stationary: {results['returns']['is_stationary']}")
    
    # Save processed data
    preprocessor.save_processed_data()


if __name__ == "__main__":
    main()
