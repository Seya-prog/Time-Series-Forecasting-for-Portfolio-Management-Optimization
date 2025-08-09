"""
Data collection module for financial time series data.
Fetches historical data for TSLA, BND, and SPY using YFinance.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
from typing import List, Dict, Optional

# Configure logging - suppress verbose output for user-friendly experience
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class FinancialDataCollector:
    """Collects and manages financial data from YFinance."""
    
    def __init__(self, start_date: str = "2015-07-01", end_date: str = "2025-07-31"):
        """
        Initialize the data collector.
        
        Args:
            start_date: Start date for data collection (YYYY-MM-DD)
            end_date: End date for data collection (YYYY-MM-DD)
        """
        self.start_date = start_date
        self.end_date = end_date
        self.symbols = ["TSLA", "BND", "SPY"]
        self.data = {}
        
    def fetch_data(self, symbols: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for specified symbols.
        
        Args:
            symbols: List of ticker symbols to fetch. If None, uses default symbols.
            
        Returns:
            Dictionary with symbol as key and DataFrame as value
        """
        if symbols is None:
            symbols = self.symbols
            
        logger.info(f"Fetching data for symbols: {symbols}")
        logger.info(f"Date range: {self.start_date} to {self.end_date}")
        
        for symbol in symbols:
            try:
                logger.info(f"Downloading data for {symbol}...")
                ticker = yf.Ticker(symbol)
                
                # Fetch historical data
                data = ticker.history(
                    start=self.start_date,
                    end=self.end_date,
                    interval="1d"
                )
                
                if data.empty:
                    logger.warning(f"No data found for {symbol}")
                    continue
                    
                # Reset index to make Date a column
                data.reset_index(inplace=True)
                
                # Add symbol column for identification
                data['Symbol'] = symbol
                
                # Store the data
                self.data[symbol] = data
                
                logger.info(f"Successfully fetched {len(data)} records for {symbol}")
                logger.info(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
                
        return self.data
    
    def get_asset_info(self) -> Dict[str, Dict]:
        """
        Get detailed information about each asset.
        
        Returns:
            Dictionary with asset information
        """
        asset_info = {
            "TSLA": {
                "name": "Tesla Inc.",
                "sector": "Consumer Discretionary",
                "industry": "Automobile Manufacturing",
                "description": "High-growth, high-risk stock in the consumer discretionary sector",
                "risk_profile": "High Risk, High Return Potential"
            },
            "BND": {
                "name": "Vanguard Total Bond Market ETF",
                "sector": "Fixed Income",
                "industry": "Bond ETF",
                "description": "Bond ETF tracking U.S. investment-grade bonds, providing stability and income",
                "risk_profile": "Low Risk, Stable Returns"
            },
            "SPY": {
                "name": "SPDR S&P 500 ETF Trust",
                "sector": "Equity",
                "industry": "Broad Market ETF",
                "description": "ETF tracking the S&P 500 Index, offering broad U.S. market exposure",
                "risk_profile": "Moderate Risk, Market Returns"
            }
        }
        return asset_info
    
    def save_data(self, output_dir: str = "data/raw") -> None:
        """
        Save collected data to CSV files.
        
        Args:
            output_dir: Directory to save the data files
        """
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        for symbol, data in self.data.items():
            filename = f"{symbol}_historical_data.csv"
            filepath = os.path.join(output_dir, filename)
            
            # Save to CSV
            data.to_csv(filepath, index=False)
            logger.info(f"Saved {symbol} data to {filepath}")
            
        # Save combined data to processed folder (since it's derived/processed data)
        if self.data:
            combined_data = pd.concat(self.data.values(), ignore_index=True)
            # Create processed directory if it doesn't exist
            processed_dir = output_dir.replace("raw", "processed")
            os.makedirs(processed_dir, exist_ok=True)
            combined_filepath = os.path.join(processed_dir, "combined_historical_data.csv")
            combined_data.to_csv(combined_filepath, index=False)
            logger.info(f"Saved combined data to {combined_filepath}")
    
    def get_data_summary(self) -> pd.DataFrame:
        """
        Get summary statistics for all collected data.
        
        Returns:
            DataFrame with summary statistics
        """
        summaries = []
        
        for symbol, data in self.data.items():
            summary = {
                'Symbol': symbol,
                'Records': len(data),
                'Start_Date': data['Date'].min(),
                'End_Date': data['Date'].max(),
                'Avg_Close': data['Close'].mean(),
                'Min_Close': data['Close'].min(),
                'Max_Close': data['Close'].max(),
                'Avg_Volume': data['Volume'].mean(),
                'Missing_Values': data.isnull().sum().sum()
            }
            summaries.append(summary)
            
        return pd.DataFrame(summaries)


def main():
    """Main function to collect data."""
    collector = FinancialDataCollector()
    
    # Fetch data
    data = collector.fetch_data()
    
    # Save data
    collector.save_data()
    
    # Print summary
    summary = collector.get_data_summary()
    print("\nData Collection Summary:")
    print(summary.to_string(index=False))
    
    # Print asset information
    asset_info = collector.get_asset_info()
    print("\nAsset Information:")
    for symbol, info in asset_info.items():
        print(f"\n{symbol} - {info['name']}:")
        print(f"  Sector: {info['sector']}")
        print(f"  Risk Profile: {info['risk_profile']}")
        print(f"  Description: {info['description']}")


if __name__ == "__main__":
    main()
