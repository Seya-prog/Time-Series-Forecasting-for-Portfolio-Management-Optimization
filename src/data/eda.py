"""
Exploratory Data Analysis module for financial time series data.
Provides comprehensive analysis and visualization capabilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import warnings
import logging
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')
# Configure logging - suppress verbose output for user-friendly experience
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class FinancialEDA:
    """Comprehensive Exploratory Data Analysis for financial data."""

    def __init__(self, data: Dict[str, pd.DataFrame]):
        """
        Initialize EDA with processed data.

        Args:
            data: Dictionary with symbol as key and processed DataFrame as value
        """
        self.data = data
        self.symbols = list(data.keys())

    def plot_price_trends(self, save_path: str = None) -> None:
        """Plot closing price trends for all symbols."""
        fig, axes = plt.subplots(len(self.symbols), 1,
                                 figsize=(15, 5 * len(self.symbols)))
        if len(self.symbols) == 1:
            axes = [axes]

        for i, symbol in enumerate(self.symbols):
            data = self.data[symbol]

            # Determine the close price column
            close_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'

            axes[i].plot(
                data.index,
                data[close_col],
                linewidth=2,
                label=f'{symbol} Close Price')
            if 'MA_20' in data.columns:
                axes[i].plot(
                    data.index,
                    data['MA_20'],
                    alpha=0.7,
                    label='20-day MA')
            if 'MA_50' in data.columns:
                axes[i].plot(
                    data.index,
                    data['MA_50'],
                    alpha=0.7,
                    label='50-day MA')

            axes[i].set_title(
                f'{symbol} - Price Trends Over Time',
                fontsize=14,
                fontweight='bold')
            axes[i].set_xlabel('Date')
            axes[i].set_ylabel('Price ($)')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_returns_analysis(self, save_path: str = None) -> None:
        """Analyze and plot daily returns."""
        fig, axes = plt.subplots(len(self.symbols), 2,
                                 figsize=(20, 5 * len(self.symbols)))
        if len(self.symbols) == 1:
            axes = axes.reshape(1, -1)

        for i, symbol in enumerate(self.symbols):
            data = self.data[symbol]
            returns = data['Daily_Return'].dropna()

            # Time series plot
            axes[i, 0].plot(data.index[1:], returns, alpha=0.7, linewidth=1)
            axes[i, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[i, 0].set_title(f'{symbol} - Daily Returns Over Time')
            axes[i, 0].set_xlabel('Date')
            axes[i, 0].set_ylabel('Daily Return')
            axes[i, 0].grid(True, alpha=0.3)

            # Distribution plot
            axes[i, 1].hist(returns, bins=50, alpha=0.7,
                            density=True, edgecolor='black')

            # Overlay normal distribution
            mu, sigma = returns.mean(), returns.std()
            x = np.linspace(returns.min(), returns.max(), 100)
            axes[i, 1].plot(x, stats.norm.pdf(x, mu, sigma),
                            'r-', linewidth=2, label='Normal Distribution')

            axes[i, 1].set_title(f'{symbol} - Returns Distribution')
            axes[i, 1].set_xlabel('Daily Return')
            axes[i, 1].set_ylabel('Density')
            axes[i, 1].legend()
            axes[i, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_volatility_analysis(self, save_path: str = None) -> None:
        """Analyze and plot volatility patterns."""
        fig, axes = plt.subplots(len(self.symbols), 1,
                                 figsize=(15, 5 * len(self.symbols)))
        if len(self.symbols) == 1:
            axes = [axes]

        for i, symbol in enumerate(self.symbols):
            data = self.data[symbol]

            # Plot rolling volatility
            axes[i].plot(
                data.index,
                data['Volatility_5d'],
                alpha=0.7,
                label='5-day Volatility')
            axes[i].plot(
                data.index,
                data['Volatility_20d'],
                alpha=0.8,
                label='20-day Volatility')

            axes[i].set_title(f'{symbol} - Rolling Volatility Analysis')
            axes[i].set_xlabel('Date')
            axes[i].set_ylabel('Volatility')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def detect_outliers(self) -> Dict[str, Dict]:
        """Detect outliers in daily returns using statistical methods."""
        outlier_results = {}

        for symbol in self.symbols:
            data = self.data[symbol]
            returns = data['Daily_Return'].dropna()

            # Z-score method
            z_scores = np.abs(stats.zscore(returns))
            z_outliers = returns[z_scores > 3]

            # IQR method
            Q1 = returns.quantile(0.25)
            Q3 = returns.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = returns[(returns < lower_bound)
                                   | (returns > upper_bound)]

            outlier_results[symbol] = {
                'z_score_outliers': len(z_outliers),
                'iqr_outliers': len(iqr_outliers),
                'extreme_positive': returns.nlargest(5).tolist(),
                'extreme_negative': returns.nsmallest(5).tolist(),
                'outlier_dates_z': z_outliers.index.tolist(),
                'outlier_dates_iqr': iqr_outliers.index.tolist()
            }

        return outlier_results

    def plot_correlation_analysis(self, save_path: str = None) -> None:
        """Analyze correlations between assets."""
        # Combine returns data
        returns_data = {}
        for symbol in self.symbols:
            returns_data[symbol] = self.data[symbol]['Daily_Return']

        returns_df = pd.DataFrame(returns_data).dropna()

        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()

        # Plot correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title(
            'Asset Returns Correlation Matrix',
            fontsize=16,
            fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        return correlation_matrix

    def calculate_risk_metrics(self) -> pd.DataFrame:
        """Calculate comprehensive risk metrics."""
        risk_metrics = []

        for symbol in self.symbols:
            data = self.data[symbol]

            # Check if Daily_Return exists, if not skip this symbol
            if 'Daily_Return' not in data.columns:
                logger.warning(
                    f"Daily_Return not found for {symbol}, skipping risk metrics")
                continue

            returns = data['Daily_Return'].dropna()

            if len(returns) == 0:
                logger.warning(f"No valid returns data for {symbol}, skipping")
                continue

            # Basic metrics
            annual_return = returns.mean() * 252
            annual_volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0

            # Value at Risk (VaR) - 5% confidence level
            var_5 = returns.quantile(0.05)

            # Conditional Value at Risk (CVaR)
            cvar_5 = returns[returns <= var_5].mean()

            # Maximum drawdown
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()

            # Skewness and Kurtosis
            skewness = returns.skew()
            kurtosis = returns.kurtosis()

            risk_metrics.append({
                'Symbol': symbol,
                'Annual_Return': annual_return,
                'Annual_Volatility': annual_volatility,
                'Sharpe_Ratio': sharpe_ratio,
                'VaR_5%': var_5,
                'CVaR_5%': cvar_5,
                'Max_Drawdown': max_drawdown,
                'Skewness': skewness,
                'Kurtosis': kurtosis
            })

        return pd.DataFrame(risk_metrics)

    def generate_comprehensive_report(self) -> Dict:
        """Generate a comprehensive analysis report."""
        report = {
            'data_summary': {},
            'risk_metrics': self.calculate_risk_metrics(),
            'outliers': self.detect_outliers(),
            'correlations': None,
            'key_insights': []
        }

        # Data summary
        for symbol in self.symbols:
            data = self.data[symbol]
            close_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
            report['data_summary'][symbol] = {
                'total_observations': len(data),
                'date_range': (data.index.min(), data.index.max()),
                'price_range': (data[close_col].min(), data[close_col].max()),
                'average_daily_volume': data['Volume'].mean(),
                'missing_data_points': data.isnull().sum().sum()
            }

        # Calculate correlations
        returns_data = {}
        for symbol in self.symbols:
            returns_data[symbol] = self.data[symbol]['Daily_Return']
        returns_df = pd.DataFrame(returns_data).dropna()
        report['correlations'] = returns_df.corr()

        # Generate key insights
        risk_df = report['risk_metrics']

        # Highest return asset
        highest_return = risk_df.loc[risk_df['Annual_Return'].idxmax(
        ), 'Symbol']
        report['key_insights'].append(
            f"{highest_return} has the highest annual return")

        # Highest volatility asset
        highest_vol = risk_df.loc[risk_df['Annual_Volatility'].idxmax(
        ), 'Symbol']
        report['key_insights'].append(
            f"{highest_vol} has the highest volatility")

        # Best Sharpe ratio
        best_sharpe = risk_df.loc[risk_df['Sharpe_Ratio'].idxmax(), 'Symbol']
        report['key_insights'].append(
            f"{best_sharpe} has the best risk-adjusted returns (Sharpe ratio)")

        return report


def run_complete_analysis():
    """
    Execute complete Task 1: Preprocess and Explore the Data.
    This is the main runner function for comprehensive data analysis.
    """
    from datetime import datetime

    print("=" * 80)
    print("TASK 1: PREPROCESS AND EXPLORE THE DATA")
    print("=" * 80)
    print(
        f"Execution started at: {
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Step 1: Data Collection
    print("STEP 1: DATA COLLECTION")
    print("-" * 40)

    from .data_collector import FinancialDataCollector
    collector = FinancialDataCollector(
        start_date="2015-07-01",
        end_date="2025-07-31"
    )

    print("Fetching historical data for TSLA, BND, and SPY...")
    raw_data = collector.fetch_data()

    # Save raw data
    collector.save_data("data/raw")

    # Display data summary
    summary = collector.get_data_summary()
    print("\nData Collection Summary:")
    print(summary.to_string(index=False))

    # Display asset information
    asset_info = collector.get_asset_info()
    print("\nAsset Information:")
    for symbol, info in asset_info.items():
        print(f"\n{symbol} - {info['name']}:")
        print(f"  • Sector: {info['sector']}")
        print(f"  • Risk Profile: {info['risk_profile']}")
        print(f"  • Description: {info['description']}")

    print("\n" + "=" * 80)

    # Step 2: Data Preprocessing
    print("STEP 2: DATA PREPROCESSING AND CLEANING")
    print("-" * 40)

    from .preprocessor import FinancialDataPreprocessor
    preprocessor = FinancialDataPreprocessor()
    preprocessor.load_data(data_dict=raw_data)

    # Check data quality
    print("Checking data quality...")
    quality_report = preprocessor.check_data_quality()

    print("\nData Quality Report:")
    for symbol, report in quality_report.items():
        print(f"\n{symbol}:")
        print(f"  • Total records: {report['total_records']:,}")
        print(
            f"  • Date range: {
                report['date_range'][0]} to {
                report['date_range'][1]}")
        print(f"  • Missing values: {sum(report['missing_values'].values())}")
        print(f"  • Duplicate dates: {report['duplicate_dates']}")
        print(f"  • Zero volume days: {report['zero_volume']}")

        # Check for price inconsistencies
        consistency = report['price_consistency']
        total_issues = sum(consistency.values())
        if total_issues > 0:
            print(f"  • Price consistency issues: {total_issues}")
        else:
            print(f"  • Price data: Consistent ✓")

    # Clean and preprocess data
    print("\nCleaning and preprocessing data...")
    processed_data = preprocessor.clean_data()

    # Get basic statistics
    stats = preprocessor.get_basic_statistics()
    print("\nBasic Statistics:")
    print(stats.round(4).to_string(index=False))

    # Test stationarity
    print("\nStationarity Test Results (Augmented Dickey-Fuller):")
    stationarity = preprocessor.test_stationarity()
    for symbol, results in stationarity.items():
        print(f"\n{symbol}:")
        prices_stationary = "✓" if results['prices']['is_stationary'] else "✗"
        returns_stationary = "✓" if results['returns']['is_stationary'] else "✗"
        print(
            f"  • Prices: p-value = {results['prices']['p_value']:.6f} {prices_stationary}")
        print(
            f"  • Returns: p-value = {results['returns']['p_value']:.6f} {returns_stationary}")

    # Save processed data
    preprocessor.save_processed_data("data/processed")

    print("\n" + "=" * 80)

    # Step 3: Exploratory Data Analysis
    print("STEP 3: EXPLORATORY DATA ANALYSIS")
    print("-" * 40)

    eda = FinancialEDA(processed_data)

    # Create results directory
    import os
    os.makedirs("results/figures", exist_ok=True)

    # Generate visualizations
    print("Generating price trend analysis...")
    eda.plot_price_trends(save_path="results/figures/price_trends.png")

    print("Generating returns analysis...")
    eda.plot_returns_analysis(save_path="results/figures/returns_analysis.png")

    print("Generating volatility analysis...")
    eda.plot_volatility_analysis(
        save_path="results/figures/volatility_analysis.png")

    print("Generating correlation analysis...")
    correlation_matrix = eda.plot_correlation_analysis(
        save_path="results/figures/correlation_matrix.png")

    # Calculate comprehensive risk metrics
    print("\nCalculating risk metrics...")
    risk_metrics = eda.calculate_risk_metrics()
    print("\nRisk Metrics Summary:")
    print(risk_metrics.round(4).to_string(index=False))

    # Save risk metrics
    risk_metrics.to_csv("results/risk_metrics.csv", index=False)

    # Detect outliers
    print("\nDetecting outliers...")
    outliers = eda.detect_outliers()

    print("\nOutlier Analysis:")
    for symbol, outlier_info in outliers.items():
        print(f"\n{symbol}:")
        print(
            f"  • Z-score outliers (|z| > 3): {outlier_info['z_score_outliers']}")
        print(f"  • IQR outliers: {outlier_info['iqr_outliers']}")
        print(
            f"  • Extreme positive returns: {[f'{x:.4f}' for x in outlier_info['extreme_positive'][:3]]}")
        print(
            f"  • Extreme negative returns: {[f'{x:.4f}' for x in outlier_info['extreme_negative'][:3]]}")

    # Generate comprehensive report
    print("\nGenerating comprehensive analysis report...")
    report = eda.generate_comprehensive_report()

    print("\n" + "=" * 80)
    print("KEY INSIGHTS AND FINDINGS")
    print("=" * 80)

    # Display key insights
    for insight in report['key_insights']:
        print(f"• {insight}")

    # Additional insights based on the analysis
    print("\nAdditional Analysis:")

    # Volatility comparison
    vol_data = risk_metrics.set_index('Symbol')['Annual_Volatility']
    print(
        f"• Volatility ranking: {
            ' > '.join(
                vol_data.sort_values(
                    ascending=False).index.tolist())}")

    # Return comparison
    ret_data = risk_metrics.set_index('Symbol')['Annual_Return']
    print(
        f"• Return ranking: {
            ' > '.join(
                ret_data.sort_values(
                    ascending=False).index.tolist())}")

    # Correlation insights
    print(f"\nCorrelation Analysis:")
    if 'TSLA' in correlation_matrix.index and 'SPY' in correlation_matrix.index:
        tsla_spy_corr = correlation_matrix.loc['TSLA', 'SPY']
        print(f"• TSLA-SPY correlation: {tsla_spy_corr:.3f}")

    if 'BND' in correlation_matrix.index and 'SPY' in correlation_matrix.index:
        bnd_spy_corr = correlation_matrix.loc['BND', 'SPY']
        print(
            f"• BND-SPY correlation: {bnd_spy_corr:.3f} (diversification benefit)")

    # Risk-adjusted performance
    sharpe_data = risk_metrics.set_index('Symbol')['Sharpe_Ratio']
    best_sharpe = sharpe_data.idxmax()
    print(
        f"• Best risk-adjusted performance: {best_sharpe} (Sharpe: {sharpe_data[best_sharpe]:.3f})")

    print("\n" + "=" * 80)
    print("TASK 1 COMPLETION SUMMARY")
    print("=" * 80)

    print("✓ Data Collection: Successfully fetched 10+ years of data for TSLA, BND, SPY")
    print("✓ Data Cleaning: Handled missing values, validated price consistency")
    print("✓ Feature Engineering: Added returns, volatility, moving averages")
    print("✓ Statistical Analysis: Performed stationarity tests, calculated risk metrics")
    print("✓ Exploratory Analysis: Generated comprehensive visualizations")
    print("✓ Outlier Detection: Identified anomalous trading days")
    print("✓ Risk Assessment: Calculated VaR, CVaR, Sharpe ratios, max drawdown")
    print("✓ Correlation Analysis: Analyzed asset relationships for diversification")

    print(f"\nFiles created:")
    print(f"• Raw data: data/raw/")
    print(f"• Processed data: data/processed/")
    print(f"• Visualizations: results/figures/")
    print(f"• Risk metrics: results/risk_metrics.csv")

    print(
        f"\nTask 1 completed successfully at: {
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return processed_data, report


def main():
    """Main function - runs the complete analysis."""
    return run_complete_analysis()


if __name__ == "__main__":
    main()
