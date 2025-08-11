"""
Task 1: Preprocess and Explore the Data
Main execution script for comprehensive data analysis.
"""

from src.data.eda import FinancialEDA
from src.data.preprocessor import FinancialDataPreprocessor
from src.data.data_collector import FinancialDataCollector
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


warnings.filterwarnings('ignore')


def main():
    """Execute Task 1: Preprocess and Explore the Data."""

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

    # Create interactive dashboard
    print("\nLaunching interactive dashboard...")
    try:
        eda.plot_interactive_dashboard()
    except Exception as e:
        print(
            f"Note: Interactive dashboard requires Jupyter environment. Error: {e}")

    return processed_data, report


if __name__ == "__main__":
    processed_data, analysis_report = main()
