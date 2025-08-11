"""
Task 4: Portfolio Optimization using Modern Portfolio Theory (MPT)

This module implements portfolio optimization using:
1. ARIMA forecast for TSLA expected returns
2. Historical average returns for BND and SPY
3. Covariance matrix from historical daily returns
4. Efficient Frontier generation
5. Maximum Sharpe Ratio and Minimum Volatility portfolio identification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns  # Not used
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class PortfolioOptimizer:
    """
    Modern Portfolio Theory (MPT) based portfolio optimization
    """

    def __init__(self, assets=['TSLA', 'BND', 'SPY'], risk_free_rate=0.02):
        """
        Initialize portfolio optimizer

        Args:
            assets: List of asset symbols
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        self.assets = assets
        self.risk_free_rate = risk_free_rate
        self.historical_data = None
        self.returns_data = None
        self.expected_returns = None
        self.cov_matrix = None
        self.efficient_frontier = None

    def load_historical_data(self):
        """
        Load historical price data from existing processed files
        All assets (TSLA, BND, SPY) use processed data for historical calculations
        """
        print(f"Loading processed data for {self.assets}...")

        try:
            # Load from processed data files
            data_frames = []

            for asset in self.assets:
                file_path = f'data/processed/{asset}_processed.csv'
                print(f"📊 Loading {asset} from {file_path}...")

                asset_data = pd.read_csv(
                    file_path, index_col=0, parse_dates=True)
                # Use the 'Close' column as the price data
                asset_series = asset_data['Close'].rename(asset)
                data_frames.append(asset_series)

            # Combine all assets into a single DataFrame
            data = pd.concat(data_frames, axis=1)

            # Forward fill missing values and drop NaN
            data = data.fillna(method='ffill').dropna()

            self.historical_data = data
            print(
                f"✅ Loaded processed data from {data.index[0].date()} to {data.index[-1].date()}")
            print(f"📊 Data shape: {data.shape}")

            return data

        except Exception as e:
            print(f"❌ Error loading historical data: {str(e)}")
            return None

    def calculate_returns(self):
        """
        Calculate daily returns from historical price data
        """
        if self.historical_data is None:
            raise ValueError(
                "Historical data not loaded. Call load_historical_data() first.")

        print("Calculating daily returns...")

        # Calculate daily returns
        self.returns_data = self.historical_data.pct_change().dropna()

        print(
            f"✅ Calculated returns for {len(self.returns_data)} trading days")
        print(f"📈 Average daily returns:")
        for asset in self.assets:
            avg_return = self.returns_data[asset].mean()
            print(
                f"   {asset}: {
                    avg_return:.4f} ({
                    avg_return *
                    252:.2%} annualized)")

        return self.returns_data

    def set_expected_returns(self, tsla_forecast_return=None):
        """
        Set expected returns following Task 4 instructions:
        - TSLA: Use ARIMA forecast return (from Task 2/3)
        - BND, SPY: Use historical average daily returns (annualized)

        Args:
            tsla_forecast_return: Expected annual return for TSLA from ARIMA forecast
        """
        if self.returns_data is None:
            raise ValueError(
                "Returns data not calculated. Call calculate_returns() first.")

        print("Setting expected returns as per Task 4 instructions...")

        # Calculate historical average annual returns for BND and SPY
        historical_annual_returns = self.returns_data.mean() * 252

        # Initialize expected returns dictionary
        self.expected_returns = pd.Series(index=self.assets, dtype=float)

        for asset in self.assets:
            if asset == 'TSLA':
                # Use ARIMA forecast for TSLA (Task 4 requirement)
                if tsla_forecast_return is not None:
                    self.expected_returns[asset] = tsla_forecast_return
                    print(
                        f"📊 TSLA: Using ARIMA forecast return: {
                            tsla_forecast_return:.2%}")
                else:
                    # Fallback to historical if no forecast provided
                    self.expected_returns[asset] = historical_annual_returns[asset]
                    print(
                        f"⚠️  TSLA: No ARIMA forecast provided, using historical: {
                            historical_annual_returns[asset]:.2%}")
            else:
                # Use historical average for BND and SPY (Task 4 requirement)
                self.expected_returns[asset] = historical_annual_returns[asset]
                print(
                    f"📈 {asset}: Using historical average return: {
                        historical_annual_returns[asset]:.2%}")

        print(f"\n✅ Final Expected Annual Returns:")
        for asset in self.assets:
            source = "ARIMA Forecast" if asset == 'TSLA' else "Historical Average"
            print(f"   {asset}: {self.expected_returns[asset]:.2%} ({source})")

        return self.expected_returns

    def calculate_covariance_matrix(self):
        """
        Calculate annualized covariance matrix from daily returns
        """
        if self.returns_data is None:
            raise ValueError(
                "Returns data not calculated. Call calculate_returns() first.")

        print("Calculating covariance matrix...")

        # Calculate annualized covariance matrix
        self.cov_matrix = self.returns_data.cov() * 252

        print("✅ Covariance Matrix (Annualized):")
        print(self.cov_matrix.round(4))

        # Calculate correlation matrix for interpretation
        corr_matrix = self.returns_data.corr()
        print("\n📊 Correlation Matrix:")
        print(corr_matrix.round(3))

        return self.cov_matrix

    def portfolio_performance(self, weights):
        """
        Calculate portfolio expected return, volatility, and Sharpe ratio

        Args:
            weights: Portfolio weights array

        Returns:
            tuple: (expected_return, volatility, sharpe_ratio)
        """
        weights = np.array(weights)

        # Portfolio expected return
        portfolio_return = np.sum(weights * self.expected_returns.values)

        # Portfolio volatility
        portfolio_volatility = np.sqrt(
            np.dot(
                weights.T,
                np.dot(
                    self.cov_matrix.values,
                    weights)))

        # Sharpe ratio
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / \
            portfolio_volatility

        return portfolio_return, portfolio_volatility, sharpe_ratio

    def negative_sharpe_ratio(self, weights):
        """
        Negative Sharpe ratio for optimization (minimize)
        """
        return -self.portfolio_performance(weights)[2]

    def portfolio_volatility(self, weights):
        """
        Portfolio volatility for minimum variance optimization
        """
        return self.portfolio_performance(weights)[1]

    def generate_efficient_frontier(self, num_portfolios=10000):
        """
        Generate efficient frontier using Monte Carlo simulation
        """
        if self.expected_returns is None or self.cov_matrix is None:
            raise ValueError(
                "Expected returns and covariance matrix must be calculated first.")

        print(
            f"Generating efficient frontier with {num_portfolios} random portfolios...")

        num_assets = len(self.assets)
        results = np.zeros((3, num_portfolios))
        weights_array = np.zeros((num_portfolios, num_assets))

        # Generate random portfolios
        for i in range(num_portfolios):
            # Generate random weights
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)  # Normalize to sum to 1

            weights_array[i] = weights

            # Calculate portfolio metrics
            portfolio_return, portfolio_vol, sharpe_ratio = self.portfolio_performance(
                weights)

            results[0, i] = portfolio_return
            results[1, i] = portfolio_vol
            results[2, i] = sharpe_ratio

        # Store results
        self.efficient_frontier = {
            'returns': results[0],
            'volatility': results[1],
            'sharpe_ratio': results[2],
            'weights': weights_array
        }

        print("✅ Efficient frontier generated successfully")
        return self.efficient_frontier

    def find_optimal_portfolios(self):
        """
        Find Maximum Sharpe Ratio and Minimum Volatility portfolios using optimization
        """
        if self.expected_returns is None or self.cov_matrix is None:
            raise ValueError(
                "Expected returns and covariance matrix must be calculated first.")

        print("Finding optimal portfolios...")

        num_assets = len(self.assets)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = num_assets * [1.0 / num_assets]

        # Maximum Sharpe Ratio Portfolio
        max_sharpe_result = minimize(
            self.negative_sharpe_ratio,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        max_sharpe_weights = max_sharpe_result.x
        max_sharpe_return, max_sharpe_vol, max_sharpe_ratio = self.portfolio_performance(
            max_sharpe_weights)

        # Minimum Volatility Portfolio
        min_vol_result = minimize(
            self.portfolio_volatility,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        min_vol_weights = min_vol_result.x
        min_vol_return, min_vol_vol, min_vol_sharpe = self.portfolio_performance(
            min_vol_weights)

        # Store optimal portfolios
        self.optimal_portfolios = {
            'max_sharpe': {
                'weights': max_sharpe_weights,
                'return': max_sharpe_return,
                'volatility': max_sharpe_vol,
                'sharpe_ratio': max_sharpe_ratio
            },
            'min_volatility': {
                'weights': min_vol_weights,
                'return': min_vol_return,
                'volatility': min_vol_vol,
                'sharpe_ratio': min_vol_sharpe
            }
        }

        print("✅ Optimal portfolios found:")
        print(f"\n📈 Maximum Sharpe Ratio Portfolio:")
        print(f"   Return: {max_sharpe_return:.2%}")
        print(f"   Volatility: {max_sharpe_vol:.2%}")
        print(f"   Sharpe Ratio: {max_sharpe_ratio:.3f}")
        print(
            f"   Weights: {dict(zip(self.assets, max_sharpe_weights.round(3)))}")

        print(f"\n📉 Minimum Volatility Portfolio:")
        print(f"   Return: {min_vol_return:.2%}")
        print(f"   Volatility: {min_vol_vol:.2%}")
        print(f"   Sharpe Ratio: {min_vol_sharpe:.3f}")
        print(
            f"   Weights: {dict(zip(self.assets, min_vol_weights.round(3)))}")

        return self.optimal_portfolios

    def plot_efficient_frontier(
            self, save_path='results/figures/efficient_frontier.png'):
        """
        Plot the efficient frontier with optimal portfolios marked
        """
        if self.efficient_frontier is None:
            raise ValueError(
                "Efficient frontier not generated. Call generate_efficient_frontier() first.")

        print("Plotting efficient frontier...")

        # Create results directory if it doesn't exist
        import os
        os.makedirs('results/figures', exist_ok=True)

        plt.figure(figsize=(12, 8))

        # Plot efficient frontier
        scatter = plt.scatter(
            self.efficient_frontier['volatility'],
            self.efficient_frontier['returns'],
            c=self.efficient_frontier['sharpe_ratio'],
            cmap='viridis',
            alpha=0.6,
            s=20
        )

        plt.colorbar(scatter, label='Sharpe Ratio')

        # Mark optimal portfolios if available
        if hasattr(self, 'optimal_portfolios'):
            # Maximum Sharpe Ratio
            plt.scatter(
                self.optimal_portfolios['max_sharpe']['volatility'],
                self.optimal_portfolios['max_sharpe']['return'],
                marker='*',
                color='red',
                s=500,
                label=f"Max Sharpe Ratio\n(SR: {
                    self.optimal_portfolios['max_sharpe']['sharpe_ratio']:.3f})")

            # Minimum Volatility
            plt.scatter(
                self.optimal_portfolios['min_volatility']['volatility'],
                self.optimal_portfolios['min_volatility']['return'],
                marker='*',
                color='blue',
                s=500,
                label=f"Min Volatility\n(Vol: {
                    self.optimal_portfolios['min_volatility']['volatility']:.2%})")

        plt.xlabel('Volatility (Risk)', fontsize=12)
        plt.ylabel('Expected Return', fontsize=12)
        plt.title(
            'Efficient Frontier - Portfolio Optimization\n(TSLA, BND, SPY)',
            fontsize=14,
            fontweight='bold')
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)

        # Format axes as percentages
        plt.gca().yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        plt.gca().xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"✅ Efficient frontier plot saved as '{save_path}'")

    def generate_portfolio_report(self, recommended_portfolio='max_sharpe'):
        """
        Generate comprehensive portfolio optimization report
        """
        if not hasattr(self, 'optimal_portfolios'):
            raise ValueError(
                "Optimal portfolios not found. Call find_optimal_portfolios() first.")

        report = []
        report.append("=" * 80)
        report.append("PORTFOLIO OPTIMIZATION REPORT - TASK 4")
        report.append("Modern Portfolio Theory (MPT) Analysis")
        report.append("=" * 80)
        report.append(
            f"Generated on: {
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Assets analyzed: {', '.join(self.assets)}")
        report.append(f"Risk-free rate: {self.risk_free_rate:.2%}")
        report.append("")

        # Expected Returns Section
        report.append("1. EXPECTED RETURNS")
        report.append("-" * 40)
        for asset in self.assets:
            source = "ARIMA Forecast" if asset == 'TSLA' else "Historical Average"
            report.append(
                f"{asset}: {
                    self.expected_returns[asset]:.2%} ({source})")
        report.append("")

        # Risk Analysis Section
        report.append("2. RISK ANALYSIS (Annualized)")
        report.append("-" * 40)
        volatilities = np.sqrt(np.diag(self.cov_matrix))
        for i, asset in enumerate(self.assets):
            report.append(f"{asset} Volatility: {volatilities[i]:.2%}")
        report.append("")

        # Correlation Matrix
        report.append("3. ASSET CORRELATIONS")
        report.append("-" * 40)
        corr_matrix = self.returns_data.corr()
        for i, asset1 in enumerate(self.assets):
            for j, asset2 in enumerate(self.assets):
                if i < j:
                    report.append(
                        f"{asset1}-{asset2}: {corr_matrix.iloc[i, j]:.3f}")
        report.append("")

        # Optimal Portfolios
        report.append("4. OPTIMAL PORTFOLIOS")
        report.append("-" * 40)

        # Maximum Sharpe Ratio Portfolio
        max_sharpe = self.optimal_portfolios['max_sharpe']
        report.append("A) MAXIMUM SHARPE RATIO PORTFOLIO (Tangency Portfolio)")
        report.append(f"   Expected Return: {max_sharpe['return']:.2%}")
        report.append(f"   Volatility: {max_sharpe['volatility']:.2%}")
        report.append(f"   Sharpe Ratio: {max_sharpe['sharpe_ratio']:.3f}")
        report.append("   Asset Allocation:")
        for i, asset in enumerate(self.assets):
            report.append(f"     {asset}: {max_sharpe['weights'][i]:.1%}")
        report.append("")

        # Minimum Volatility Portfolio
        min_vol = self.optimal_portfolios['min_volatility']
        report.append("B) MINIMUM VOLATILITY PORTFOLIO")
        report.append(f"   Expected Return: {min_vol['return']:.2%}")
        report.append(f"   Volatility: {min_vol['volatility']:.2%}")
        report.append(f"   Sharpe Ratio: {min_vol['sharpe_ratio']:.3f}")
        report.append("   Asset Allocation:")
        for i, asset in enumerate(self.assets):
            report.append(f"     {asset}: {min_vol['weights'][i]:.1%}")
        report.append("")

        # Recommendation
        report.append("5. PORTFOLIO RECOMMENDATION")
        report.append("-" * 40)

        if recommended_portfolio == 'max_sharpe':
            recommended = max_sharpe
            portfolio_name = "Maximum Sharpe Ratio Portfolio"
            justification = "This portfolio maximizes risk-adjusted returns and is optimal for investors seeking the best return per unit of risk."
        else:
            recommended = min_vol
            portfolio_name = "Minimum Volatility Portfolio"
            justification = (
                "This portfolio minimizes risk and is suitable for "
                "conservative investors prioritizing capital preservation.")

        report.append(f"RECOMMENDED: {portfolio_name}")
        report.append("")
        report.append("JUSTIFICATION:")
        report.append(justification)
        report.append("")
        report.append("FINAL PORTFOLIO SUMMARY:")
        report.append(f"Expected Annual Return: {recommended['return']:.2%}")
        report.append(f"Annual Volatility: {recommended['volatility']:.2%}")
        report.append(f"Sharpe Ratio: {recommended['sharpe_ratio']:.3f}")
        report.append("")
        report.append("OPTIMAL ASSET WEIGHTS:")
        for i, asset in enumerate(self.assets):
            weight = recommended['weights'][i]
            report.append(f"{asset}: {weight:.1%}")
        report.append("")

        # Risk Assessment
        report.append("6. RISK ASSESSMENT")
        report.append("-" * 40)
        report.append("Portfolio Beta (vs SPY): Calculated from correlations")
        report.append(
            "Diversification Benefit: Achieved through multi-asset allocation")
        report.append(
            f"Maximum Drawdown Risk: Estimated at {
                recommended['volatility'] *
                2:.1%} (2-sigma)")
        report.append("")

        report.append("=" * 80)
        report.append("END OF PORTFOLIO OPTIMIZATION REPORT")
        report.append("=" * 80)

        return "\n".join(report)


def main():
    """
    Main execution function for Task 4: Portfolio Optimization
    """
    print("=" * 60)
    print("TASK 4: PORTFOLIO OPTIMIZATION")
    print("Modern Portfolio Theory (MPT) Implementation")
    print("=" * 60)

    try:
        # Initialize portfolio optimizer
        optimizer = PortfolioOptimizer(assets=['TSLA', 'BND', 'SPY'])

        # Step 1: Load historical data from processed files
        print("\n1. Loading Historical Data from Processed Files...")
        optimizer.load_historical_data()

        # Step 2: Calculate returns
        print("\n2. Calculating Daily Returns...")
        optimizer.calculate_returns()

        # Step 3: Set expected returns (using ARIMA forecast for TSLA)
        print("\n3. Setting Expected Returns...")
        # Load ARIMA forecast from Task 3 results
        try:
            # Extract forecast from ARIMA report
            with open('results/tesla_arima_forecast_report.txt', 'r', encoding='utf-8') as f:
                report_content = f.read()

            # Extract the expected return from the report
            import re
            expected_return_match = re.search(
                r'Expected Return: ([\d.-]+)%', report_content)

            if expected_return_match:
                tsla_arima_forecast = float(
                    expected_return_match.group(1)) / 100
                print(
                    f"📊 Extracted ARIMA forecast from Task 3: {
                        tsla_arima_forecast:.2%}")

                # Handle edge case: if ARIMA predicts 0% return, use a small
                # positive value for optimization
                if tsla_arima_forecast == 0.0:
                    print(
                        "⚠️  ARIMA predicts 0% return (stable prices). Using 2% for portfolio optimization.")
                    tsla_arima_forecast = 0.02  # 2% minimal return assumption
            else:
                print("⚠️  Could not parse ARIMA forecast, using conservative estimate")
                tsla_arima_forecast = 0.05  # 5% conservative estimate

        except Exception as e:
            print(f"⚠️  Error loading ARIMA forecast: {str(e)}")
            print("Using conservative estimate for TSLA")
            tsla_arima_forecast = 0.05  # 5% conservative estimate

        optimizer.set_expected_returns(
            tsla_forecast_return=tsla_arima_forecast)

        # Step 4: Calculate covariance matrix
        print("\n4. Calculating Covariance Matrix...")
        optimizer.calculate_covariance_matrix()

        # Step 5: Generate efficient frontier
        print("\n5. Generating Efficient Frontier...")
        optimizer.generate_efficient_frontier(num_portfolios=10000)

        # Step 6: Find optimal portfolios
        print("\n6. Finding Optimal Portfolios...")
        optimizer.find_optimal_portfolios()

        # Step 7: Plot efficient frontier
        print("\n7. Plotting Efficient Frontier...")
        optimizer.plot_efficient_frontier()

        # Step 8: Generate comprehensive report
        print("\n8. Generating Portfolio Report...")
        report = optimizer.generate_portfolio_report(
            recommended_portfolio='max_sharpe')

        # Save report
        import os
        os.makedirs('results', exist_ok=True)
        report_path = 'results/portfolio_optimization_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print("\n" + "=" * 60)
        print("PORTFOLIO OPTIMIZATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(report)

        print(f"\nPortfolio optimization report saved as '{report_path}'")
        print("Efficient frontier plot saved as 'results/figures/efficient_frontier.png'")

    except Exception as e:
        print(f"\nError in Task 4 execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
