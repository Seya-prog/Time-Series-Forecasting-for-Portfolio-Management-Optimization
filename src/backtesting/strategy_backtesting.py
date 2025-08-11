"""
Task 5: Strategy Backtesting

This module implements strategy backtesting to validate portfolio performance by:
1. Defining a backtesting period (last year of dataset)
2. Creating a benchmark portfolio (60% SPY / 40% BND)
3. Simulating the optimized portfolio strategy from Task 4
4. Analyzing performance and comparing against benchmark
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional
import os

warnings.filterwarnings('ignore')


class StrategyBacktester:
    """
    Backtesting framework for portfolio strategy validation
    """

    def __init__(self, assets=['TSLA', 'BND', 'SPY']):
        """
        Initialize strategy backtester

        Args:
            assets: List of asset symbols to backtest
        """
        self.assets = assets
        self.historical_data = None
        self.backtest_data = None
        self.benchmark_weights = {
            'SPY': 0.6,
            'BND': 0.4,
            'TSLA': 0.0}  # 60/40 benchmark
        self.strategy_weights = None
        self.backtest_results = {}

    def load_historical_data(self):
        """
        Load historical price data from processed files for backtesting
        """
        print("Loading historical data for backtesting...")

        try:
            data_frames = []

            for asset in self.assets:
                file_path = f'data/processed/{asset}_processed.csv'
                print(f"📊 Loading {asset} from {file_path}...")

                asset_data = pd.read_csv(
                    file_path, index_col=0, parse_dates=True)
                asset_series = asset_data['Close'].rename(asset)
                data_frames.append(asset_series)

            # Combine all assets into a single DataFrame
            data = pd.concat(data_frames, axis=1)
            data = data.fillna(method='ffill').dropna()

            self.historical_data = data
            print(
                f"✅ Loaded data from {data.index[0].date()} to {data.index[-1].date()}")
            print(f"📊 Data shape: {data.shape}")

            return data

        except Exception as e:
            print(f"❌ Error loading historical data: {str(e)}")
            return None

    def define_backtesting_period(self, start_date=None, end_date=None):
        """
        Define backtesting period - last year of dataset as instructed

        Args:
            start_date: Start date for backtesting (default: last year)
            end_date: End date for backtesting (default: latest data)
        """
        if self.historical_data is None:
            raise ValueError("Load historical data first")

        # Use last year of dataset as instructed
        if end_date is None:
            end_date = self.historical_data.index.max()
        if start_date is None:
            start_date = end_date - timedelta(days=365)

        # Filter data for backtesting period
        self.backtest_data = self.historical_data[
            (self.historical_data.index >= start_date) &
            (self.historical_data.index <= end_date)
        ]

        print(f"\n📅 BACKTESTING PERIOD DEFINED")
        print(f"Start Date: {start_date.date()}")
        print(f"End Date: {end_date.date()}")
        print(f"Trading Days: {len(self.backtest_data)}")

        return self.backtest_data

    def load_strategy_weights(self):
        """
        Load optimal portfolio weights from Task 4 results
        """
        print("\n📊 Loading strategy weights from Task 4...")

        try:
            # Load portfolio optimization report
            with open('results/portfolio_optimization_report.txt', 'r', encoding='utf-8') as f:
                report_content = f.read()

            # Extract Maximum Sharpe Ratio portfolio weights
            import re

            # Find the weights section for Maximum Sharpe Ratio portfolio
            weights_section = re.search(
                r'A\) MAXIMUM SHARPE RATIO PORTFOLIO.*?Asset Allocation:(.*?)(?=\n\n|\nB\))',
                report_content,
                re.DOTALL)

            if weights_section:
                weights_text = weights_section.group(1)

                # Extract individual weights
                weights = {}
                for asset in self.assets:
                    weight_match = re.search(
                        rf'{asset}:\s*([\d.]+)%', weights_text)
                    if weight_match:
                        weights[asset] = float(weight_match.group(1)) / 100

                self.strategy_weights = weights
                print("✅ Strategy weights loaded:")
                for asset, weight in weights.items():
                    print(f"   {asset}: {weight:.1%}")

            else:
                raise ValueError("Could not extract weights from report")

        except Exception as e:
            print(f"⚠️ Error loading strategy weights: {str(e)}")
            print("Using equal weights as fallback")
            self.strategy_weights = {asset: 1 /
                                     len(self.assets) for asset in self.assets}

        return self.strategy_weights

    def simulate_portfolio_performance(
            self, weights: Dict, portfolio_name: str) -> Dict:
        """
        Simulate portfolio performance over backtesting period

        Args:
            weights: Portfolio weights dictionary
            portfolio_name: Name for the portfolio

        Returns:
            Dictionary with performance metrics
        """
        if self.backtest_data is None:
            raise ValueError("Define backtesting period first")

        print(f"\n📈 Simulating {portfolio_name} performance...")

        # Calculate daily returns
        returns = self.backtest_data.pct_change().dropna()

        # Calculate portfolio daily returns
        portfolio_returns = pd.Series(0.0, index=returns.index)
        for asset in self.assets:
            if asset in weights and weights[asset] > 0:
                portfolio_returns += returns[asset] * weights[asset]

        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()

        # Calculate performance metrics
        total_return = cumulative_returns.iloc[-1] - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - 0.02) / \
            volatility  # Assuming 2% risk-free rate

        # Calculate maximum drawdown
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        performance = {
            'portfolio_name': portfolio_name,
            'weights': weights,
            'daily_returns': portfolio_returns,
            'cumulative_returns': cumulative_returns,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_value': cumulative_returns.iloc[-1]
        }

        print(f"✅ {portfolio_name} Performance:")
        print(f"   Total Return: {total_return:.2%}")
        print(f"   Annualized Return: {annualized_return:.2%}")
        print(f"   Volatility: {volatility:.2%}")
        print(f"   Sharpe Ratio: {sharpe_ratio:.3f}")
        print(f"   Max Drawdown: {max_drawdown:.2%}")

        return performance

    def run_backtest(self):
        """
        Run complete backtesting analysis as per Task 5 instructions
        """
        print("=" * 70)
        print("TASK 5: STRATEGY BACKTESTING")
        print("Validating Portfolio Strategy Performance")
        print("=" * 70)

        # Step 1: Define backtesting period (last year)
        print("\n1. DEFINING BACKTESTING PERIOD")
        print("-" * 40)
        self.define_backtesting_period()

        # Step 2: Define benchmark (60% SPY / 40% BND)
        print("\n2. DEFINING BENCHMARK PORTFOLIO")
        print("-" * 40)
        print("Benchmark: 60% SPY / 40% BND (Static Portfolio)")

        # Step 3: Load strategy weights from Task 4
        print("\n3. LOADING STRATEGY WEIGHTS")
        print("-" * 40)
        self.load_strategy_weights()

        # Step 4: Simulate benchmark performance
        print("\n4. SIMULATING BENCHMARK PERFORMANCE")
        print("-" * 40)
        benchmark_performance = self.simulate_portfolio_performance(
            self.benchmark_weights,
            "Benchmark (60% SPY / 40% BND)"
        )

        # Step 5: Simulate strategy performance
        print("\n5. SIMULATING STRATEGY PERFORMANCE")
        print("-" * 40)
        strategy_performance = self.simulate_portfolio_performance(
            self.strategy_weights,
            "Optimized Strategy (Task 4)"
        )

        # Store results
        self.backtest_results = {
            'benchmark': benchmark_performance,
            'strategy': strategy_performance
        }

        return self.backtest_results

    def plot_performance_comparison(
            self, save_path='results/figures/backtest_performance.png'):
        """
        Plot cumulative returns comparison between strategy and benchmark
        """
        if not self.backtest_results:
            raise ValueError("Run backtest first")

        print("\n📊 Plotting performance comparison...")

        # Create results directory
        os.makedirs('results/figures', exist_ok=True)

        plt.figure(figsize=(14, 8))

        # Plot cumulative returns
        benchmark_cum_returns = self.backtest_results['benchmark']['cumulative_returns']
        strategy_cum_returns = self.backtest_results['strategy']['cumulative_returns']

        plt.plot(
            benchmark_cum_returns.index,
            benchmark_cum_returns,
            label='Benchmark (60% SPY / 40% BND)',
            linewidth=2,
            color='blue')
        plt.plot(strategy_cum_returns.index, strategy_cum_returns,
                 label='Optimized Strategy (Task 4)', linewidth=2, color='red')

        plt.title('Portfolio Performance Comparison - Backtesting Results',
                  fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Returns', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        # Format y-axis as percentage
        plt.gca().yaxis.set_major_formatter(
            plt.FuncFormatter(
                lambda y,
                _: '{:.0%}'.format(
                    y - 1)))

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"✅ Performance comparison plot saved as '{save_path}'")

    def generate_backtest_report(self):
        """
        Generate comprehensive backtesting report
        """
        if not self.backtest_results:
            raise ValueError("Run backtest first")

        benchmark = self.backtest_results['benchmark']
        strategy = self.backtest_results['strategy']

        report = []
        report.append("=" * 80)
        report.append("STRATEGY BACKTESTING REPORT - TASK 5")
        report.append("Portfolio Performance Validation")
        report.append("=" * 80)
        report.append(
            f"Generated on: {
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(
            f"Backtesting Period: {self.backtest_data.index[0].date()} to {self.backtest_data.index[-1].date()}")
        report.append(f"Trading Days: {len(self.backtest_data)}")
        report.append("")

        # Portfolio Definitions
        report.append("1. PORTFOLIO DEFINITIONS")
        report.append("-" * 40)
        report.append("BENCHMARK PORTFOLIO (Static):")
        for asset, weight in self.benchmark_weights.items():
            if weight > 0:
                report.append(f"   {asset}: {weight:.1%}")
        report.append("")
        report.append("STRATEGY PORTFOLIO (Task 4 Optimized):")
        for asset, weight in self.strategy_weights.items():
            if weight > 0:
                report.append(f"   {asset}: {weight:.1%}")
        report.append("")

        # Performance Comparison
        report.append("2. PERFORMANCE COMPARISON")
        report.append("-" * 40)

        report.append("A) BENCHMARK PERFORMANCE:")
        report.append(f"   Total Return: {benchmark['total_return']:.2%}")
        report.append(
            f"   Annualized Return: {
                benchmark['annualized_return']:.2%}")
        report.append(f"   Volatility: {benchmark['volatility']:.2%}")
        report.append(f"   Sharpe Ratio: {benchmark['sharpe_ratio']:.3f}")
        report.append(f"   Maximum Drawdown: {benchmark['max_drawdown']:.2%}")
        report.append("")

        report.append("B) STRATEGY PERFORMANCE:")
        report.append(f"   Total Return: {strategy['total_return']:.2%}")
        report.append(
            f"   Annualized Return: {
                strategy['annualized_return']:.2%}")
        report.append(f"   Volatility: {strategy['volatility']:.2%}")
        report.append(f"   Sharpe Ratio: {strategy['sharpe_ratio']:.3f}")
        report.append(f"   Maximum Drawdown: {strategy['max_drawdown']:.2%}")
        report.append("")

        # Performance Difference
        return_diff = strategy['total_return'] - benchmark['total_return']
        sharpe_diff = strategy['sharpe_ratio'] - benchmark['sharpe_ratio']
        vol_diff = strategy['volatility'] - benchmark['volatility']

        report.append("C) PERFORMANCE DIFFERENCE (Strategy vs Benchmark):")
        report.append(f"   Return Difference: {return_diff:.2%}")
        report.append(f"   Sharpe Ratio Difference: {sharpe_diff:.3f}")
        report.append(f"   Volatility Difference: {vol_diff:.2%}")
        report.append("")

        # Analysis and Conclusions
        report.append("3. BACKTESTING ANALYSIS")
        report.append("-" * 40)

        # Strategy outperformance analysis
        if return_diff > 0.02:  # 2% outperformance
            performance_verdict = "🚀 STRATEGY SIGNIFICANTLY OUTPERFORMED"
        elif return_diff > 0:
            performance_verdict = "✅ STRATEGY OUTPERFORMED"
        elif return_diff > -0.02:
            performance_verdict = "➡️ STRATEGY PERFORMED SIMILARLY"
        else:
            performance_verdict = "📉 STRATEGY UNDERPERFORMED"

        report.append(f"PERFORMANCE VERDICT: {performance_verdict}")
        report.append("")

        # Risk-adjusted performance
        if sharpe_diff > 0.1:
            risk_adjusted_verdict = "🎯 SUPERIOR RISK-ADJUSTED RETURNS"
        elif sharpe_diff > 0:
            risk_adjusted_verdict = "👍 BETTER RISK-ADJUSTED RETURNS"
        elif sharpe_diff > -0.1:
            risk_adjusted_verdict = "➡️ SIMILAR RISK-ADJUSTED RETURNS"
        else:
            risk_adjusted_verdict = "⚠️ INFERIOR RISK-ADJUSTED RETURNS"

        report.append(f"RISK-ADJUSTED VERDICT: {risk_adjusted_verdict}")
        report.append("")

        # Strategy viability assessment
        report.append("4. STRATEGY VIABILITY ASSESSMENT")
        report.append("-" * 40)

        viability_factors = []

        if return_diff > 0:
            viability_factors.append("✅ Positive excess returns")
        else:
            viability_factors.append("❌ Negative excess returns")

        if sharpe_diff > 0:
            viability_factors.append("✅ Better risk-adjusted performance")
        else:
            viability_factors.append("❌ Worse risk-adjusted performance")

        if abs(vol_diff) < 0.05:  # Similar volatility
            viability_factors.append("✅ Comparable risk level")
        elif vol_diff > 0:
            viability_factors.append("⚠️ Higher volatility")
        else:
            viability_factors.append("✅ Lower volatility")

        for factor in viability_factors:
            report.append(f"   {factor}")
        report.append("")

        # Final conclusion
        report.append("5. FINAL CONCLUSION")
        report.append("-" * 40)

        positive_factors = sum(
            1 for factor in viability_factors if factor.startswith("✅"))

        if positive_factors >= 2:
            conclusion = "🎯 STRATEGY IS VIABLE - Model-driven approach shows promise"
            recommendation = "RECOMMENDED: Implement the optimized portfolio strategy"
        elif positive_factors == 1:
            conclusion = "⚠️ STRATEGY SHOWS MIXED RESULTS - Requires further refinement"
            recommendation = "CAUTION: Consider strategy improvements before implementation"
        else:
            conclusion = "❌ STRATEGY UNDERPERFORMED - Model needs significant improvement"
            recommendation = "NOT RECOMMENDED: Stick with benchmark or improve forecasting model"

        report.append(conclusion)
        report.append("")
        report.append("RECOMMENDATION:")
        report.append(recommendation)
        report.append("")

        # Model-driven approach assessment
        report.append("6. MODEL-DRIVEN APPROACH ASSESSMENT")
        report.append("-" * 40)
        report.append("This backtest validates the viability of using:")
        report.append("• ARIMA forecasting for expected returns estimation")
        report.append("• Modern Portfolio Theory for optimization")
        report.append("• Historical data for risk modeling")
        report.append("")

        if positive_factors >= 2:
            model_assessment = "The model-driven approach demonstrates value in portfolio construction."
        else:
            model_assessment = "The model-driven approach requires refinement for practical application."

        report.append(f"ASSESSMENT: {model_assessment}")
        report.append("")

        report.append("=" * 80)
        report.append("END OF BACKTESTING REPORT")
        report.append("=" * 80)

        return "\n".join(report)

    def analyze_performance(self):
        """
        Analyze and compare performance metrics
        """
        if not self.backtest_results:
            raise ValueError("Run backtest first")

        print("\n🎯 PERFORMANCE ANALYSIS")
        print("=" * 50)

        benchmark = self.backtest_results['benchmark']
        strategy = self.backtest_results['strategy']

        # Performance comparison table
        comparison_data = {
            'Metric': ['Total Return', 'Annualized Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown'],
            'Benchmark': [
                f"{benchmark['total_return']:.2%}",
                f"{benchmark['annualized_return']:.2%}",
                f"{benchmark['volatility']:.2%}",
                f"{benchmark['sharpe_ratio']:.3f}",
                f"{benchmark['max_drawdown']:.2%}"
            ],
            'Strategy': [
                f"{strategy['total_return']:.2%}",
                f"{strategy['annualized_return']:.2%}",
                f"{strategy['volatility']:.2%}",
                f"{strategy['sharpe_ratio']:.3f}",
                f"{strategy['max_drawdown']:.2%}"
            ]
        }

        comparison_df = pd.DataFrame(comparison_data)
        print("\nPerformance Comparison:")
        print(comparison_df.to_string(index=False))

        return comparison_df


def main():
    """
    Main execution function for Task 5: Strategy Backtesting
    """
    print("=" * 70)
    print("TASK 5: STRATEGY BACKTESTING")
    print("Validating Portfolio Strategy Performance")
    print("=" * 70)

    try:
        # Initialize backtester
        backtester = StrategyBacktester(assets=['TSLA', 'BND', 'SPY'])

        # Load historical data
        backtester.load_historical_data()

        # Run complete backtest
        results = backtester.run_backtest()

        # Analyze performance
        backtester.analyze_performance()

        # Plot performance comparison
        backtester.plot_performance_comparison()

        # Generate comprehensive report
        print("\n📝 Generating backtesting report...")
        report = backtester.generate_backtest_report()

        # Save report
        os.makedirs('results', exist_ok=True)
        report_path = 'results/strategy_backtesting_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print("\n" + "=" * 70)
        print("STRATEGY BACKTESTING COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(report)

        print(f"\nBacktesting report saved as '{report_path}'")
        print("Performance comparison plot saved as 'results/figures/backtest_performance.png'")

    except Exception as e:
        print(f"\nError in Task 5 execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
