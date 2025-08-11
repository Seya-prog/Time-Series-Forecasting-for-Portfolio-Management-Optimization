"""
Basic tests for portfolio management system
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from portfolio.portfolio_optimization import PortfolioOptimizer
from backtesting.strategy_backtesting import StrategyBacktester


class TestPortfolioOptimizer(unittest.TestCase):
    """Test portfolio optimization functionality"""
    
    def setUp(self):
        self.optimizer = PortfolioOptimizer()
        
    def test_portfolio_initialization(self):
        """Test portfolio optimizer initialization"""
        self.assertEqual(self.optimizer.assets, ['TSLA', 'BND', 'SPY'])
        self.assertEqual(self.optimizer.risk_free_rate, 0.02)
        
    def test_portfolio_performance_calculation(self):
        """Test portfolio performance calculation"""
        # Mock data
        self.optimizer.expected_returns = pd.Series([0.1, 0.05, 0.08], index=['TSLA', 'BND', 'SPY'])
        self.optimizer.cov_matrix = pd.DataFrame(
            [[0.04, 0.01, 0.02], [0.01, 0.01, 0.005], [0.02, 0.005, 0.02]],
            index=['TSLA', 'BND', 'SPY'], columns=['TSLA', 'BND', 'SPY']
        )
        
        weights = [0.3, 0.3, 0.4]
        ret, vol, sharpe = self.optimizer.portfolio_performance(weights)
        
        self.assertIsInstance(ret, float)
        self.assertIsInstance(vol, float)
        self.assertIsInstance(sharpe, float)
        self.assertGreater(vol, 0)


class TestStrategyBacktester(unittest.TestCase):
    """Test strategy backtesting functionality"""
    
    def setUp(self):
        self.backtester = StrategyBacktester()
        
    def test_backtester_initialization(self):
        """Test backtester initialization"""
        self.assertEqual(self.backtester.assets, ['TSLA', 'BND', 'SPY'])
        self.assertEqual(self.backtester.benchmark_weights['SPY'], 0.6)
        self.assertEqual(self.backtester.benchmark_weights['BND'], 0.4)


if __name__ == '__main__':
    unittest.main()
