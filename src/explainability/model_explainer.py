"""
Model explainability module using SHAP and other interpretability techniques.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from ..config import config
from ..utils.logging_config import setup_logging, log_performance

logger = setup_logging("model_explainer")


class PortfolioExplainer:
    """Explainability tools for portfolio management models."""
    
    def __init__(self):
        self.feature_names = []
        self.shap_explainer = None
        self.shap_values = None
        self.feature_importance = {}
        
    @log_performance
    def explain_portfolio_allocation(self, 
                                   returns_data: pd.DataFrame,
                                   portfolio_weights: Dict[str, float],
                                   risk_factors: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Explain portfolio allocation decisions using SHAP values.
        
        Args:
            returns_data: Historical returns for assets
            portfolio_weights: Current portfolio weights
            risk_factors: Additional risk factors (optional)
            
        Returns:
            Dictionary containing explanation results
        """
        logger.info("Starting portfolio allocation explanation")
        
        # Prepare features for explanation
        features = self._prepare_features(returns_data, risk_factors)
        
        # Create target variable (portfolio returns)
        target = self._calculate_portfolio_returns(returns_data, portfolio_weights)
        
        # Train surrogate model for explanation
        surrogate_model = self._train_surrogate_model(features, target)
        
        # Generate SHAP explanations
        shap_results = self._generate_shap_explanations(surrogate_model, features)
        
        # Calculate feature importance
        feature_importance = self._calculate_feature_importance(surrogate_model, features)
        
        # Generate visualizations
        plots = self._create_explanation_plots(shap_results, feature_importance)
        
        results = {
            'shap_values': shap_results['shap_values'],
            'feature_importance': feature_importance,
            'surrogate_model_score': shap_results['model_score'],
            'plots': plots,
            'feature_names': self.feature_names,
            'explanation_summary': self._generate_explanation_summary(feature_importance)
        }
        
        logger.info("Portfolio allocation explanation completed")
        return results
    
    def _prepare_features(self, returns_data: pd.DataFrame, risk_factors: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Prepare features for model explanation."""
        features = pd.DataFrame()
        
        # Statistical features from returns
        for asset in returns_data.columns:
            asset_returns = returns_data[asset].dropna()
            
            # Basic statistics
            features[f'{asset}_mean'] = [asset_returns.mean()]
            features[f'{asset}_std'] = [asset_returns.std()]
            features[f'{asset}_skew'] = [asset_returns.skew()]
            features[f'{asset}_kurt'] = [asset_returns.kurtosis()]
            
            # Rolling statistics (last 30 days)
            if len(asset_returns) >= 30:
                features[f'{asset}_rolling_mean_30'] = [asset_returns.tail(30).mean()]
                features[f'{asset}_rolling_std_30'] = [asset_returns.tail(30).std()]
            else:
                features[f'{asset}_rolling_mean_30'] = [asset_returns.mean()]
                features[f'{asset}_rolling_std_30'] = [asset_returns.std()]
            
            # Momentum indicators
            if len(asset_returns) >= 5:
                features[f'{asset}_momentum_5d'] = [asset_returns.tail(5).sum()]
            else:
                features[f'{asset}_momentum_5d'] = [0]
                
            if len(asset_returns) >= 21:
                features[f'{asset}_momentum_21d'] = [asset_returns.tail(21).sum()]
            else:
                features[f'{asset}_momentum_21d'] = [0]
        
        # Cross-asset features
        if len(returns_data.columns) > 1:
            corr_matrix = returns_data.corr()
            for i, asset1 in enumerate(returns_data.columns):
                for j, asset2 in enumerate(returns_data.columns):
                    if i < j:
                        features[f'corr_{asset1}_{asset2}'] = [corr_matrix.loc[asset1, asset2]]
        
        # Risk factors if provided
        if risk_factors is not None:
            for factor in risk_factors.columns:
                features[f'risk_factor_{factor}'] = [risk_factors[factor].iloc[-1]]
        
        # Market regime features
        market_returns = returns_data.mean(axis=1)
        if len(market_returns) >= 21:
            features['market_volatility'] = [market_returns.tail(21).std()]
            features['market_trend'] = [market_returns.tail(21).mean()]
        else:
            features['market_volatility'] = [market_returns.std()]
            features['market_trend'] = [market_returns.mean()]
        
        self.feature_names = list(features.columns)
        
        # Replicate features for multiple observations (SHAP needs multiple samples)
        # Add some noise to create variation
        n_samples = 100
        expanded_features = pd.DataFrame()
        
        for _ in range(n_samples):
            sample = features.copy()
            # Add small random noise to create variation
            noise = np.random.normal(0, 0.01, sample.shape)
            sample += noise
            expanded_features = pd.concat([expanded_features, sample], ignore_index=True)
        
        return expanded_features
    
    def _calculate_portfolio_returns(self, returns_data: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
        """Calculate portfolio returns based on weights."""
        portfolio_returns = pd.Series(0, index=returns_data.index)
        
        for asset, weight in weights.items():
            if asset in returns_data.columns:
                portfolio_returns += returns_data[asset] * weight
        
        return portfolio_returns
    
    def _train_surrogate_model(self, features: pd.DataFrame, target: pd.Series) -> RandomForestRegressor:
        """Train a surrogate model for explanation."""
        # Expand target to match features
        expanded_target = np.tile(target.mean(), len(features))
        
        # Add noise to target to create variation
        noise = np.random.normal(0, target.std() * 0.1, len(expanded_target))
        expanded_target += noise
        
        # Train Random Forest as surrogate model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=config.system.random_seed,
            n_jobs=config.system.n_jobs
        )
        
        model.fit(features, expanded_target)
        
        logger.info(f"Surrogate model trained with R² score: {model.score(features, expanded_target):.3f}")
        return model
    
    def _generate_shap_explanations(self, model: RandomForestRegressor, features: pd.DataFrame) -> Dict[str, Any]:
        """Generate SHAP explanations for the model."""
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features)
        
        # Calculate model performance
        predictions = model.predict(features)
        model_score = model.score(features, predictions)
        
        self.shap_explainer = explainer
        self.shap_values = shap_values
        
        return {
            'explainer': explainer,
            'shap_values': shap_values,
            'base_value': explainer.expected_value,
            'model_score': model_score,
            'features': features
        }
    
    def _calculate_feature_importance(self, model: RandomForestRegressor, features: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature importance from the surrogate model."""
        importance_scores = model.feature_importances_
        feature_importance = dict(zip(self.feature_names, importance_scores))
        
        # Sort by importance
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        self.feature_importance = feature_importance
        return feature_importance
    
    def _create_explanation_plots(self, shap_results: Dict[str, Any], feature_importance: Dict[str, float]) -> Dict[str, Any]:
        """Create explanation visualizations."""
        plots = {}
        
        # Feature importance plot
        plots['feature_importance'] = self._plot_feature_importance(feature_importance)
        
        # SHAP summary plot
        plots['shap_summary'] = self._plot_shap_summary(shap_results)
        
        # SHAP waterfall plot
        plots['shap_waterfall'] = self._plot_shap_waterfall(shap_results)
        
        # Feature interaction plot
        plots['feature_interactions'] = self._plot_feature_interactions(shap_results)
        
        return plots
    
    def _plot_feature_importance(self, feature_importance: Dict[str, float]) -> go.Figure:
        """Create feature importance plot."""
        # Get top 15 features
        top_features = dict(list(feature_importance.items())[:15])
        
        fig = go.Figure(go.Bar(
            x=list(top_features.values()),
            y=list(top_features.keys()),
            orientation='h',
            marker_color='steelblue'
        ))
        
        fig.update_layout(
            title='Top Feature Importance for Portfolio Allocation',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            height=600,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    def _plot_shap_summary(self, shap_results: Dict[str, Any]) -> go.Figure:
        """Create SHAP summary plot."""
        shap_values = shap_results['shap_values']
        features = shap_results['features']
        
        # Calculate mean absolute SHAP values for each feature
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        # Get top 15 features
        top_indices = np.argsort(mean_shap)[-15:]
        top_features = [self.feature_names[i] for i in top_indices]
        top_shap_values = mean_shap[top_indices]
        
        fig = go.Figure(go.Bar(
            x=top_shap_values,
            y=top_features,
            orientation='h',
            marker_color='coral'
        ))
        
        fig.update_layout(
            title='SHAP Feature Impact on Portfolio Returns',
            xaxis_title='Mean |SHAP Value|',
            yaxis_title='Features',
            height=600,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    def _plot_shap_waterfall(self, shap_results: Dict[str, Any]) -> go.Figure:
        """Create SHAP waterfall plot for a single prediction."""
        shap_values = shap_results['shap_values'][0]  # First sample
        base_value = shap_results['base_value']
        
        # Get top 10 features by absolute SHAP value
        abs_shap = np.abs(shap_values)
        top_indices = np.argsort(abs_shap)[-10:]
        
        # Create waterfall data
        values = [base_value]
        labels = ['Base Value']
        
        cumulative = base_value
        for idx in top_indices:
            values.append(shap_values[idx])
            labels.append(self.feature_names[idx])
            cumulative += shap_values[idx]
        
        values.append(cumulative)
        labels.append('Final Prediction')
        
        # Create waterfall chart
        fig = go.Figure(go.Waterfall(
            name="SHAP Waterfall",
            orientation="v",
            measure=["absolute"] + ["relative"] * len(top_indices) + ["total"],
            x=labels,
            y=values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(
            title="SHAP Waterfall Plot - Feature Contributions",
            xaxis_title="Features",
            yaxis_title="SHAP Value",
            height=500
        )
        
        return fig
    
    def _plot_feature_interactions(self, shap_results: Dict[str, Any]) -> go.Figure:
        """Create feature interaction plot."""
        shap_values = shap_results['shap_values']
        
        # Calculate feature interaction strength (simplified)
        n_features = len(self.feature_names)
        interaction_matrix = np.zeros((n_features, n_features))
        
        for i in range(n_features):
            for j in range(n_features):
                if i != j:
                    # Correlation between SHAP values as proxy for interaction
                    interaction_matrix[i, j] = np.corrcoef(shap_values[:, i], shap_values[:, j])[0, 1]
        
        # Create heatmap
        fig = px.imshow(
            interaction_matrix,
            x=self.feature_names,
            y=self.feature_names,
            color_continuous_scale='RdBu_r',
            title='Feature Interaction Heatmap (SHAP Correlations)'
        )
        
        fig.update_layout(height=600)
        return fig
    
    def _generate_explanation_summary(self, feature_importance: Dict[str, float]) -> Dict[str, Any]:
        """Generate a text summary of the explanation."""
        top_5_features = list(feature_importance.items())[:5]
        
        summary = {
            'most_important_feature': top_5_features[0][0],
            'most_important_score': top_5_features[0][1],
            'top_5_features': [f[0] for f in top_5_features],
            'explanation_text': f"The most important factor in portfolio allocation is '{top_5_features[0][0]}' "
                              f"with an importance score of {top_5_features[0][1]:.3f}. "
                              f"The top 5 factors are: {', '.join([f[0] for f in top_5_features])}."
        }
        
        return summary


class ForecastExplainer:
    """Explainability tools for forecasting models."""
    
    def __init__(self):
        self.feature_names = []
        self.explanation_results = {}
    
    @log_performance
    def explain_forecast_drivers(self, 
                               historical_data: pd.DataFrame,
                               forecast_values: np.ndarray,
                               model_type: str = "ARIMA") -> Dict[str, Any]:
        """
        Explain what drives the forecast predictions.
        
        Args:
            historical_data: Historical price/return data
            forecast_values: Forecast predictions
            model_type: Type of forecasting model
            
        Returns:
            Dictionary containing explanation results
        """
        logger.info(f"Starting forecast explanation for {model_type} model")
        
        if model_type.upper() == "ARIMA":
            return self._explain_arima_forecast(historical_data, forecast_values)
        else:
            return self._explain_generic_forecast(historical_data, forecast_values)
    
    def _explain_arima_forecast(self, data: pd.DataFrame, forecast: np.ndarray) -> Dict[str, Any]:
        """Explain ARIMA forecast drivers."""
        target_col = 'Close' if 'Close' in data.columns else data.columns[0]
        prices = data[target_col].dropna()
        
        # Calculate various time series features
        features = self._extract_time_series_features(prices)
        
        # Analyze trend and seasonality components
        trend_analysis = self._analyze_trend_components(prices)
        
        # Volatility analysis
        volatility_analysis = self._analyze_volatility_patterns(prices)
        
        # Create visualizations
        plots = self._create_forecast_explanation_plots(prices, forecast, features)
        
        explanation = {
            'model_type': 'ARIMA',
            'features': features,
            'trend_analysis': trend_analysis,
            'volatility_analysis': volatility_analysis,
            'plots': plots,
            'summary': self._generate_forecast_summary(features, trend_analysis, volatility_analysis)
        }
        
        return explanation
    
    def _extract_time_series_features(self, prices: pd.Series) -> Dict[str, float]:
        """Extract interpretable time series features."""
        returns = prices.pct_change().dropna()
        
        features = {
            'mean_return': returns.mean(),
            'volatility': returns.std(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'autocorr_1': returns.autocorr(1),
            'autocorr_5': returns.autocorr(5),
            'autocorr_21': returns.autocorr(21),
            'recent_trend_5d': returns.tail(5).mean(),
            'recent_trend_21d': returns.tail(21).mean(),
            'recent_volatility_21d': returns.tail(21).std(),
            'max_drawdown': self._calculate_max_drawdown(prices),
            'current_price_vs_ma_21': (prices.iloc[-1] / prices.tail(21).mean()) - 1,
            'current_price_vs_ma_63': (prices.tail(63).mean() / prices.tail(252).mean()) - 1 if len(prices) >= 252 else 0
        }
        
        return features
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = prices / prices.iloc[0]
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()
    
    def _analyze_trend_components(self, prices: pd.Series) -> Dict[str, Any]:
        """Analyze trend components of the time series."""
        returns = prices.pct_change().dropna()
        
        # Simple trend analysis
        short_ma = prices.rolling(21).mean()
        long_ma = prices.rolling(63).mean()
        
        trend_analysis = {
            'current_trend': 'bullish' if short_ma.iloc[-1] > long_ma.iloc[-1] else 'bearish',
            'trend_strength': abs(short_ma.iloc[-1] / long_ma.iloc[-1] - 1),
            'momentum_5d': returns.tail(5).sum(),
            'momentum_21d': returns.tail(21).sum(),
            'trend_consistency': (returns.tail(21) > 0).mean(),  # Fraction of positive days
            'price_vs_short_ma': (prices.iloc[-1] / short_ma.iloc[-1]) - 1,
            'price_vs_long_ma': (prices.iloc[-1] / long_ma.iloc[-1]) - 1
        }
        
        return trend_analysis
    
    def _analyze_volatility_patterns(self, prices: pd.Series) -> Dict[str, Any]:
        """Analyze volatility patterns."""
        returns = prices.pct_change().dropna()
        
        # Rolling volatility
        vol_21d = returns.rolling(21).std()
        vol_63d = returns.rolling(63).std()
        
        volatility_analysis = {
            'current_volatility_21d': vol_21d.iloc[-1],
            'current_volatility_63d': vol_63d.iloc[-1],
            'volatility_trend': 'increasing' if vol_21d.iloc[-1] > vol_63d.iloc[-1] else 'decreasing',
            'volatility_percentile': (vol_21d.iloc[-1] > vol_21d).mean(),  # Current vol vs historical
            'extreme_moves_21d': (abs(returns.tail(21)) > 2 * returns.std()).sum(),
            'volatility_clustering': returns.tail(21).std() / returns.std()  # Recent vs overall volatility
        }
        
        return volatility_analysis
    
    def _create_forecast_explanation_plots(self, prices: pd.Series, forecast: np.ndarray, features: Dict[str, float]) -> Dict[str, Any]:
        """Create plots explaining the forecast."""
        plots = {}
        
        # Price and forecast plot
        plots['price_forecast'] = self._plot_price_forecast(prices, forecast)
        
        # Feature importance plot
        plots['feature_importance'] = self._plot_forecast_features(features)
        
        # Volatility analysis plot
        plots['volatility_analysis'] = self._plot_volatility_analysis(prices)
        
        return plots
    
    def _plot_price_forecast(self, prices: pd.Series, forecast: np.ndarray) -> go.Figure:
        """Plot historical prices with forecast."""
        fig = go.Figure()
        
        # Historical prices
        fig.add_trace(go.Scatter(
            x=prices.index,
            y=prices.values,
            mode='lines',
            name='Historical Prices',
            line=dict(color='blue', width=2)
        ))
        
        # Forecast
        forecast_dates = pd.date_range(
            start=prices.index[-1] + pd.Timedelta(days=1),
            periods=len(forecast),
            freq='D'
        )
        
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast,
            mode='lines',
            name='Forecast',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Price History and Forecast',
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified'
        )
        
        return fig
    
    def _plot_forecast_features(self, features: Dict[str, float]) -> go.Figure:
        """Plot forecast driving features."""
        # Select most relevant features
        relevant_features = {
            'Recent Trend (5d)': features['recent_trend_5d'],
            'Recent Trend (21d)': features['recent_trend_21d'],
            'Volatility': features['volatility'],
            'Momentum (21d)': features['momentum_21d'],
            'Autocorrelation (1d)': features['autocorr_1'],
            'Price vs MA(21)': features['current_price_vs_ma_21']
        }
        
        fig = go.Figure(go.Bar(
            x=list(relevant_features.keys()),
            y=list(relevant_features.values()),
            marker_color=['green' if v > 0 else 'red' for v in relevant_features.values()]
        ))
        
        fig.update_layout(
            title='Key Forecast Driving Factors',
            xaxis_title='Features',
            yaxis_title='Feature Value',
            xaxis_tickangle=-45
        )
        
        return fig
    
    def _plot_volatility_analysis(self, prices: pd.Series) -> go.Figure:
        """Plot volatility analysis."""
        returns = prices.pct_change().dropna()
        vol_21d = returns.rolling(21).std() * np.sqrt(252)  # Annualized
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Price Evolution', 'Rolling Volatility (21d)'],
            vertical_spacing=0.1
        )
        
        # Price plot
        fig.add_trace(
            go.Scatter(x=prices.index, y=prices.values, name='Price', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Volatility plot
        fig.add_trace(
            go.Scatter(x=vol_21d.index, y=vol_21d.values, name='Volatility', line=dict(color='orange')),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Price and Volatility Analysis',
            height=600,
            showlegend=False
        )
        
        return fig
    
    def _generate_forecast_summary(self, features: Dict[str, float], trend: Dict[str, Any], volatility: Dict[str, Any]) -> str:
        """Generate a summary of forecast drivers."""
        summary = f"""
        Forecast Analysis Summary:
        
        Trend Analysis:
        - Current trend: {trend['current_trend']}
        - Trend strength: {trend['trend_strength']:.3f}
        - 21-day momentum: {trend['momentum_21d']:.3f}
        
        Volatility Analysis:
        - Current volatility trend: {volatility['volatility_trend']}
        - Volatility clustering factor: {volatility['volatility_clustering']:.2f}
        - Recent extreme moves: {volatility['extreme_moves_21d']} in last 21 days
        
        Key Features:
        - Recent 5-day trend: {features['recent_trend_5d']:.4f}
        - Autocorrelation (1-day): {features['autocorr_1']:.3f}
        - Price vs 21-day MA: {features['current_price_vs_ma_21']:.3f}
        """
        
        return summary.strip()
    
    def _explain_generic_forecast(self, data: pd.DataFrame, forecast: np.ndarray) -> Dict[str, Any]:
        """Generic forecast explanation for other model types."""
        return {
            'model_type': 'Generic',
            'message': 'Detailed explanation not available for this model type',
            'forecast_length': len(forecast),
            'forecast_mean': np.mean(forecast),
            'forecast_std': np.std(forecast)
        }
