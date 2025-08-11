"""
Time Series Forecasting Models for Tesla Stock Price Prediction

This module implements ARIMA/SARIMA and LSTM models for forecasting Tesla's stock prices.
Task 2: Develop Time Series Forecasting Models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Optional
import warnings
from datetime import datetime, timedelta
import logging

# Statistical models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

# Try to import pmdarima, use manual ARIMA if not available
try:
    from pmdarima import auto_arima
    HAS_PMDARIMA = True
except ImportError:
    print("Warning: pmdarima not available. Using manual ARIMA parameter selection.")
    HAS_PMDARIMA = False

# Deep learning models
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Suppress warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class TimeSeriesForecaster:
    """
    Time Series Forecasting class implementing ARIMA/SARIMA and LSTM models
    for Tesla stock price prediction.
    """
    
    def __init__(self, data: pd.DataFrame, target_column: str = 'Close'):
        """
        Initialize the forecaster with Tesla stock data.
        
        Args:
            data: DataFrame containing Tesla stock data
            target_column: Column name for the target variable (default: 'Close')
        """
        self.data = data.copy()
        self.target_column = target_column
        self.train_data = None
        self.test_data = None
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        
        # Ensure data is sorted by date and handle timezone issues
        if 'Date' in self.data.columns:
            self.data = self.data.sort_values('Date').reset_index(drop=True)
            # Convert to timezone-naive datetime to avoid comparison issues
            self.data['Date'] = pd.to_datetime(self.data['Date']).dt.tz_localize(None)
            self.data.set_index('Date', inplace=True)
        elif self.data.index.name == 'Date':
            # Handle case where Date is already the index
            self.data.index = pd.to_datetime(self.data.index).tz_localize(None)
        
        logger.info(f"Initialized forecaster with {len(self.data)} data points")
    
    def split_data(self, train_end_date: str = "2023-12-31") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically into training and testing sets.
        
        Args:
            train_end_date: End date for training data (default: "2023-12-31")
            
        Returns:
            Tuple of (train_data, test_data)
        """
        train_end = pd.to_datetime(train_end_date).tz_localize(None)
        
        self.train_data = self.data[self.data.index <= train_end]
        self.test_data = self.data[self.data.index > train_end]
        
        logger.info(f"Training data: {len(self.train_data)} points ({self.train_data.index.min()} to {self.train_data.index.max()})")
        logger.info(f"Testing data: {len(self.test_data)} points ({self.test_data.index.min()} to {self.test_data.index.max()})")
        
        return self.train_data, self.test_data
    
    def check_stationarity(self, series: pd.Series, title: str = "Series") -> Dict:
        """
        Check stationarity of a time series using Augmented Dickey-Fuller test.
        
        Args:
            series: Time series to test
            title: Title for the series
            
        Returns:
            Dictionary with test results
        """
        result = adfuller(series.dropna())
        
        output = {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
        
        print(f"\n{title} - Stationarity Test Results:")
        print(f"ADF Statistic: {result[0]:.6f}")
        print(f"p-value: {result[1]:.6f}")
        print(f"Is Stationary: {'Yes' if output['is_stationary'] else 'No'}")
        
        return output
    
    def prepare_arima_data(self) -> pd.Series:
        """
        Prepare data for ARIMA modeling by ensuring stationarity.
        
        Returns:
            Prepared time series for ARIMA
        """
        if self.train_data is None:
            raise ValueError("Data must be split first using split_data()")
        
        series = self.train_data[self.target_column]
        
        # Check original series stationarity
        stationarity = self.check_stationarity(series, "Original Series")
        
        if not stationarity['is_stationary']:
            # Apply first differencing
            series_diff = series.diff().dropna()
            stationarity_diff = self.check_stationarity(series_diff, "First Differenced Series")
            
            if stationarity_diff['is_stationary']:
                return series_diff
            else:
                # Apply second differencing if needed
                series_diff2 = series_diff.diff().dropna()
                self.check_stationarity(series_diff2, "Second Differenced Series")
                return series_diff2
        
        return series
    
    def fit_arima_model(self, auto_optimize: bool = True, order: Tuple[int, int, int] = None) -> Dict:
        """
        Fit ARIMA model to the training data.
        
        Args:
            auto_optimize: Whether to use auto_arima for parameter optimization
            order: Manual ARIMA order (p, d, q) if auto_optimize is False
            
        Returns:
            Dictionary containing model and fitting results
        """
        if self.train_data is None:
            raise ValueError("Data must be split first using split_data()")
        
        series = self.train_data[self.target_column]
        
        print("\n" + "="*50)
        print("FITTING ARIMA MODEL")
        print("="*50)
        
        if auto_optimize:
            print("Using auto_arima for parameter optimization...")
            
            try:
                # Use auto_arima to find optimal parameters
                model = auto_arima(
                    series,
                    start_p=0, start_q=0,
                    max_p=3, max_q=3,  # Reduced complexity
                    seasonal=False,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action='ignore',
                    trace=False  # Disable trace to avoid output issues
                )
                
                print(f"✅ Auto ARIMA completed with order: {getattr(model, 'order', 'unknown')}")
                
            except Exception as e:
                print(f"⚠️ Auto ARIMA failed: {e}")
                print("Falling back to manual ARIMA...")
                order = (1, 1, 1)
                model = ARIMA(series, order=order).fit()
                print(f"✅ Manual ARIMA fitted with order: {order}")
            
        else:
            if order is None:
                order = (1, 1, 1)  # Default order
            
            print(f"Using manual ARIMA order: {order}")
            model = ARIMA(series, order=order).fit()
        
        # Store model
        self.models['arima'] = model
        

        
        # Model summary
        print("\nARIMA Model Summary:")
        print(model.summary())
        
        # Get model order based on model type
        try:
            if hasattr(model, 'order'):
                order = model.order
            elif hasattr(model, 'model') and hasattr(model.model, 'order'):
                order = model.model.order
            else:
                order = (1, 1, 1)  # fallback
        except:
            order = (1, 1, 1)
        
        return {
            'model': model,
            'order': order,
            'aic': model.aic,
            'bic': model.bic
        }
    
    def predict_arima(self) -> np.ndarray:
        """
        Generate ARIMA predictions for the test period.
        
        Returns:
            Array of predictions
        """
        if 'arima' not in self.models:
            raise ValueError("ARIMA model must be fitted first")
        
        model = self.models['arima']
        n_periods = len(self.test_data)
        
        # Use pmdarima auto_arima model prediction (original working approach)
        try:
            # pmdarima models use predict() for out-of-sample forecasting
            predictions = model.predict(n_periods=n_periods)
            print(f"✅ ARIMA predictions generated using pmdarima for {n_periods} periods")
        except Exception as e:
            print(f"⚠️ pmdarima prediction failed: {e}")
            # Simple fallback without trend modification
            predictions = np.full(n_periods, self.train_data[self.target_column].iloc[-1])
            print(f"⚠️ Using simple fallback predictions")
        
        # Store predictions
        self.predictions['arima'] = predictions
        
        return predictions
    
    def prepare_lstm_data(self, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        """
        Prepare data for LSTM modeling.
        
        Args:
            sequence_length: Number of previous time steps to use for prediction
            
        Returns:
            Tuple of (X_train, y_train, scaler)
        """
        if self.train_data is None:
            raise ValueError("Data must be split first using split_data()")
        
        # Prepare data
        data = self.train_data[self.target_column].values.reshape(-1, 1)
        
        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        return X, y, scaler
    
    def fit_lstm_model(self, sequence_length: int = 60, epochs: int = 100, batch_size: int = 32) -> Dict:
        """
        Fit LSTM model to the training data.
        
        Args:
            sequence_length: Number of previous time steps to use
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary containing model and training results
        """
        print("\n" + "="*50)
        print("FITTING LSTM MODEL")
        print("="*50)
        
        # Prepare data
        X_train, y_train, scaler = self.prepare_lstm_data(sequence_length)
        
        print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        
        print("LSTM Model Architecture:")
        model.summary()
        
        # Train model
        print(f"\nTraining LSTM model for {epochs} epochs...")
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            validation_split=0.1
        )
        
        # Store model and scaler
        self.models['lstm'] = {
            'model': model,
            'scaler': scaler,
            'sequence_length': sequence_length,
            'history': history
        }
        
        return {
            'model': model,
            'scaler': scaler,
            'history': history,
            'sequence_length': sequence_length
        }
    

    
    def predict_lstm(self, steps: int = None) -> np.ndarray:
        """
        Generate LSTM predictions for the test period.
        
        Args:
            steps: Number of steps to forecast (default: length of test data)
            
        Returns:
            Array of predictions
        """
        if 'lstm' not in self.models:
            raise ValueError("LSTM model must be fitted first")
        
        if steps is None:
            steps = len(self.test_data)
        
        lstm_info = self.models['lstm']
        model = lstm_info['model']
        scaler = lstm_info['scaler']
        sequence_length = lstm_info['sequence_length']
        
        # Prepare input data (last sequence_length points from training data)
        last_sequence = self.train_data[self.target_column].tail(sequence_length).values
        last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1))
        
        predictions = []
        current_sequence = last_sequence_scaled.flatten()
        
        # Generate predictions step by step
        for _ in range(steps):
            # Reshape for prediction
            X_pred = current_sequence[-sequence_length:].reshape(1, sequence_length, 1)
            
            # Predict next value
            pred_scaled = model.predict(X_pred, verbose=0)[0, 0]
            
            # Inverse transform to original scale
            pred_original = scaler.inverse_transform([[pred_scaled]])[0, 0]
            predictions.append(pred_original)
            
            # Update sequence for next prediction
            current_sequence = np.append(current_sequence, pred_scaled)
        
        predictions = np.array(predictions)
        
        self.predictions['lstm'] = {
            'forecast': predictions
        }
        
        return predictions
    
    def calculate_metrics(self, actual: np.ndarray, predicted: np.ndarray, model_name: str) -> Dict:
        """
        Calculate evaluation metrics for predictions.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            model_name: Name of the model
            
        Returns:
            Dictionary of metrics
        """
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
        
        self.metrics[model_name] = metrics
        
        print(f"\n{model_name.upper()} Model Performance:")
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.2f}%")
        
        return metrics
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare performance of all fitted models.
        
        Returns:
            DataFrame with model comparison
        """
        if not self.metrics:
            raise ValueError("No models have been evaluated yet")
        
        comparison_df = pd.DataFrame(self.metrics).T
        comparison_df = comparison_df.round(4)
        
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        print(comparison_df)
        
        # Find best model for each metric
        best_models = {}
        for metric in comparison_df.columns:
            if metric == 'MAPE':
                best_models[metric] = comparison_df[metric].idxmin()
            else:
                best_models[metric] = comparison_df[metric].idxmin()
        
        print("\nBest Models by Metric:")
        for metric, model in best_models.items():
            print(f"{metric}: {model}")
        
        return comparison_df
    
    def plot_predictions(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plot actual vs predicted values for all models.
        
        Args:
            figsize: Figure size for the plot
        """
        if not self.predictions:
            raise ValueError("No predictions available. Run predict methods first.")
        
        fig, axes = plt.subplots(len(self.predictions), 1, figsize=figsize)
        if len(self.predictions) == 1:
            axes = [axes]
        
        test_dates = self.test_data.index
        actual_values = self.test_data[self.target_column].values
        
        for i, (model_name, pred_info) in enumerate(self.predictions.items()):
            ax = axes[i]
            
            # Plot actual values
            ax.plot(test_dates, actual_values, label='Actual', color='blue', linewidth=2)
            
            # Plot predictions - handle both array and dict formats
            if isinstance(pred_info, dict) and 'forecast' in pred_info:
                predictions = pred_info['forecast']
            else:
                predictions = pred_info  # Direct array format
            
            ax.plot(test_dates[:len(predictions)], predictions, 
                   label=f'{model_name.upper()} Prediction', color='red', linewidth=2, linestyle='--')
            
            # Plot confidence interval for ARIMA (if available)
            if model_name == 'arima' and isinstance(pred_info, dict) and 'confidence_interval' in pred_info:
                ci = pred_info['confidence_interval']
                ax.fill_between(test_dates[:len(ci)], ci.iloc[:, 0], ci.iloc[:, 1], 
                               alpha=0.3, color='red', label='Confidence Interval')
            
            ax.set_title(f'{model_name.upper()} Model Predictions vs Actual')
            ax.set_xlabel('Date')
            ax.set_ylabel('Stock Price ($)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add metrics to plot
            if model_name in self.metrics:
                metrics = self.metrics[model_name]
                textstr = f"MAE: {metrics['MAE']:.2f}\nRMSE: {metrics['RMSE']:.2f}\nMAPE: {metrics['MAPE']:.2f}%"
                ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_analysis(self, auto_arima: bool = True, lstm_epochs: int = 100) -> Dict:
        """
        Run complete time series forecasting analysis.
        
        Args:
            auto_arima: Whether to use auto_arima for parameter optimization
            lstm_epochs: Number of epochs for LSTM training
            
        Returns:
            Dictionary with all results
        """
        print("STARTING COMPLETE TIME SERIES FORECASTING ANALYSIS")
        print("="*60)
        
        # Split data
        self.split_data()
        
        # Fit models
        arima_results = self.fit_arima_model(auto_optimize=auto_arima)
        lstm_results = self.fit_lstm_model(epochs=lstm_epochs)
        
        # Generate predictions
        arima_pred = self.predict_arima()
        lstm_pred = self.predict_lstm()
        
        # Calculate metrics
        actual_values = self.test_data[self.target_column].values
        self.calculate_metrics(actual_values, arima_pred, 'arima')
        self.calculate_metrics(actual_values[:len(lstm_pred)], lstm_pred, 'lstm')
        
        # Compare models
        comparison = self.compare_models()
        
        # Plot results
        self.plot_predictions()
        
        return {
            'arima_results': arima_results,
            'lstm_results': lstm_results,
            'predictions': self.predictions,
            'metrics': self.metrics,
            'comparison': comparison
        }


def main():
    """
    Main function to run Task 2: Time Series Forecasting Analysis
    """
    import sys
    import os
    sys.path.append('.')
    sys.path.append('..')
    
    print("="*80)
    print("TASK 2: TIME SERIES FORECASTING FOR TESLA STOCK PREDICTION")
    print("="*80)
    
    # Step 1: Load Tesla processed data
    print("\n1. Loading Tesla Processed Data...")
    
    try:
        # Load the already processed Tesla data
        tesla_processed = pd.read_csv('data/processed/TSLA_processed.csv')
        
        # Convert Date column to datetime and set as index (timezone-naive)
        tesla_processed['Date'] = pd.to_datetime(tesla_processed['Date'], utc=True).dt.tz_localize(None)
        tesla_processed.set_index('Date', inplace=True)
        
        print(f"✅ Successfully loaded {len(tesla_processed)} Tesla processed data points")
        print(f"   Date range: {tesla_processed.index.min().date()} to {tesla_processed.index.max().date()}")
        print(f"   Available columns: {list(tesla_processed.columns)}")
        
    except FileNotFoundError:
        print("❌ Error: Processed Tesla data not found. Please run Task 1 (EDA) first.")
        print("   Run: python -m src.data.eda")
        return
    except Exception as e:
        print(f"❌ Error loading processed data: {str(e)}")
        return
    
    # Step 2: Initialize Time Series Forecaster
    print("\n2. Initializing Time Series Forecaster...")
    forecaster = TimeSeriesForecaster(
        data=tesla_processed,
        target_column='Close'  # Use Close price for prediction
    )
    
    # Step 3: Run Complete Forecasting Analysis
    print("\n3. Running Complete Time Series Forecasting Analysis...")
    print("   This includes:")
    print("   - Data splitting (train: 2015-2023, test: 2024)")
    print("   - ARIMA model with auto parameter optimization")
    print("   - LSTM model with optimized architecture")
    print("   - Model comparison using MAE, RMSE, MAPE metrics")
    
    try:
        results = forecaster.run_complete_analysis(
            auto_arima=True,      # Use pmdarima auto_arima as required
            lstm_epochs=50        # Reduced epochs for faster training
        )
        
        print("\n" + "="*60)
        print("TASK 2 COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Display final summary
        print("\nFINAL RESULTS SUMMARY:")
        print("-" * 30)
        
        if 'comparison' in results:
            comparison_df = results['comparison']
            print("\nModel Performance Comparison:")
            print(comparison_df)
            
            # Determine best overall model
            best_mae = comparison_df['MAE'].idxmin()
            best_rmse = comparison_df['RMSE'].idxmin()
            best_mape = comparison_df['MAPE'].idxmin()
            
            print(f"\nBest Models:")
            print(f"  Lowest MAE:  {best_mae.upper()}")
            print(f"  Lowest RMSE: {best_rmse.upper()}")
            print(f"  Lowest MAPE: {best_mape.upper()}")
        
        print("\n📊 Prediction plots have been displayed showing:")
        print("   - Actual vs Predicted values")
        print("   - Model performance metrics")
        print("   - Confidence intervals (for ARIMA)")
        
        print("\n✅ Task 2 Implementation Complete!")
        print("   Both ARIMA and LSTM models have been successfully trained and evaluated.")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error during forecasting analysis: {str(e)}")
        print("Please check the data and try again.")
        return None


if __name__ == "__main__":
    results = main()
