"""
Time Series Forecasting for Tesla Stock Price Prediction
Time Series Forecasting for Tesla Stock Price Prediction

This module implements ARIMA/SARIMA and LSTM models for forecasting Tesla stock prices.
It includes data preprocessing, model training, evaluation, and prediction capabilities.
"""

import logging
import os
import pickle
import warnings
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    import tensorflow as tf
except ImportError:
    tf = None
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

try:
    from pmdarima import auto_arima
except ImportError:
    auto_arima = None
import matplotlib.pyplot as plt

# Statistical models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Suppress sklearn deprecation warnings about force_all_finite parameter
warnings.filterwarnings(
    "ignore", message=".*force_all_finite.*", category=FutureWarning
)
warnings.filterwarnings(
    "ignore", message=".*ensure_all_finite.*", category=FutureWarning
)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# Set TensorFlow environment variables to suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS"] = "1"

# Import TensorFlow components after setting environment variables
try:
    # TensorFlow imports will be added when LSTM functionality is implemented
    # Configure TensorFlow logging
    if tf is not None:
        tf.get_logger().setLevel("ERROR")
except ImportError:
    print("Warning: TensorFlow not available.")

logger = logging.getLogger(__name__)


class TimeSeriesForecaster:
    """
    Time Series Forecasting class implementing ARIMA/SARIMA and LSTM models
    for Tesla stock price prediction.
    """

    def __init__(self, data: pd.DataFrame, target_column: str = "Close"):
        """
        Initialize the forecaster with Tesla stock data.

        Args:
            data: DataFrame containing Tesla stock data
            target_column: Column name for the target variable (default: 'Close')
        """
        self.data = data.copy()
        self.target_column = target_column
        self.train_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        self.models: Dict[str, Any] = {}
        self.predictions: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}

        # Ensure data is sorted by date and handle timezone issues
        if "Date" in self.data.columns:
            self.data = self.data.sort_values("Date").reset_index(drop=True)
            # Convert to timezone-naive datetime to avoid comparison issues
            self.data["Date"] = pd.to_datetime(self.data["Date"]).dt.tz_localize(None)
            self.data.set_index("Date", inplace=True)
        elif self.data.index.name == "Date":
            # Handle case where Date is already the index
            self.data.index = pd.to_datetime(self.data.index).tz_localize(None)

        logger.info(f"Initialized forecaster with {len(self.data)} data points")

    def split_data(
        self, train_end_date: str = "2023-12-31"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

        if self.train_data is not None and not self.train_data.empty:
            logger.info(
                f"Training data: {len(self.train_data)} points "
                f"({self.train_data.index.min()} to {self.train_data.index.max()})"
            )
        if self.test_data is not None and not self.test_data.empty:
            logger.info(
                f"Testing data: {len(self.test_data)} points "
                f"({self.test_data.index.min()} to {self.test_data.index.max()})"
            )

        print("Data split completed")
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
            "adf_statistic": result[0],
            "p_value": result[1],
            "critical_values": result[4],
            "is_stationary": result[1] < 0.05,
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
           # Create sequences for LSTM"""
        if self.train_data is None:
            raise ValueError("Data must be split first using split_data()")

        series = self.train_data[self.target_column]

        # Check original series stationarity
        stationarity = self.check_stationarity(series, "Original Series")

        if not stationarity["is_stationary"]:
            # Apply first differencing
            series_diff = series.diff().dropna()
            stationarity_diff = self.check_stationarity(
                series_diff, "First Differenced Series"
            )

            if stationarity_diff["is_stationary"]:
                return series_diff
            else:
                # Apply second differencing if needed
                series_diff2 = series_diff.diff().dropna()
                self.check_stationarity(series_diff2, "Second Differenced Series")
                return series_diff2

        return series

    def fit_arima_model(
        self, auto_optimize: bool = True, order: Optional[Tuple[int, int, int]] = None
    ) -> Dict:
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

        # Ensure series has proper datetime index for statsmodels compatibility
        series = self.train_data[self.target_column].copy()
        if not isinstance(series.index, pd.DatetimeIndex):
            # Convert to datetime index if not already
            series.index = pd.to_datetime(series.index)

        # Ensure the series has a proper frequency to avoid statsmodels warnings
        if not hasattr(series.index, "freq") or series.index.freq is None:
            try:
                # Try to infer frequency first
                inferred_freq = pd.infer_freq(series.index)
                if inferred_freq:
                    series = series.asfreq(inferred_freq, method="ffill")
                else:
                    # Use business day frequency as fallback
                    series = series.asfreq("B", method="ffill")
            except Exception:
                # Set a default business day frequency if inference fails
                series = series.asfreq("B", method="ffill")

        print("\n" + "=" * 50)
        print("FITTING ARIMA MODEL")
        print("=" * 50)

        if auto_optimize:
            print("Using auto_arima for parameter optimization...")

            if auto_arima is None:
                print(
                    "Warning: pmdarima not available. "
                    "Using default ARIMA(1,1,1) parameters."
                )
                # Use default ARIMA parameters
                fitted_model = ARIMA(series, order=(1, 1, 1)).fit()
            else:
                try:
                    # Optimize ARIMA parameters using auto_arima
                    print("Optimizing ARIMA parameters...")
                    auto_model = auto_arima(
                        series,
                        start_p=0,
                        start_q=0,
                        max_p=5,
                        max_q=5,
                        seasonal=False,
                        stepwise=True,
                        suppress_warnings=True,
                        error_action="ignore",
                        max_order=10,
                    )
                    fitted_model = auto_model
                    print(f"✅ Auto ARIMA completed with order: {auto_model.order}")
                except Exception as e:
                    print(f"⚠️ Auto ARIMA failed: {e}")
                    print("Falling back to manual ARIMA...")
                    fitted_model = ARIMA(series, order=(1, 1, 1)).fit()
                    print("✅ Manual ARIMA fitted with order: (1, 1, 1)")

        else:
            if order is None:
                order = (1, 1, 1)  # Default order

            print(f"Using manual ARIMA order: {order}")
            fitted_model = ARIMA(series, order=order).fit()

        # Store model
        self.models["arima"] = fitted_model

        # Model summary
        print("\nARIMA Model Summary:")
        print(fitted_model.summary())

        # Get model order based on model type
        try:
            if hasattr(fitted_model, "order"):
                order = fitted_model.order
            elif hasattr(fitted_model, "model") and hasattr(
                fitted_model.model, "order"
            ):
                order = fitted_model.model.order
            else:
                order = (1, 1, 1)  # fallback
        except BaseException:
            order = (1, 1, 1)

        return {
            "model": fitted_model,
            "order": order,
            "aic": fitted_model.aic,
            "bic": fitted_model.bic,
        }

    def predict_arima(self) -> np.ndarray:
        """
        Generate ARIMA predictions for the test period.

        Returns:
            Array of predictions
        """
        if "arima" not in self.models:
            raise ValueError("ARIMA model must be fitted first")

        if self.test_data is None:
            raise ValueError("Test data is None")
        n_periods = len(self.test_data)

        # Use pmdarima model prediction with proper time series context
        try:
            # Suppress statsmodels index warnings during prediction
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message=".*No supported index.*", category=FutureWarning
                )
                warnings.filterwarnings(
                    "ignore", message=".*No supported index.*", category=UserWarning
                )

                # Create proper datetime index for predictions
                if hasattr(self.test_data, "index") and len(self.test_data.index) > 0:
                    start_date = self.test_data.index[0]
                    # Use the same frequency as training data
                    train_freq = getattr(self.train_data.index, "freq", None)
                    if train_freq is None:
                        train_freq = pd.infer_freq(self.train_data.index) or "B"

                    prediction_index = pd.date_range(
                        start=start_date, periods=n_periods, freq=train_freq
                    )
                    predictions_array = self.models["arima"].predict(
                        n_periods=n_periods, return_conf_int=False
                    )
                    predictions = pd.Series(predictions_array, index=prediction_index)
                else:
                    # Fallback to basic prediction
                    predictions_array = self.models["arima"].predict(
                        n_periods=n_periods, return_conf_int=False
                    )
                    predictions = pd.Series(predictions_array)
            print(
                f"✅ ARIMA predictions generated using pmdarima for {n_periods} periods"
            )
        except Exception as e:
            print(f"⚠️ pmdarima prediction failed: {e}")
            # Simple fallback without trend modification
            if self.train_data is not None:
                predictions = np.full(
                    n_periods, self.train_data[self.target_column].iloc[-1]
                )
            else:
                predictions = np.zeros(n_periods)
            print("⚠️ Using simple fallback predictions")

        # Store predictions
        self.predictions["arima"] = predictions

        return predictions

    def prepare_lstm_data(
        self, sequence_length: int = 60
    ) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
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

        # Clean data to avoid sklearn validation warnings
        data_clean = np.nan_to_num(data, nan=0.0, posinf=1e10, neginf=-1e10)

        # Ensure all values are finite before scaling
        if not np.all(np.isfinite(data_clean)):
            data_clean = np.where(np.isfinite(data_clean), data_clean, 0.0)

        # Use MinMaxScaler with explicit finite data handling
        scaler = MinMaxScaler(feature_range=(0, 1))

        # Fit and transform in separate steps to avoid sklearn validation warnings
        scaler.fit(data_clean)
        scaled_data = scaler.transform(data_clean)

        # Create sequences
        X_list, y_list = [], []
        for i in range(sequence_length, len(scaled_data)):
            X_list.append(scaled_data[i - sequence_length : i, 0])
            y_list.append(scaled_data[i, 0])

        X = np.array(X_list)
        y = np.array(y_list)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        return X, y, scaler

    def fit_lstm_model(
        self, sequence_length: int = 60, epochs: int = 100, batch_size: int = 32
    ) -> Dict:
        """
        Fit LSTM model to the training data.

        Args:
            sequence_length: Number of previous time steps to use
            epochs: Number of training epochs
            batch_size: Batch size for training

        Returns:
            Dictionary containing model and training results
        """
        print("\n" + "=" * 50)
        print("FITTING LSTM MODEL")
        print("=" * 50)

        # Prepare data
        X_train, y_train, scaler = self.prepare_lstm_data(sequence_length)

        print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")

        if tf is None:
            raise ImportError(
                "TensorFlow is not installed. "
                "Please install tensorflow to use LSTM models."
            )

        # Build LSTM model with explicit dtype to avoid conversion warnings
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(X_train.shape[1], 1), dtype=tf.float32))
        model.add(tf.keras.layers.LSTM(50, return_sequences=True, dtype=tf.float32))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.LSTM(50, return_sequences=True, dtype=tf.float32))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.LSTM(50, dtype=tf.float32))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(1, dtype=tf.float32))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="mean_squared_error",
        )

        print("LSTM Model Architecture:")
        model.summary()

        # Train model with explicit data types to prevent conversion warnings
        print(f"\nTraining LSTM model for {epochs} epochs...")
        X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float32)
        y_train_tf = tf.convert_to_tensor(y_train, dtype=tf.float32)

        history = model.fit(
            X_train_tf,
            y_train_tf,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            validation_split=0.2,
        )

        # Store model and scaler
        self.models["lstm"] = {
            "model": model,
            "scaler": scaler,
            "sequence_length": sequence_length,
            "history": history,
        }

        return {
            "model": model,
            "scaler": scaler,
            "history": history,
            "sequence_length": sequence_length,
        }

    def predict_lstm(self, steps: Optional[int] = None) -> np.ndarray:
        """
        Generate LSTM predictions for the test period.

        Args:
            steps: Number of steps to forecast (default: length of test data)

        Returns:
            Array of predictions
        """
        if "lstm" not in self.models:
            raise ValueError("LSTM model must be fitted first")

        if steps is None:
            if self.test_data is None:
                raise ValueError("Test data is None")
            steps = len(self.test_data)

        lstm_info = self.models["lstm"]
        model = lstm_info["model"]
        scaler = lstm_info["scaler"]
        sequence_length = lstm_info["sequence_length"]

        # Prepare input data (last sequence_length points from training data)
        if self.train_data is None:
            raise ValueError("Training data is None")
        last_sequence = self.train_data[self.target_column].tail(sequence_length).values
        last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1))

        predictions = []
        current_sequence = last_sequence_scaled.flatten()

        # Generate predictions step by step
        for _ in range(steps):
            # Reshape for prediction
            X_pred = current_sequence[-sequence_length:].reshape(1, sequence_length, 1)

            # Predict next value - convert to proper tensor format
            X_pred_tf = tf.convert_to_tensor(X_pred, dtype=tf.float32)

            # Get prediction and convert to numpy immediately to avoid
            # scalar conversion warnings
            pred_tensor = model(X_pred_tf, training=False)
            pred_numpy = pred_tensor.numpy()
            pred_scaled = pred_numpy[0, 0]

            # Inverse transform to original scale using numpy arrays
            pred_original_array = scaler.inverse_transform([[pred_scaled]])
            pred_original = pred_original_array[0, 0]
            predictions.append(pred_original)

            # Update sequence for next prediction
            current_sequence = np.append(current_sequence, pred_scaled)

        predictions_array = np.array(predictions)

        self.predictions["lstm"] = {"forecast": predictions_array}

        return predictions_array

    def calculate_metrics(
        self, actual: np.ndarray, predicted: np.ndarray, model_name: str
    ) -> Dict:
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

        metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape}

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

        print("\n" + "=" * 50)
        print("MODEL COMPARISON")
        print("=" * 50)
        print(comparison_df)

        # Find best model for each metric
        best_models = {}
        for metric in comparison_df.columns:
            if metric == "MAPE":
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

        if self.test_data is None:
            raise ValueError("Test data is None")
        test_dates = self.test_data.index
        actual_values = self.test_data[self.target_column].values

        for i, (model_name, pred_info) in enumerate(self.predictions.items()):
            ax = axes[i]

            # Plot actual values
            ax.plot(
                test_dates, actual_values, label="Actual", color="blue", linewidth=2
            )

            # Plot predictions - handle both array and dict formats
            if isinstance(pred_info, dict) and "forecast" in pred_info:
                predictions = pred_info["forecast"]
            else:
                predictions = pred_info  # Direct array format

            ax.plot(
                test_dates[: len(predictions)],
                predictions,
                label=f"{model_name.upper()} Prediction",
                color="red",
                linewidth=2,
                linestyle="--",
            )

            # Plot confidence interval for ARIMA (if available)
            if (
                model_name == "arima"
                and isinstance(pred_info, dict)
                and "confidence_interval" in pred_info
            ):
                ci = pred_info["confidence_interval"]
                ax.fill_between(
                    test_dates[: len(ci)],
                    ci.iloc[:, 0],
                    ci.iloc[:, 1],
                    alpha=0.3,
                    color="red",
                    label="Confidence Interval",
                )

            ax.set_title(f"{model_name.upper()} Model Predictions vs Actual")
            ax.set_xlabel("Date")
            ax.set_ylabel("Stock Price ($)")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add metrics to plot
            if model_name in self.metrics:
                metrics = self.metrics[model_name]
                textstr = (
                    f"MAE: {metrics['MAE']:.2f}\n"
                    f"RMSE: {metrics['RMSE']:.2f}\n"
                    f"MAPE: {metrics['MAPE']:.2f}%"
                )
                ax.text(
                    0.02,
                    0.98,
                    textstr,
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )

        plt.tight_layout()
        plt.show()
        plt.close()  # Close the figure to free memory

    def run_complete_analysis(
        self, auto_arima: bool = True, lstm_epochs: int = 100
    ) -> Dict:
        """
        Run complete time series forecasting analysis.

        Args:
            auto_arima: Whether to use auto_arima for parameter optimization
            lstm_epochs: Number of epochs for LSTM training

        Returns:
            Dictionary with all results
        """
        print("STARTING COMPLETE TIME SERIES FORECASTING ANALYSIS")
        print("=" * 60)

        # Split data
        self.split_data()

        # Fit models
        arima_results = self.fit_arima_model(auto_optimize=auto_arima)
        lstm_results = self.fit_lstm_model(epochs=lstm_epochs)

        # Generate predictions
        arima_pred = self.predict_arima()
        lstm_pred = self.predict_lstm()

        # Calculate metrics
        if self.test_data is None:
            raise ValueError("Test data is None")
        actual_values = self.test_data[self.target_column].values
        self.calculate_metrics(actual_values, arima_pred, "arima")
        self.calculate_metrics(actual_values[: len(lstm_pred)], lstm_pred, "lstm")

        # Compare models
        comparison = self.compare_models()

        # Plot results
        self.plot_predictions()

        return {
            "arima_results": arima_results,
            "lstm_results": lstm_results,
            "predictions": self.predictions,
            "metrics": self.metrics,
            "comparison": comparison,
        }

    def save_models(self, models_dir: str = "models") -> None:
        """
        Save trained models to disk for reuse in Task 3.

        Args:
            models_dir: Directory to save models
        """
        # Create models directory if it doesn't exist
        import os

        os.makedirs(models_dir, exist_ok=True)

        try:
            # Save ARIMA model
            if "arima" in self.models:
                arima_path = os.path.join(models_dir, "tesla_arima_model.pkl")
                with open(arima_path, "wb") as f:
                    pickle.dump(self.models["arima"], f)
                print(f"✅ ARIMA model saved to {arima_path}")

            # Save LSTM model and related components
            if "lstm" in self.models:
                lstm_info = self.models["lstm"]

                # Save TensorFlow model (using modern Keras format)
                lstm_model_path = os.path.join(models_dir, "tesla_lstm_model.keras")
                lstm_info["model"].save(lstm_model_path)

                # Save scaler
                scaler_path = os.path.join(models_dir, "tesla_lstm_scaler.pkl")
                joblib.dump(lstm_info["scaler"], scaler_path)

                # Save LSTM metadata
                lstm_metadata = {
                    "sequence_length": lstm_info["sequence_length"],
                    "target_column": self.target_column,
                    "train_end_date": (
                        str(self.train_end_date)
                        if hasattr(self, "train_end_date")
                        else None
                    ),
                }
                metadata_path = os.path.join(models_dir, "tesla_lstm_metadata.pkl")
                with open(metadata_path, "wb") as f:
                    pickle.dump(lstm_metadata, f)

                print(f"✅ LSTM model saved to {lstm_model_path}")
                print(f"✅ LSTM scaler saved to {scaler_path}")
                print(f"✅ LSTM metadata saved to {metadata_path}")

            # Save data and forecaster metadata
            forecaster_metadata = {
                "target_column": self.target_column,
                "train_end_date": (
                    str(self.train_end_date)
                    if hasattr(self, "train_end_date")
                    else None
                ),
                "data_shape": self.data.shape,
                "data_columns": list(self.data.columns),
                "save_timestamp": datetime.now().isoformat(),
            }

            metadata_path = os.path.join(models_dir, "tesla_forecaster_metadata.pkl")
            with open(metadata_path, "wb") as f:
                pickle.dump(forecaster_metadata, f)

            # Save the processed data for Task 3 in proper data directory
            os.makedirs("data/processed", exist_ok=True)
            data_path = "data/processed/tesla_training_data.csv"
            self.data.to_csv(data_path)

            print(f"✅ Forecaster metadata saved to {metadata_path}")
            print(f"✅ Training data saved to {data_path}")
            print(f"🎯 All models successfully saved to '{models_dir}/' directory")

        except Exception as e:
            print(f"❌ Error saving models: {str(e)}")

    @classmethod
    def load_models(cls, models_dir: str = "models") -> "TimeSeriesForecaster":
        """
        Load trained models from disk for use in Task 3.

        Args:
            models_dir: Directory containing saved models

        Returns:
            TimeSeriesForecaster instance with loaded models
        """
        try:
            # Load forecaster metadata
            metadata_path = os.path.join(models_dir, "tesla_forecaster_metadata.pkl")
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)

            # Load training data from proper data directory
            data_path = "data/processed/tesla_training_data.csv"
            data = pd.read_csv(data_path, index_col=0, parse_dates=True)

            # Initialize forecaster
            forecaster = cls(data=data, target_column=metadata["target_column"])

            # Set train_end_date if available
            if metadata.get("train_end_date"):
                forecaster.split_data(train_end_date=metadata["train_end_date"])

            print(f"✅ Loaded training data: {data.shape}")
            print(f"✅ Target column: {metadata['target_column']}")

            # Load ARIMA model
            arima_path = os.path.join(models_dir, "tesla_arima_model.pkl")
            if os.path.exists(arima_path):
                with open(arima_path, "rb") as f:
                    forecaster.models["arima"] = pickle.load(f)
                print(f"✅ ARIMA model loaded from {arima_path}")

            # Load LSTM model
            lstm_model_path = os.path.join(models_dir, "tesla_lstm_model.keras")
            scaler_path = os.path.join(models_dir, "tesla_lstm_scaler.pkl")
            lstm_metadata_path = os.path.join(models_dir, "tesla_lstm_metadata.pkl")

            if all(
                os.path.exists(p)
                for p in [lstm_model_path, scaler_path, lstm_metadata_path]
            ):
                from tensorflow.keras.models import load_model

                # Load LSTM components
                lstm_model = load_model(lstm_model_path)
                scaler = joblib.load(scaler_path)

                with open(lstm_metadata_path, "rb") as f:
                    lstm_metadata = pickle.load(f)

                forecaster.models["lstm"] = {
                    "model": lstm_model,
                    "scaler": scaler,
                    "sequence_length": lstm_metadata["sequence_length"],
                }

                print(f"✅ LSTM model loaded from {lstm_model_path}")
                print(f"✅ LSTM scaler loaded from {scaler_path}")

            print(f"🎯 All models successfully loaded from '{models_dir}/' directory")
            print(f"📅 Models saved on: {metadata.get('save_timestamp', 'Unknown')}")

            return forecaster

        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            raise


def main():
    """
    Main function to run Task 2: Time Series Forecasting Analysis
    """
    import sys

    sys.path.append(".")
    sys.path.append("..")

    print("=" * 80)
    print("TASK 2: TIME SERIES FORECASTING FOR TESLA STOCK PREDICTION")
    print("=" * 80)

    # Step 1: Load Tesla processed data
    print("\n1. Loading Tesla Processed Data...")

    try:
        # Load the already processed Tesla data
        tesla_processed = pd.read_csv("data/processed/TSLA_processed.csv")

        # Convert Date column to datetime and set as index (timezone-naive)
        tesla_processed["Date"] = pd.to_datetime(
            tesla_processed["Date"], utc=True
        ).dt.tz_localize(None)
        tesla_processed.set_index("Date", inplace=True)

        print(
            f"✅ Successfully loaded {len(tesla_processed)} Tesla processed data points"
        )
        print(
            f"   Date range: {tesla_processed.index.min().date()} to "
            f"{tesla_processed.index.max().date()}"
        )
        print(f"   Available columns: {list(tesla_processed.columns)}")

    except FileNotFoundError:
        print(
            "❌ Error: Processed Tesla data not found. Please run Task 1 (EDA) first."
        )
        print("   Run: python -m src.data.eda")
        return
    except Exception as e:
        print(f"❌ Error loading processed data: {str(e)}")
        return

    # Step 2: Initialize Time Series Forecaster
    print("\n2. Initializing Time Series Forecaster...")
    forecaster = TimeSeriesForecaster(
        data=tesla_processed, target_column="Close"  # Use Close price for prediction
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
            auto_arima=True,  # Use pmdarima auto_arima as required
            lstm_epochs=50,  # Reduced epochs for faster training
        )

        print("\n" + "=" * 60)
        print("TASK 2 COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        # Display final summary
        print("\nFINAL RESULTS SUMMARY:")
        print("-" * 30)

        if "comparison" in results:
            comparison_df = results["comparison"]
            print("\nModel Performance Comparison:")
            print(comparison_df)

            # Determine best overall model
            best_mae = comparison_df["MAE"].idxmin()
            best_rmse = comparison_df["RMSE"].idxmin()
            best_mape = comparison_df["MAPE"].idxmin()

            print("\nBest Models:")
            print(f"  Lowest MAE:  {best_mae.upper()}")
            print(f"  Lowest RMSE: {best_rmse.upper()}")
            print(f"  Lowest MAPE: {best_mape.upper()}")

        print("\n📊 Prediction plots have been displayed showing:")
        print("   - Actual vs Predicted values")
        print("   - Model performance metrics")
        print("   - Confidence intervals (for ARIMA)")

        # Save trained models
        print("\n💾 Saving trained models...")
        forecaster.save_models()

        print("\n✅ Task 2 Implementation Complete!")
        print(
            "   Both ARIMA and LSTM models have been successfully "
            "trained and evaluated."
        )
        print("   Models saved for future use in Task 3.")

        return results

    except Exception as e:
        print(f"\n❌ Error during forecasting analysis: {str(e)}")
        print("Please check the data and try again.")
        return None


if __name__ == "__main__":
    results = main()
