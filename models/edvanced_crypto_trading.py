import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import keras
from keras import layers
from typing import List, Tuple, Dict
import traceback # Import traceback for better error handling

# --- Mock Classes (replace with your actual implementations) ---
# Assuming these classes are defined elsewhere as provided in previous interactions.
# I'm re-including them here for completeness so the example_usage can run.

class MockDaysLevel:
    def __init__(self, data):
        self.data = data # Store the data in the instance
        self.timeframe_weights = {
            '1H': 1, '4H': 2, '1D': 3, '1W': 4, '1M': 5
        }

    def _find_extremes(self, timeframe: str, kind: str) -> List[Tuple]:
        np.random.seed(hash((timeframe, kind)) % 2**32)
        if self.data.empty: # Handle empty data case
            return []
        current_price = self.data['close'].iloc[-1]
        return [
            (self.data.index[-1],
             current_price * (1 + np.random.normal(0, 0.05)),
             np.random.uniform(1, 10),
             timeframe,
             kind if kind != 'mirror' else 'mirror', # ensure 'mirror' type is correct
             np.random.choice(['up', 'down', 'neutral']))
            for _ in range(3) # Reduce number of mock levels for performance
        ]

    def level_cinching(self, timeframe: str) -> List[Tuple]:
        return self._find_extremes(timeframe, kind='cinching')

    def level_mirror(self, timeframe: str) -> List[Tuple]:
        return self._find_extremes(timeframe, kind='mirror')

    def level_change(self, timeframe: str) -> List[Tuple]:
        return self._find_extremes(timeframe, kind='change')

    def level_paranorm(self, timeframe: str) -> List[Tuple]:
        return self._find_extremes(timeframe, kind='paranorm')

    def get_all_levels(self, timeframes: List[str] = None) -> List[Tuple]:
        if timeframes is None:
            timeframes = ['1H', '4H', '1D', '1W']
        all_levels = []
        for tf in timeframes:
            try:
                for method in [self.level_cinching, self.level_mirror, self.level_change, self.level_paranorm]:
                    all_levels.extend(method(tf))
            except Exception as e:
                # print(f"⚠️ Ошибка в {tf}: {e}") # Suppress for cleaner output during mocks
                continue
        return all_levels

    def filter_close_levels(self, levels: List[Tuple], threshold: float) -> List[Tuple]:
        filtered = []
        # Sort by: 1. Is it '1W'? (False for '1W' puts them first), 2. Strength (descending)
        for lvl in sorted(levels, key=lambda x: (x[3] != '1W', -x[2])):
            # Check if this level is too close to any already added filtered level
            if all(abs(lvl[1] - r[1]) > threshold for r in filtered):
                filtered.append(lvl)
        return filtered

    def get_strongest_levels(self, timeframes: List[str] = None, top_n: int = 12) -> List[Tuple]:
        all_levels = self.get_all_levels(timeframes)
        # Ensure 'atri' exists in data and is not NaN for threshold calculation
        atri = self.data['atri'].iloc[-1] if 'atri' in self.data.columns and not self.data['atri'].empty and not pd.isna(self.data['atri'].iloc[-1]) else 0.01
        threshold = 0.25 * atri # Using a multiple of ATRI for threshold
        filtered_levels = self.filter_close_levels(all_levels, threshold)
        return sorted(filtered_levels, key=lambda x: x[2], reverse=True)[:top_n]

class TradingSignalGenerator:
    def calculate_entry_exit_points(self, current_price, predicted_price, daily_atri, level_info):
        direction = "LONG" if predicted_price > current_price else "SHORT"
        entry_price = current_price
        stop_loss = entry_price * (1 - 0.01) if direction == "LONG" else entry_price * (1 + 0.01)
        risk = abs(entry_price - stop_loss)

        take_profits = {
            'TP1': {'price': predicted_price * 1.01, 'ratio': 2.0, 'distance_atri': 1.0, 'probability': 0.7},
            'TP2': {'price': predicted_price * 1.02, 'ratio': 3.0, 'distance_atri': 2.0, 'probability': 0.6},
        }

        hot_deals = []
        if direction == "LONG" and predicted_price > current_price * 1.03:
            hot_deals.append({
                'tp_level': 'TP_Hot', 'rr_ratio': 3.5, 'probability': 0.85, 'profit_potential': (predicted_price * 1.03 - entry_price)
            })

        return {
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'risk': risk,
            'take_profits': take_profits,
            'hot_deals': hot_deals
        }

    def rank_signals(self, signals):
        for signal in signals:
            signal['score'] = signal['confidence'] * (1 + abs(signal['predicted_change_percent']) / 100)
        return sorted(signals, key=lambda x: x['score'], reverse=True)


class CriptoPredictionNN:
    """
    Улучшенная нейросеть для предсказания цены с системой торговых сигналов
    """

    def __init__(self, data, days_level, lookback_period=20, prediction_horizon=5,
                 prediction_type='price', model_save_dir='models'):
        self.days_level = days_level
        self.lookback_period = lookback_period
        self.prediction_horizon = prediction_horizon
        self.prediction_type = prediction_type
        self.is_trained = False
        self.model_save_dir = model_save_dir
        self.data = data # Store the data in the instance

        os.makedirs(model_save_dir, exist_ok=True)

        self.price_scaler = MinMaxScaler()
        self.feature_scaler = StandardScaler()

        self.model = None
        self.training_history = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.signal_generator = TradingSignalGenerator()

    def _calculate_distance_to_levels(self, current_price: float, levels: List[Tuple]) -> Dict:
        if not levels:
            return {
                'nearest_support_dist': 0.01, 'nearest_resistance_dist': 0.01,
                'nearest_support_strength': 5, 'nearest_resistance_strength': 5,
                'nearest_mirror_dist': 0.02, 'nearest_mirror_strength': 6,
                'total_levels_above': 3, 'total_levels_below': 2,
                'weighted_support_strength': 15, 'weighted_resistance_strength': 12,
            }

        supports = []
        resistances = []
        mirrors = []

        for level_data in levels:
            _, level, strength, timeframe, level_type, direction = level_data

            if level_type == 'mirror':
                mirrors.append((level, strength))
            elif level < current_price: # Assuming supports are below current price
                supports.append((level, strength))
            else: # Assuming resistances are above current price
                resistances.append((level, strength))

        nearest_support = max(supports, key=lambda x: x[0]) if supports else (0, 0)
        nearest_resistance = min(resistances, key=lambda x: x[0]) if resistances else (999999, 0)
        nearest_mirror = min(mirrors, key=lambda x: abs(x[0] - current_price)) if mirrors else (0, 0)

        support_dist = (current_price - nearest_support[0]) / current_price if nearest_support[0] > 0 else 0.05
        resistance_dist = (nearest_resistance[0] - current_price) / current_price if nearest_resistance[0] > 0 else 0.05
        mirror_dist = abs(nearest_mirror[0] - current_price) / current_price if nearest_mirror[0] > 0 else 0.02

        weighted_support = sum(s * (1 / (1 + abs(l - current_price))) for l, s in supports) if supports else 0
        weighted_resistance = sum(s * (1 / (1 + abs(l - current_price))) for l, s in resistances) if resistances else 0

        return {
            'nearest_support_dist': support_dist, 'nearest_resistance_dist': resistance_dist,
            'nearest_support_strength': nearest_support[1], 'nearest_resistance_strength': nearest_resistance[1],
            'nearest_mirror_dist': mirror_dist, 'nearest_mirror_strength': nearest_mirror[1],
            'total_levels_above': len(resistances), 'total_levels_below': len(supports),
            'weighted_support_strength': weighted_support, 'weighted_resistance_strength': weighted_resistance
        }

    def _calculate_atri(self, data_for_atri: pd.DataFrame, window: int = 14):
        high = data_for_atri['high']
        low = data_for_atri['low']
        close = data_for_atri['close']

        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        tr4 = np.abs(close - close.shift(1))
        tr5 = np.abs(high - low.shift(1))

        tr = pd.concat([tr1, tr2, tr3, tr4, tr5], axis=1).max(axis=1)
        atri_raw = tr.rolling(window=window).mean()
        atri_mean = atri_raw.mean()
        atri_filtered = atri_raw.where(
            (atri_raw < 1.5 * atri_mean) & (atri_raw > 0.6 * atri_mean)
        )
        return atri_filtered
    
    def _calculate_volatility_features(self, data: pd.DataFrame, window: int = 10) -> Dict:
        if len(data) < window:
            return {
                'volatility_ratio': 0.03, 'is_accumulation': False,
                'price_range_ratio': 0.05, 'volume_trend': 0.01
            }

        # Ensure 'atri' column is present for calculation or handle its absence
        # It's better to ensure ATRI is pre-calculated on the full dataset
        if 'atri' not in data.columns:
            # Fallback or error if ATRI is expected but not present
            # print("Warning: ATRI column not found in data for volatility features.")
            # For this example, let's re-calculate if missing (though not ideal for performance)
            temp_atri = self._calculate_atri(data, window=window)
            data = data.copy() # Avoid SettingWithCopyWarning
            data['atri'] = temp_atri
            if data['atri'].isnull().all(): # If ATRI is still all nulls, provide dummy values
                 return {
                    'volatility_ratio': 0.03, 'is_accumulation': False,
                    'price_range_ratio': 0.05, 'volume_trend': 0.01
                }


        avg_atri = data['atri'].rolling(window=window).mean().iloc[-1] if not data['atri'].empty else 0
        current_volatility = data['atri'].iloc[-1] if not data['atri'].empty else 0

        volatility_ratio = current_volatility / avg_atri if avg_atri > 0 else 0

        is_accumulation = volatility_ratio < 0.5

        price_range = (data['high'].iloc[-1] - data['low'].iloc[-1]) / data['close'].iloc[-1]
        avg_price_range = ((data['high'] - data['low']) / data['close']).rolling(window=window).mean().iloc[-1]
        price_range_ratio = price_range / avg_price_range if avg_price_range > 0 else 0

        volume_trend = 0
        if 'volume' in data.columns and len(data) >= 2*window:
            recent_volume = data['volume'].iloc[-window:].mean()
            prev_volume = data['volume'].iloc[-2*window:-window].mean()
            volume_trend = (recent_volume - prev_volume) / prev_volume if prev_volume > 0 else 0

        return {
            'volatility_ratio': volatility_ratio,
            'is_accumulation': is_accumulation,
            'price_range_ratio': price_range_ratio,
            'volume_trend': volume_trend
        }

    def _create_features(self, data: pd.DataFrame, index: int) -> np.ndarray:
        if index < self.lookback_period:
            return None

        # Ensure 'close', 'high', 'low', 'volume', 'atri' are available and not NaN in the slice
        required_cols = ['close', 'high', 'low', 'volume', 'atri']
        data_slice = data.iloc[max(0, index - self.lookback_period):index+1]

        for col in required_cols:
            if col not in data_slice.columns or data_slice[col].isnull().any():
                # print(f"Warning: Missing or NaN values in '{col}' for feature creation at index {index}. Skipping.")
                return None # Skip this feature creation if data is incomplete

        historical_prices = data_slice['close'].iloc[-self.lookback_period:-1].values # Correct slicing
        if len(historical_prices) < 2:
            return None # Not enough data for returns
        price_returns = np.diff(historical_prices) / historical_prices[:-1]

        current_price = data_slice['close'].iloc[-1]

        strongest_levels = self.days_level.get_strongest_levels(timeframes=['1H', '4H', '1D'], top_n=20)
        level_features = self._calculate_distance_to_levels(current_price, strongest_levels)

        volatility_features = self._calculate_volatility_features(data_slice)

        rsi = self._calculate_rsi(historical_prices)
        macd, macd_signal = self._calculate_macd(historical_prices)

        # Ensure all features are numerical and not NaN/Inf
        # It's good practice to ensure all values are finite before concatenating
        features_list = [
            price_returns.flatten(),
            [np.mean(price_returns), np.std(price_returns), np.min(price_returns), np.max(price_returns)],
            list(level_features.values()),
            list(volatility_features.values()),
            [rsi, macd, macd_signal],
            [current_price / np.mean(historical_prices),
             (current_price - np.min(historical_prices)) / (np.max(historical_prices) - np.min(historical_prices))]
        ]

        # Flatten all lists and convert to numpy array, handling potential NaN/Inf
        flat_features = []
        for item in features_list:
            if isinstance(item, np.ndarray):
                flat_features.extend(item.flatten())
            elif isinstance(item, list):
                flat_features.extend(item)
            else: # Single value
                flat_features.append(item)
        
        features_array = np.array(flat_features, dtype=np.float32)
        if not np.all(np.isfinite(features_array)):
            # print(f"Warning: Non-finite values in features at index {index}. Skipping.")
            return None

        return features_array

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0 # Handle division by zero

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float]:
        if len(prices) < slow:
            return 0.0, 0.0

        ema_fast = self._ema(prices, fast)
        ema_slow = self._ema(prices, slow)

        macd = ema_fast - ema_slow

        if len(prices) >= slow + signal:
            # Need to re-calculate MACD line for EMA of MACD
            macd_line_values = []
            for i in range(slow-1, len(prices)):
                # This is an approximation; true MACD needs EMA of MACD values
                # For simplicity here, we're using a moving average of recent MACD values
                # A more precise implementation would track MACD values over time
                if i - (slow -1) >= fast - 1: # Ensure enough points for fast EMA
                    current_ema_fast = self._ema(prices[max(0, i-fast+1):i+1], fast)
                else:
                    current_ema_fast = np.mean(prices[max(0, i-fast+1):i+1])

                current_ema_slow = self._ema(prices[max(0, i-slow+1):i+1], slow)
                macd_line_values.append(current_ema_fast - current_ema_slow)
            
            if len(macd_line_values) >= signal:
                macd_signal = self._ema(np.array(macd_line_values), signal)
            else:
                macd_signal = np.mean(macd_line_values) if macd_line_values else 0.0
        else:
            macd_signal = 0.0

        return macd, macd_signal

    def _ema(self, prices: np.ndarray, period: int) -> float:
        if len(prices) == 0:
            return 0.0
        if len(prices) < period: # If not enough data for EMA, use simple mean
            return np.mean(prices)

        alpha = 2.0 / (period + 1)
        # Calculate EMA for the entire series and take the last value
        ema_values = np.zeros_like(prices, dtype=float)
        ema_values[0] = prices[0] # First EMA is simply the first price

        for i in range(1, len(prices)):
            ema_values[i] = alpha * prices[i] + (1 - alpha) * ema_values[i-1]

        return ema_values[-1]

    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []

        # Iterate through the data, ensuring enough lookback and prediction horizon
        # The loop range needs to be carefully chosen to avoid index out of bounds
        # and ensure sufficient data for both features and target.
        # `index - self.lookback_period` should be >= 0
        # `index + self.prediction_horizon` should be < len(data)
        
        # Start from the first index where enough lookback data is available
        start_index = self.lookback_period 
        # End before the last index that allows for the prediction horizon
        end_index = len(data) - self.prediction_horizon 

        for i in range(start_index, end_index):
            features = self._create_features(data, i)
            if features is None:
                continue

            current_price = data['close'].iloc[i]
            future_price = data['close'].iloc[i + self.prediction_horizon]

            if self.prediction_type == 'price':
                target = future_price
            else:  # change
                target = (future_price - current_price) / current_price

            X.append(features)
            y.append(target)

        # Handle empty X and y
        if not X:
            print("Warning: No data generated for training after feature creation and filtering.")
            return np.array([]), np.array([])
            
        return np.array(X), np.array(y)

    def create_model(self, input_shape: int) -> keras.Model:
        model = keras.Sequential([
            layers.Input(shape=(input_shape,)),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model

    def train(self, data: pd.DataFrame, epochs: int = 100, batch_size: int = 32,
              validation_split: float = 0.2, verbose: int = 1):
        # Store the training data for later use in evaluate_model if needed
        self.data = data 
        print("Подготовка данных...")
        X, y = self.prepare_data(data)

        if len(X) == 0:
            print("Недостаточно данных для обучения после подготовки.")
            self.is_trained = False
            return None # Indicate training failed

        print(f"Размер обучающей выборки: {X.shape}")
        print(f"Размер целевого вектора: {y.shape}")

        if self.prediction_type == 'price':
            y = y.reshape(-1, 1)
            y_scaled = self.price_scaler.fit_transform(y)
        else:
            y_scaled = y.reshape(-1, 1) # Still reshape for consistency

        X_scaled = self.feature_scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42, shuffle=False
        )

        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

        self.model = self.create_model(X.shape[1])

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=20, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6
            )
        ]

        print("Начало обучения...")
        self.training_history = self.model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            callbacks=callbacks,
            batch_size=batch_size,
            verbose=verbose
        )

        test_loss, test_mae = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nРезультаты на тестовой выборке:")
        print(f"MSE: {test_loss:.6f}")
        print(f"MAE: {test_mae:.6f}")

        y_pred = self.model.predict(X_test, verbose=0)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"RMSE: {rmse:.6f}")

        self.is_trained = True
        self.save_model_auto()
        return self.training_history

    def predict_with_trading_signals(self, data: pd.DataFrame, current_index: int = None) -> Dict:
        if not self.is_trained or self.model is None:
            raise ValueError("Модель не обучена или не инициализирована")

        if current_index is None:
            current_index = len(data) - 1

        features = self._create_features(data, current_index)
        if features is None:
            raise ValueError(f"Недостаточно данных для создания признаков на индексе {current_index}")

        features_scaled = self.feature_scaler.transform(features.reshape(1, -1))
        prediction = self.model.predict(features_scaled, verbose=0)[0][0]
        current_price = data['close'].iloc[current_index]
        
        # Ensure 'atri' column is present and not NaN for daily_atri
        daily_atri = data['atri'].iloc[current_index] if 'atri' in data.columns and not pd.isna(data['atri'].iloc[current_index]) else 0.01

        if self.prediction_type == 'price':
            prediction_scaled = self.price_scaler.inverse_transform([[prediction]])
            predicted_price = prediction_scaled[0][0]
            predicted_change = (predicted_price - current_price) / current_price
        else:
            predicted_change = prediction
            predicted_price = current_price * (1 + predicted_change)

        strongest_levels = self.days_level.get_strongest_levels(timeframes=['1H', '4H', '1D'], top_n=10)
        level_info = self._calculate_distance_to_levels(current_price, strongest_levels)
        volatility_info = self._calculate_volatility_features(data.iloc[max(0, current_index-10):current_index+1])

        trading_signals = self.signal_generator.calculate_entry_exit_points(
            current_price, predicted_price, daily_atri, level_info
        )

        # Pass daily_atri to determine_strategy
        strategy = self._determine_strategy(current_price, predicted_price, level_info, volatility_info, daily_atri)

        return {
            'current_price': current_price,
            'predicted_price': predicted_price,
            'predicted_change': predicted_change,
            'predicted_change_percent': predicted_change * 100,
            'strategy': strategy,
            'level_info': level_info,
            'volatility_info': volatility_info,
            'confidence': self._calculate_confidence(features_scaled),
            'trading_signals': trading_signals,
            'daily_atri': daily_atri
        }

    def _determine_strategy(self, current_price: float, predicted_price: float,
                           level_info: Dict, volatility_info: Dict, daily_atri: float) -> str:
        """
        Определение торговой стратегии на основе предсказания, информации об уровнях и волатильности.
        """
        price_change_abs = abs(predicted_price - current_price)
        price_change_percent = price_change_abs / current_price

        # --- Level Proximity ---
        # Convert absolute distances to ATRI multiples for better context
        nearest_support_dist_abs = level_info['nearest_support_dist'] * current_price
        nearest_resistance_dist_abs = level_info['nearest_resistance_dist'] * current_price
        nearest_mirror_dist_abs = level_info['nearest_mirror_dist'] * current_price

        nearest_support_dist_atri = nearest_support_dist_abs / daily_atri if daily_atri > 0 else float('inf')
        nearest_resistance_dist_atri = nearest_resistance_dist_abs / daily_atri if daily_atri > 0 else float('inf')
        nearest_mirror_dist_atri = nearest_mirror_dist_abs / daily_atri if daily_atri > 0 else float('inf')

        # Define thresholds in terms of ATRI for "nearness"
        NEAR_LEVEL_ATRI_THRESHOLD = 0.5 # e.g., within 0.5 ATRI
        
        near_support = nearest_support_dist_atri < NEAR_LEVEL_ATRI_THRESHOLD and level_info['nearest_support_strength'] > 0
        near_resistance = nearest_resistance_dist_atri < NEAR_LEVEL_ATRI_THRESHOLD and level_info['nearest_resistance_strength'] > 0
        near_mirror = nearest_mirror_dist_atri < NEAR_LEVEL_ATRI_THRESHOLD and level_info['nearest_mirror_strength'] > 0

        # --- Volatility and Consolidation Checks ---
        is_low_volatility = volatility_info['is_accumulation']

        # Simplified check for low recent volume (you'd need more robust logic)
        has_low_recent_volume = volatility_info['volume_trend'] < 0.005 # Example: low positive or negative trend

        # --- Strategy Logic ---

        # Strategy 1: Price targeting nearest strong level (magnet effect)
        # Only if predicted change is relatively small (e.g., within 2 ATRI)
        if price_change_abs < daily_atri * 2 and daily_atri > 0.001:
            if near_support and predicted_price > current_price:
                return "Ожидаем отскок от поддержки к ближайшему уровню"
            if near_resistance and predicted_price < current_price:
                return "Ожидаем отскок от сопротивления к ближайшему уровню"

        # Strategy 2: Breakout from Consolidation (Low Volatility, Dojis, Low Volume)
        BREAKOUT_ATRI_TARGET = 3.5 # Minimum target for a breakout in ATRI multiples

        if is_low_volatility and has_low_recent_volume and daily_atri > 0.001:
            target_level = None
            target_distance_atri = 0

            # Find the next significant weekly level in the predicted direction
            if predicted_price > current_price: # Bullish breakout
                # Get all weekly levels and filter for those above current price
                weekly_levels_above = [lvl for lvl in self.days_level.get_strongest_levels(timeframes=['1W']) if lvl[1] > current_price]
                if weekly_levels_above:
                    target_level_tuple = min(weekly_levels_above, key=lambda x: x[1]) # Nearest weekly resistance above
                    target_level = target_level_tuple[1]
                    target_distance_atri = (target_level - current_price) / daily_atri
            elif predicted_price < current_price: # Bearish breakout
                # Get all weekly levels and filter for those below current price
                weekly_levels_below = [lvl for lvl in self.days_level.get_strongest_levels(timeframes=['1W']) if lvl[1] < current_price]
                if weekly_levels_below:
                    target_level_tuple = max(weekly_levels_below, key=lambda x: x[1]) # Nearest weekly support below
                    target_level = target_level_tuple[1]
                    target_distance_atri = (current_price - target_level) / daily_atri
            
            # Check if a target weekly level was found, it's far enough, and prediction supports reaching it
            if target_level is not None and target_distance_atri >= BREAKOUT_ATRI_TARGET and price_change_abs >= BREAKOUT_ATRI_TARGET * daily_atri:
                if (predicted_price > current_price and predicted_price <= target_level) or \
                   (predicted_price < current_price and predicted_price >= target_level):
                    return f"Прорыв из консолидации к недельному уровню {target_level:.2f} ({target_distance_atri:.1f} ATRI)"


        # --- Existing Strategies (as in your original code) ---
        if near_support and level_info['nearest_support_strength'] > 5:
            if predicted_price > current_price:
                return "Отбой от сильного уровня поддержки - покупка"
            else:
                return "Возможный пробой уровня поддержки - продажа"

        if near_resistance and level_info['nearest_resistance_strength'] > 5:
            if predicted_price < current_price:
                return "Отбой от сильного уровня сопротивления - продажа"
            else:
                return "Возможный пробой уровня сопротивления - покупка"

        if near_mirror and level_info['nearest_mirror_strength'] > 3:
            mirror_distance = level_info['nearest_mirror_dist'] * current_price
            if predicted_price > current_price:
                return f"Отработка зеркального уровня вверх - цель +{mirror_distance:.2f}"
            else:
                return f"Отработка зеркального уровня вниз - цель -{mirror_distance:.2f}"

        if price_change_percent > 0.02 or price_change_percent < -0.02: # Significant movement > 2%
            direction = "покупка" if predicted_price > current_price else "продажа"
            return f"Значительное движение - {direction}"

        return "Боковое движение - ожидание"

    def _calculate_confidence(self, features_scaled: np.ndarray) -> float:
        feature_variance = np.var(features_scaled)
        confidence = 1 / (1 + feature_variance)
        return min(max(confidence, 0.1), 0.9)

    def evaluate_model(self):
        if self.model is None or not self.is_trained:
            print("❌ Модель не обучена!")
            return None

        y_pred = self.model.predict(self.X_test)
        y_test = self.y_test

        if self.prediction_type == 'price':
            y_test_original = self.price_scaler.inverse_transform(y_test)
            y_pred_original = self.price_scaler.inverse_transform(y_pred)
        else:
            y_test_original = y_test
            y_pred_original = y_pred

        mse = mean_squared_error(y_test_original, y_pred_original)
        mae = mean_absolute_error(y_test_original, y_pred_original)
        r2 = r2_score(y_test_original, y_pred_original)

        print("📈 ОЦЕНКА МОДЕЛИ:")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"R²: {r2:.4f}")

        if self.prediction_type == 'change':
            actual_directions = np.sign(y_test_original.flatten())
            pred_directions = np.sign(y_pred_original.flatten())
            direction_accuracy = np.mean(actual_directions == pred_directions)
            print(f"Точность направления: {direction_accuracy:.4f}")

        return {
            'mse': mse, 'mae': mae, 'r2': r2,
            'y_test': y_test_original, 'y_pred': y_pred_original
        }

    def print_trading_report(self, data: pd.DataFrame, last_n_predictions: int = 10):
        report = self.generate_trading_report(data, last_n_predictions)
        
        print("\n" + "="*70)
        print(f"📊 ОТЧЕТ ПО ТОРГОВЫМ СИГНАЛАМ ЗА ПОСЛЕДНИЕ {last_n_predictions} ДНЕЙ")
        print("="*70)
        
        print(f"Всего сгенерировано сигналов: {report['total_signals']}")
        print(f"Всего 'горячих' сделок (R/R >= 3:1): {report['total_hot_deals']}")
        print(f"Средний скор сигнала: {report['avg_score']:.2f}")
        
        print("\nСводка по 'горячим' сделкам по R/R:")
        if report['hot_deals_summary']:
            for rr, summary in sorted(report['hot_deals_summary'].items()):
                print(f"  R/R {rr}:1: Количество {summary['count']}, Ср.вероятность {summary['avg_probability']:.2f}, Общий потенциал ${summary['total_profit_potential']:.2f}")
        else:
            print("  Горячие сделки не обнаружены.")
            
        print("\nТоп-5 недавних сигналов:")
        for i, signal in enumerate(report['ranked_signals'][:5]):
            print(f"\n--- Сигнал {i+1} ({signal['timestamp'].strftime('%Y-%m-%d')}) ---")
            print(f"  Стратегия: {signal['strategy']}")
            print(f"  Направление: {signal['trading_signals']['direction']}")
            print(f"  Уверенность: {signal['confidence']:.2f}")
            print(f"  Скор: {signal['score']:.2f}")
            print(f"  Вход: ${signal['trading_signals']['entry_price']:.2f}, SL: ${signal['trading_signals']['stop_loss']:.2f}")
            print(f"  Горячие сделки:")
            if signal['trading_signals']['hot_deals']:
                for deal in signal['trading_signals']['hot_deals']:
                    print(f"    TP: {deal['tp_level']} (R/R {deal['rr_ratio']}:1), Потенциал: ${deal['profit_potential']:.2f}")
            else:
                print("    Нет горячих сделок.")
        print("="*70)

    def generate_trading_report(self, data: pd.DataFrame, last_n_predictions: int = 10) -> Dict:
        if not self.is_trained:
            raise ValueError("Модель не обучена")
        
        signals = []
        start_idx = max(self.lookback_period, len(data) - last_n_predictions)
        
        for i in range(start_idx, len(data)):
            try:
                prediction_result = self.predict_with_trading_signals(data, i)
                prediction_result['timestamp'] = data.index[i] if hasattr(data, 'index') else i
                signals.append(prediction_result)
            except ValueError as e:
                continue
            except Exception as e:
                # print(f"An unexpected error occurred at index {i}: {e}")
                continue
        
        ranked_signals = self.signal_generator.rank_signals(signals)
        
        hot_deals_summary = {}
        total_hot_deals = 0
        
        for signal in ranked_signals:
            hot_deals = signal['trading_signals']['hot_deals']
            total_hot_deals += len(hot_deals)
            
            for deal in hot_deals:
                rr_ratio = deal['rr_ratio']
                if rr_ratio not in hot_deals_summary:
                    hot_deals_summary[rr_ratio] = {
                        'count': 0, 'avg_probability': 0, 'total_profit_potential': 0
                    }
                hot_deals_summary[rr_ratio]['count'] += 1
                hot_deals_summary[rr_ratio]['avg_probability'] += deal['probability']
                hot_deals_summary[rr_ratio]['total_profit_potential'] += deal['profit_potential']
        
        for rr_ratio in hot_deals_summary:
            count = hot_deals_summary[rr_ratio]['count']
            if count > 0:
                hot_deals_summary[rr_ratio]['avg_probability'] /= count
        
        return {
            'total_signals': len(signals),
            'ranked_signals': ranked_signals,
            'hot_deals_summary': hot_deals_summary,
            'total_hot_deals': total_hot_deals,
            'avg_score': np.mean([s['score'] for s in ranked_signals]) if ranked_signals else 0
        }

    def plot_trading_signals(self, data: pd.DataFrame, last_n_days: int = 30):
        if not self.is_trained:
            print("❌ Модель не обучена!")
            return
        
        plot_data = data.iloc[-last_n_days:].copy()
        
        signals = []
        signal_indices = []
        
        for i_plot_data in range(len(plot_data)):
            original_idx = len(data) - last_n_days + i_plot_data
            if original_idx >= self.lookback_period and original_idx < len(data) - self.prediction_horizon:
                try:
                    signal = self.predict_with_trading_signals(data, original_idx)
                    signals.append(signal)
                    signal_indices.append(i_plot_data)
                except ValueError as e:
                    continue
                except Exception as e:
                    continue
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        
        ax1.plot(plot_data.index, plot_data['close'], label='Цена закрытия', linewidth=2)
        
        for i, signal in enumerate(signals):
            idx = signal_indices[i]
            trading_signal = signal['trading_signals']
            
            if trading_signal['direction'] == 'LONG':
                ax1.scatter(plot_data.index[idx], signal['current_price'], 
                            color='green', marker='^', s=100, alpha=0.7, label='Buy Signal' if i == 0 else "")
                ax1.scatter(plot_data.index[idx], trading_signal['stop_loss'], 
                            color='red', marker='v', s=50, alpha=0.7, label='Stop Loss (Buy)' if i == 0 else "")
            else:
                ax1.scatter(plot_data.index[idx], signal['current_price'], 
                            color='red', marker='v', s=100, alpha=0.7, label='Sell Signal' if i == 0 else "")
                ax1.scatter(plot_data.index[idx], trading_signal['stop_loss'], 
                            color='green', marker='^', s=50, alpha=0.7, label='Stop Loss (Sell)' if i == 0 else "")
        
        ax1.set_title('Торговые сигналы на графике цены')
        ax1.set_ylabel('Цена USD')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Ensure 'atri' column exists for plotting
        if 'atri' in plot_data.columns and not plot_data['atri'].isnull().all():
            ax2.plot(plot_data.index, plot_data['atri'], label='ATRI', color='orange')
            ax2.set_title('Дневной ATRI')
            ax2.set_ylabel('ATRI')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.set_title('Дневной ATRI (Данные отсутствуют)')
            ax2.text(0.5, 0.5, 'ATRI data not available for plotting', 
                     horizontalalignment='center', verticalalignment='center', 
                     transform=ax2.transAxes, color='gray')


        hot_deals_data = []
        timestamps = []
        
        for i, signal in enumerate(signals):
            hot_deals = signal['trading_signals']['hot_deals']
            timestamp = plot_data.index[signal_indices[i]]
            
            if hot_deals:
                max_rr = max(deal['rr_ratio'] for deal in hot_deals)
                hot_deals_data.append(max_rr)
                timestamps.append(timestamp)
            else:
                hot_deals_data.append(0)
                timestamps.append(timestamp)
        
        ax3.bar(timestamps, hot_deals_data, alpha=0.7, color='purple', width=0.8)
        ax3.set_title('Максимальное соотношение R/R для горячих сделок')
        ax3.set_ylabel('R/R Ratio')
        ax3.axhline(y=3, color='red', linestyle='--', alpha=0.7, label='Минимум для горячих сделок')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.xlabel('Дата')
        plt.tight_layout()
        plt.show()
        
        return fig

    def plot_training_history(self):
        if self.training_history is None:
            print("Модель не обучена")
            return

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.training_history.history['loss'], label='Training Loss')
        plt.plot(self.training_history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.training_history.history['mae'], label='Training MAE')
        plt.plot(self.training_history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_predictions(self, n_samples=100):
        if self.model is None or not self.is_trained:
            print("❌ Модель не обучена!")
            return

        eval_results = self.evaluate_model()
        if eval_results is None:
            return

        y_test = eval_results['y_test'][:n_samples]
        y_pred = eval_results['y_pred'][:n_samples]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

        ax1.plot(y_test.flatten(), label='Actual', alpha=0.7)
        ax1.plot(y_pred.flatten(), label='Predicted', alpha=0.7)
        ax1.set_title(f'Сравнение предсказаний и реальных значений (первые {n_samples} точек)')
        ax1.set_xlabel('Время')
        ax1.set_ylabel('Цена')
        ax1.legend()
        ax1.grid(True)

        errors = y_test.flatten() - y_pred.flatten()
        ax2.plot(errors, color='red', alpha=0.7, label='Prediction Error')
        ax2.axhline(y=0, color='black', linestyle='--', label='Zero Error')
        ax2.set_title('Ошибки предсказаний')
        ax2.set_xlabel('Время')
        ax2.set_ylabel('Ошибка')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

        return fig

    def save_model_auto(self):
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        model_filename = os.path.join(self.model_save_dir, f"crypto_nn_model_{timestamp}.keras")
        try:
            self.model.save(model_filename)
            print(f"✅ Модель успешно сохранена: {model_filename}")
        except Exception as e:
            print(f"❌ Ошибка при сохранении модели: {e}")

    def list_saved_models(self):
        return [f for f in os.listdir(self.model_save_dir) if f.endswith('.keras')]


# == Пример использования улучшенной системы ==
def example_usage():
    print("🚀 Запуск улучшенной системы торговых сигналов...")
    
    csv_file_path = '/content/drive/MyDrive/kandles.csv'
    
    try:
        def load_data_from_csv(csv_file_path, start_date=None, end_date=None):
            data = pd.read_csv(csv_file_path, parse_dates=['open_time'])
            data.rename(columns={'open_time': 'Date', 'close': 'close', 'high': 'high', 'low': 'low', 'open': 'open', 'vol': 'volume'}, inplace=True) # Ensure all columns are lowercased and named correctly
            data.set_index('Date', inplace=True) # Set Date as index
            data.sort_index(inplace=True)
            
            if start_date:
                data = data[data.index >= pd.to_datetime(start_date)]
            if end_date:
                data = data[data.index <= pd.to_datetime(end_date)]
            
            return data
        
        # Adjust date range for more recent data, which is available in July 2025
        df = load_data_from_csv(csv_file_path, start_date='2024-07-01', end_date='2025-07-23') 
        data = df.copy()
        
    except FileNotFoundError:
        print(f"⚠️ Файл не найден по пути: {csv_file_path}, создаем синтетические...")
        np.random.seed(42)
        dates = pd.date_range('2025-01-01', periods=1000, freq='1D') # Adjusted dates
        price = 45000
        prices = [price]
        for i in range(999):
            change = np.random.normal(0, 0.025)
            price *= (1 + change)
            prices.append(price)

        data = pd.DataFrame(index=dates)
        data['close'] = prices
        data['open'] = data['close'].shift(1) * (1 + np.random.normal(0, 0.005, len(data)))
        data['high'] = np.maximum(data['open'], data['close']) * (1 + np.abs(np.random.normal(0, 0.015, len(data))))
        data['low'] = np.minimum(data['open'], data['close']) * (1 - np.abs(np.random.normal(0, 0.015, len(data))))
        data['volume'] = np.random.uniform(1000, 10000, len(data))
        data = data.dropna()
    except Exception as e:
        print(f"❌ Произошла ошибка при загрузке или создании данных: {e}")
        traceback.print_exc()
        return

    # Добавляем ATRI (Average True Range Index) - defined as standalone function for global use
    def calculate_atri_standalone(data_df: pd.DataFrame, period: int = 14):
        high_low = data_df['high'] - data_df['low']
        high_close_prev = np.abs(data_df['high'] - data_df['close'].shift(1))
        low_close_prev = np.abs(data_df['low'] - data_df['close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        atri = true_range.rolling(window=period).mean()
        
        # Optional: Filter ATRI as in _calculate_atri (if desired for standalone)
        # atri_mean = atri.mean()
        # atri_filtered = atri.where((atri < 1.5 * atri_mean) & (atri > 0.6 * atri_mean))
        # return atri_filtered
        return atri
    
    data['atri'] = calculate_atri_standalone(data)
    data = data.dropna() # Drop rows where ATRI calculation resulted in NaN (initial rows)

    print(f"📊 Загружено {len(data)} записей данных")

    # Define hyperparameters to search
    hyperparameters = {
        'lookback_period': [20, 30, 40],
        'prediction_horizon': [3, 5, 7],
        'batch_size': [16, 32],
        'epochs': [50, 100] # Use fewer epochs for faster testing
    }

    best_mae = float('inf')
    best_params = {}
    best_predictor = None

    # Iterating through hyperparameter combinations
    for lb_period in hyperparameters['lookback_period']:
        for pred_horizon in hyperparameters['prediction_horizon']:
            for b_size in hyperparameters['batch_size']:
                for n_epochs in hyperparameters['epochs']:
                    print(f"\n--- Testing Parameters: Lookback={lb_period}, Horizon={pred_horizon}, Batch={b_size}, Epochs={n_epochs} ---")
                    
                    # Create a fresh DaysLevel instance for each run if data is modified
                    days_level = MockDaysLevel(data)

                    predictor = CriptoPredictionNN(
                        data=data.copy(), # Pass a copy to avoid unintended modifications
                        days_level=days_level,
                        lookback_period=lb_period,
                        prediction_horizon=pred_horizon,
                        prediction_type='price',
                        model_save_dir='crypto_models'
                    )

                    try:
                        history = predictor.train(data.copy(), epochs=n_epochs, batch_size=b_size, verbose=0) # Use verbose=0 for cleaner output during grid search
                        if history is None: # Training might fail if data is insufficient
                            print(f"Training failed for these parameters. Skipping.")
                            continue

                        eval_results = predictor.evaluate_model()
                        if eval_results is None:
                            print(f"Evaluation failed for these parameters. Skipping.")
                            continue

                        current_mae = eval_results['mae']
                        print(f"Current MAE: {current_mae:.6f}")

                        if current_mae < best_mae:
                            best_mae = current_mae
                            best_params = {
                                'lookback_period': lb_period,
                                'prediction_horizon': pred_horizon,
                                'batch_size': b_size,
                                'epochs': n_epochs
                            }
                            best_predictor = predictor # Store the best model's predictor instance

                    except Exception as e:
                        print(f"❌ Error during training or evaluation for current parameters: {e}")
                        traceback.print_exc()
                        continue

    print("\n" + "="*70)
    print("✨ ЛУЧШАЯ МОДЕЛЬ НАЙДЕНА!")
    print("="*70)
    print(f"✅ Лучший MAE: {best_mae:.6f}")
    print(f"⚙️ Лучшие параметры: {best_params}")

    if best_predictor:
        print("\n🔮 Генерируем торговые сигналы для лучшей модели...")
        try:
            prediction = best_predictor.predict_with_trading_signals(data) # Use the original data for final prediction
            print("\n" + "="*70)
            print("📈 РЕЗУЛЬТАТ АНАЛИЗА ЛУЧШЕЙ МОДЕЛИ")
            print("="*70)
            print(f"💰 Текущая цена: ${prediction['current_price']:.2f}")
            print(f"🎯 Предсказанная цена: ${prediction['predicted_price']:.2f}")
            print(f"📊 Ожидаемое изменение: {prediction['predicted_change_percent']:.2f}%")
            print(f"📋 Стратегия: {prediction['strategy']}")
            print(f"🎲 Уверенность: {prediction['confidence']:.2f}")
            print(f"📏 Дневной ATRI: ${prediction['daily_atri']:.2f}")

            ts = prediction['trading_signals']
            print(f"\n🎯 ТОРГОВЫЕ СИГНАЛЫ:")
            print(f"   Направление: {ts['direction']}")
            print(f"   Точка входа: ${ts['entry_price']:.2f}")
            print(f"   Стоп-лосс: ${ts['stop_loss']:.2f}")
            print(f"   Риск: ${ts['risk']:.2f}")

            print(f"\n🔥 ГОРЯЧИЕ СДЕЛКИ (R/R ≥ 3:1):")
            for i, deal in enumerate(ts['hot_deals'][:5]):
                print(f"   {i+1}. {deal['tp_level']}: R/R {deal['rr_ratio']}:1, "
                      f"вероятность: {deal['probability']:.2f}, "
                      f"потенциал: ${deal['profit_potential']:.2f}")

            print(f"\n📊 ВСЕ УРОВНИ ТЕЙК-ПРОФИТА:")
            for tp_name, tp_data in ts['take_profits'].items():
                print(f"   {tp_name}: ${tp_data['price']:.2f} "
                      f"(R/R {tp_data['ratio']}:1, "
                      f"расстояние: {tp_data['distance_atri']:.1f} ATRI, "
                      f"вероятность: {tp_data['probability']:.2f})")

            best_predictor.print_trading_report(data, last_n_predictions=20)

            print("\n📊 Создание графиков для лучшей модели...")
            best_predictor.plot_training_history()
            best_predictor.plot_predictions(n_samples=200)
            best_predictor.plot_trading_signals(data, last_n_days=60)

            saved_models = best_predictor.list_saved_models()
            print(f"\n💾 Сохраненные модели: {saved_models}")

        except Exception as e:
            print(f"❌ Ошибка при использовании лучшей модели: {e}")
            traceback.print_exc()
    else:
        print("Не удалось найти лучшую модель.")

if __name__ == "__main__":
    example_usage()