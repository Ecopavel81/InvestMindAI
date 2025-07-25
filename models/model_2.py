import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from h2o.automl import H2OAutoML # Uncomment if you use H2O elsewhere
import h2o # Uncomment if you use H2O elsewhere
import warnings
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score


# == Загрузка данных ==
csv_file_path = '/mnt/c/Users/ecopa/Desktop/Proekts/Trader bot/kandles.csv'
df_raw = pd.read_csv(csv_file_path)
print(df_raw.columns)

Xtrain = pd.read_csv(csv_file_path)
Xtest = pd.read_csv(csv_file_path)

def load_data_from_csv(csv_file_path, start_date=None, end_date=None):
    """
    Загрузка данных о цене BTC из CSV и фильтрация по дате.

    Аргументы:
    filepath -- путь к CSV-файлу
    start_date, end_date -- границы фильтрации (строки 'YYYY-MM-DD')
    """
    # Читаем open_time как дату
    data = pd.read_csv(csv_file_path, parse_dates=['open_time'])

    # Переименовываем колонку для единообразия
    data.rename(columns={'open_time': 'Date', 'close': 'Close'}, inplace=True)
    data.sort_values('Date', inplace=True)

    # Фильтрация по датам
    if start_date:
        data = data[data['Date'] >= pd.to_datetime(start_date)]
    if end_date:
        data = data[data['Date'] <= pd.to_datetime(end_date)]

    return data

def visualize_data(data):
    """
    Визуализация исторических цен BTC.

    Аргументы:
    data -- DataFrame с колонками Date и Close
    """
    plt.figure(figsize=(14, 6))
    plt.plot(data['Date'], data['Close'], label='Цена закрытия')
    plt.title('Исторические цены BTC')
    plt.xlabel('Дата')
    plt.ylabel('Цена в USD')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

df = load_data_from_csv(csv_file_path, start_date='2024-11-01', end_date='2025-06-05')
visualize_data(df)

# --- Define your DaysLevel class here ---
# (Paste the complete and corrected DaysLevel class from your prompt here)
class DaysLevel:
    """Класс для поиска сильных уровней поддержки и сопротивления с ранжированием по временным периодам"""

    def __init__(self, data, symbol=None, atr_period=14, timeframe='1H',
                 min_atri_multiplier=2.0, max_atri_multiplier=5.0):
        self.data = data.copy()
        self.symbol = symbol
        self.atr_period = atr_period
        self.timeframe = timeframe
        self.min_atri_multiplier = min_atri_multiplier
        self.max_atri_multiplier = max_atri_multiplier
        self.indicators = {}
        self.combined_levels = []
        self.verbose = True

        self.timeframe_weights = {
            '1M': 10, '1W': 8, '1D': 6, '4H': 4, '1H': 2, '30m': 1, '15m': 0.5
        }

        self.data.columns = [col.lower() for col in self.data.columns]
        required_cols = ['high', 'low', 'close', 'open']
        if not all(col in self.data.columns for col in required_cols):
            raise ValueError(f"Отсутствуют необходимые колонки: {required_cols}")

        if not isinstance(self.data.index, pd.DatetimeIndex):
            if 'date' in self.data.columns or 'datetime' in self.data.columns:
                date_col = 'date' if 'date' in self.data.columns else 'datetime'
                self.data.index = pd.to_datetime(self.data[date_col])
                self.data.drop(columns=[date_col], inplace=True)
            else:
                self.data.index = pd.to_datetime(self.data.index)
        self._calculate_atri()

    def _calculate_atri(self):
        high = self.data['high']
        low = self.data['low']
        close = self.data['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range_i = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atri_raw = true_range_i.ewm(span=self.atr_period, adjust=False).mean()

        if atri_raw.isnull().all():
            if self.verbose:
                print("⚠️ ATRI raw is all NaNs. Setting ATRI to NaN for all data points.")
            self.data['atri'] = np.nan
            self.indicators['ATRI'] = np.nan
            return

        atri_mean = atri_raw.mean()

        if pd.isna(atri_mean) or atri_mean <= 1e-9:
            if self.verbose:
                print(f"⚠️ ATRI mean is NaN or very small ({atri_mean:.4f}). Skipping ATRI filtering for now.")
            self.data['atri'] = atri_raw
        else:
            atri_filtered = atri_raw.where(
                (atri_raw < self.max_atri_multiplier * atri_mean) & (atri_raw > self.min_atri_multiplier * atri_mean)
            )
            self.data['atri'] = atri_filtered.fillna(atri_raw)

        self.data['atri'].fillna(method='ffill', inplace=True)
        self.data['atri'].fillna(method='bfill', inplace=True)

        if self.data['atri'].isnull().any():
             if self.verbose: print("❗ Warning: Some ATRI values are still NaN after filling. Consider data quality or ATRI period.")

        self.indicators['ATRI'] = self.data['atri']

    def _convert_to_timeframe(self, target_timeframe):
        freq_map = {
            '15m': '15T', '30m': '30T', '1H': '1H', '4H': '4H', '1D': '1D', '1W': '1W', '1M': '1M'
        }
        if target_timeframe not in freq_map:
            raise ValueError(f"Неподдерживаемый временной период: {target_timeframe}")

        resampled_data = self.data.resample(freq_map[target_timeframe]).agg({
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'open': 'first',
            'atri': 'mean'
        }).dropna()
        return resampled_data

    def level_cinching(self, timeframe=None):
        if timeframe is None: timeframe = self.timeframe
        data = self._convert_to_timeframe(timeframe) if timeframe != self.timeframe else self.data.copy()
        highs = data['high']
        cinch_levels = []
        strength = self._get_level_strength('cinching', timeframe)
        if len(highs) < 5: return []
        for i in range(4, len(highs)):
            h1, h2, h3, h4 = highs.iloc[i - 4:i]
            if (h4 == h3) and (h3 > h2) and (h2 > h1):
                cinch_levels.append((data.index[i], h4, strength, timeframe))
        return cinch_levels

    def level_mirror(self, tolerance=0.002, timeframe=None):
        if timeframe is None: timeframe = self.timeframe
        data = self._convert_to_timeframe(timeframe) if timeframe != self.timeframe else self.data.copy()
        highs, lows = data['high'], data['low']
        mirror_levels = []
        strength = self._get_level_strength('mirror', timeframe)
        if len(data) < 4: return []
        for i in range(3, len(data)):
            h_prev, l_prev = highs.iloc[i - 1], lows.iloc[i - 1]
            h_curr, l_curr = highs.iloc[i], lows.iloc[i]
            if (abs(h_curr - l_prev) / l_prev < tolerance) or (abs(l_curr - h_prev) / h_prev < tolerance):
                mirror_levels.append((data.index[i], (h_curr + l_curr + h_prev + l_prev) / 4, strength, timeframe))
        return mirror_levels

    def level_change(self, atr_multiplier=1.0, timeframe=None):
        if timeframe is None: timeframe = self.timeframe
        data = self._convert_to_timeframe(timeframe) if timeframe != self.timeframe else self.data.copy()
        highs, lows, atri = data['high'], data['low'], data.get('atri')
        change_levels = []
        strength = self._get_level_strength('change', timeframe)

        if atri is None or atri.isnull().all():
            if self.verbose: print(f"⚠️ ATRI не найден или содержит только NaN для {timeframe}. Невозможно найти уровни 'change'.")
            return []

        atri_mean = atri.mean()
        if pd.isna(atri_mean) or atri_mean <= 1e-9:
            if self.verbose: print(f"⚠️ ATRI mean is NaN or zero for {timeframe}. Cannot use ATRI for 'change' levels.")
            return []

        if len(highs) < 6: return []

        for i in range(5, len(highs)):
            l1, l2, l3, l4, l5 = lows.iloc[i-5:i]
            if l2 < l1 and l2 < l3 and l3 < l4 and l4 < l5:
                if (l3 - l2 > atri_mean * atr_multiplier) and (l4 - l2 > atri_mean * atr_multiplier):
                    change_levels.append((data.index[i], l2, strength, timeframe, 'up'))
            h1, h2, h3, h4, h5 = highs.iloc[i-5:i]
            if h2 > h1 and h2 > h3 and h3 > h4 and h4 > h5:
                if (h2 - h3 > atri_mean * atr_multiplier) and (h2 - h4 > atri_mean * atr_multiplier):
                    change_levels.append((data.index[i], h2, strength, timeframe, 'down'))
        return change_levels

    def level_paranorm(self, atr_paranorm_multiplier=1.5, timeframe=None):
        if timeframe is None: timeframe = self.timeframe
        data = self._convert_to_timeframe(timeframe) if timeframe != self.timeframe else self.data.copy()
        highs, lows, closes, opens, atri = data['high'], data['low'], data['close'], data['open'], data.get('atri')
        paranorm_levels = []
        strength = self._get_level_strength('paranorm', timeframe)

        if atri is None or atri.isnull().all():
            if self.verbose: print(f"⚠️ ATRI не найден или содержит только NaN для {timeframe}. Невозможно найти уровни 'paranorm'.")
            return []

        atri_mean = atri.mean()
        if pd.isna(atri_mean) or atri_mean <= 1e-9:
            if self.verbose: print(f"⚠️ ATRI mean is NaN or zero for {timeframe}. Cannot use ATRI for 'paranorm' levels.")
            return []

        if len(data) < 2: return []

        for i in range(1, len(data)):
            current_close, current_open, current_high, current_low = closes.iloc[i], opens.iloc[i], highs.iloc[i], lows.iloc[i]
            body_size = abs(current_close - current_open)
            bar_range = current_high - current_low

            if (current_close > current_open and
                body_size > atri_mean * atr_paranorm_multiplier and
                (current_high - current_close) < body_size * 0.3 and
                bar_range > atri_mean * atr_paranorm_multiplier):
                paranorm_levels.append((data.index[i], current_high, strength, timeframe, 'up'))
            elif (current_close < current_open and
                  body_size > atri_mean * atr_paranorm_multiplier and
                  (current_open - current_low) < body_size * 0.3 and
                  bar_range > atri_mean * atr_paranorm_multiplier):
                paranorm_levels.append((data.index[i], current_low, strength, timeframe, 'down'))
        return paranorm_levels

    def _get_level_strength(self, level_type, timeframe):
        type_multipliers = {'cinching': 1.0, 'mirror': 1.2, 'change': 1.5, 'paranorm': 2.0}
        timeframe_weight = self.timeframe_weights.get(timeframe, 1)
        type_multiplier = type_multipliers.get(level_type, 1)
        return timeframe_weight * type_multiplier

    def get_all_levels(self, timeframes: Optional[List[str]] = None) -> Dict[str, Dict[str, List[Tuple]]]:
        if timeframes is None:
            timeframes = sorted(
                list(set([self.timeframe, '1H', '1D', '1W', '1M'])),
                key=lambda x: self.timeframe_weights.get(x, 0),
                reverse=True
            )
        all_levels = {}
        for tf in timeframes:
            try:
                levels = {
                    'cinching': self.level_cinching(timeframe=tf),
                    'mirror': self.level_mirror(timeframe=tf),
                    'change': self.level_change(timeframe=tf),
                    'paranorm': self.level_paranorm(timeframe=tf)
                }
                all_levels[tf] = levels
            except ValueError as e:
                if self.verbose: print(f"⚠️ Ошибка данных при анализе {tf}: {e}. Пропуск этого таймфрейма.")
                continue
            except Exception as e:
                if self.verbose: print(f"⚠️ Непредвиденная ошибка при анализе {tf}: {e}. Пропуск этого таймфрейма.")
                continue
        return all_levels

    def get_strongest_levels(self, timeframes: Optional[List[str]] = None, top_n: int = 10) -> List[Tuple]:
        all_levels_data = self.get_all_levels(timeframes)
        combined_levels = []
        for tf, level_types in all_levels_data.items():
            for level_type, levels_list in level_types.items():
                for level_data in levels_list:
                    if len(level_data) == 5:
                        date, level_price, strength, timeframe_found, direction = level_data
                    elif len(level_data) == 4:
                        date, level_price, strength, timeframe_found = level_data
                        direction = 'neutral'
                    else:
                        if self.verbose: print(f"⚠️ Неожиданный формат данных уровня: {level_data}. Пропуск.")
                        continue
                    combined_levels.append((date, level_price, strength, timeframe_found, level_type, direction))

        self.combined_levels = combined_levels

        combined_levels.sort(key=lambda x: x[2], reverse=True)
        return combined_levels[:top_n]

    def print_levels_summary(self, timeframes: Optional[List[str]] = None):
        all_levels_data = self.get_all_levels(timeframes)
        strongest_levels = self.get_strongest_levels(timeframes)

        print(f"📊 Анализ уровней для {self.symbol or 'инструмента'}:")
        print("=" * 60)

        sorted_timeframes = sorted(all_levels_data.keys(), key=lambda x: self.timeframe_weights.get(x, 0), reverse=True)

        for tf in sorted_timeframes:
            levels = all_levels_data[tf]
            tf_weight = self.timeframe_weights.get(tf, 1)
            print(f"\n⏰ Временной период: {tf} (вес: {tf_weight})")
            print(f"🟩 Уровни поджатия (cinching): {len(levels.get('cinching', []))}")
            print(f"🔄 Зеркальные уровни (mirror): {len(levels.get('mirror', []))}")
            print(f"📈 Уровни излома (change): {len(levels.get('change', []))}")
            print(f"⚡ Паранормальные уровни (paranorm): {len(levels.get('paranorm', []))}")

        print(f"\n🏆 ТОП-{len(strongest_levels)} самых сильных уровней:")
        print("-" * 60)
        for i, (date, level, strength, tf_found, level_type, direction) in enumerate(strongest_levels, 1):
            direction_emoji = "📈" if direction == 'up' else "📉" if direction == 'down' else "⚖️"
            print(f"{i:2d}. {date.strftime('%Y-%m-%d %H:%M')} | {level:.5f} | "
                  f"💪{strength:.1f} | {tf_found} | {level_type} {direction_emoji}")

        return all_levels_data # Returning all_levels dictionary, as per the comment
# --- End of DaysLevel Class ---


# --- Define your LevelAnalysisNN class here ---
class LevelAnalysisNN:
    def __init__(self, days_level, top_n: int, sequence_length: int = 30,
                 prediction_horizon: int = 5, prediction_type: str = 'price', verbose: bool = True):
        self.dl = days_level
        # The line below implicitly calls get_all_levels and populates self.dl.combined_levels
        self.dl.get_strongest_levels(top_n=top_n)
        self.top_n = top_n
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.prediction_type = prediction_type
        self.price_scaler = MinMaxScaler()
        self.feature_scaler = StandardScaler()
        self.model = None
        self.history = None
        self.verbose = verbose
        # Add attributes for train/test data to be accessible by Visualiser if needed
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_columns_used = [] # To store the actual feature columns used after preparation

    def prepare_level_features(self, timeframes: List[str]) -> pd.DataFrame:
        # This method should return a DataFrame with all features needed for the NN.
        # It's currently just returning a copy of dl.data, assuming all features are there.
        # If your level-based features are computed separately, merge them here.
        return self.dl.data.copy()

    def prepare_sequences(self, data: pd.DataFrame, strict: bool = False) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        feature_columns = [
            'open', 'high', 'low', 'close', 'atri',
            'price_change', 'volatility', 'momentum', 'momentum_fast', 'momentum_slow',
            'sma_ratio', 'sma_ratio_long', 'rsi', 'stoch_k', 'stoch_d',
            'ema_signal', 'support_distance', 'resistance_distance',
            'support_strength', 'resistance_strength', 'level_density',
            'level_type_cinching', 'level_type_mirror', 'level_type_change',
            'level_type_paranorm', 'timeframe_weight', 'support_touch',
            'resistance_touch', 'level_bounce', 'level_break', 'false_breakout',
            'time_at_support', 'time_at_resistance', 'level_cluster_strength',
            'level_cluster_count', 'level_cluster_range',
            'h20_sma', 'h20_volatility', 'h20_momentum', 'price_to_h20_sma',
            'h20_support_strength', 'h20_resistance_strength', 'h20_level_density',
            'h20_interaction', 'h20_cluster_strength'
        ]

        # Filter to only include columns that actually exist in the provided data
        available_columns = [col for col in feature_columns if col in data.columns]
        missing_columns = [col for col in feature_columns if col not in data.columns]
        self.feature_columns_used = available_columns # Store the columns actually used

        if strict and missing_columns:
            raise ValueError(f"Некоторые признаки отсутствуют: {missing_columns}")

        if missing_columns and self.verbose:
            print(f"⚠️ Отсутствуют признаки: {missing_columns}")

        # Ensure 'close' is always included if it exists in data, as it's typically the target
        data_clean_cols = list(set(available_columns + (['close'] if 'close' in data.columns else [])))
        data_clean = data[data_clean_cols].dropna() # Apply dropna on the selected features only

        if self.verbose: print(f"📉 После dropna: {len(data_clean)} строк")

        # Check if enough data remains after dropping NaNs
        if len(data_clean) < self.sequence_length + self.prediction_horizon:
            raise ValueError(f"Недостаточно данных: {len(data_clean)} строк < {self.sequence_length + self.prediction_horizon}")

        features = data_clean[available_columns].values

        # Scale features
        # Ensure that feature_scaler is only fitted once or managed appropriately if data changes
        features_scaled = self.feature_scaler.fit_transform(features)
        num_features = features_scaled.shape[1]

        X, y = [], []

        if self.prediction_type == 'price':
            if self.verbose: print(f"🔍 Data length: {len(data_clean)}, sequence_length: {self.sequence_length}, prediction_horizon: {self.prediction_horizon}")

            # Fit price scaler *only on the target values that will be used*
            # This is important to avoid data leakage and ensure correct scaling
            # Create a temporary series of all possible target values from the clean data
            all_possible_targets = []
            for i in range(self.sequence_length, len(data_clean) - self.prediction_horizon + 1):
                all_possible_targets.extend(data_clean['close'].values[i : i + self.prediction_horizon])

            if len(all_possible_targets) == 0:
                raise ValueError("⚠️ Не удалось собрать target-цены для масштабирования (all_possible_targets пустой). Проверьте данные и параметры.")

            self.price_scaler.fit(np.array(all_possible_targets).reshape(-1, 1))

            for i in range(self.sequence_length, len(data_clean) - self.prediction_horizon + 1):
                seq_x = features_scaled[i - self.sequence_length : i]
                target_slice = data_clean['close'].values[i : i + self.prediction_horizon]
                target_scaled = self.price_scaler.transform(target_slice.reshape(-1, 1)).flatten()

                # Check for NaNs within the sequence itself *after* slicing, though dropna should handle most
                if np.any(np.isnan(seq_x)) or np.any(np.isnan(target_scaled)):
                    if self.verbose: print(f"NaN в последовательности {i}. Пропуск.")
                    continue
                X.append(seq_x)
                y.append(target_scaled)

        elif self.prediction_type == 'direction':
            for i in range(self.sequence_length, len(features_scaled) - self.prediction_horizon + 1):
                seq_x = features_scaled[i - self.sequence_length : i]
                current_price = data_clean['close'].iloc[i - 1] # Use the last price of the sequence
                future_price = data_clean['close'].iloc[i + self.prediction_horizon - 1]
                seq_y = 1 if future_price > current_price else 0 # 1 for up, 0 for down/flat
                if seq_x.shape[0] == self.sequence_length and seq_x.shape[1] == num_features and not np.any(np.isnan(seq_x)):
                    X.append(seq_x)
                    y.append(seq_y)

        elif self.prediction_type == 'breakout':
            if 'level_break' not in data_clean.columns:
                raise ValueError("Столбец 'level_break' отсутствует в данных для 'breakout' типа прогноза.")
            for i in range(self.sequence_length, len(features_scaled) - self.prediction_horizon + 1):
                seq_x = features_scaled[i - self.sequence_length : i]
                # Target is whether a breakout occurred at the horizon
                seq_y = data_clean['level_break'].iloc[i + self.prediction_horizon - 1]
                if seq_x.shape[0] == self.sequence_length and seq_x.shape[1] == num_features and not np.any(np.isnan(seq_x)) and not np.isnan(seq_y):
                    X.append(seq_x)
                    y.append(seq_y)
        else:
            raise ValueError(f"Неизвестный тип прогноза: {self.prediction_type}")

        if len(X) == 0:
            if self.verbose:
                print(f"🔍 Debug info (prepare_sequences):")
                print(f"   - Available columns: {available_columns}")
                print(f"   - Data shape (after dropna): {data_clean.shape}")
                print(f"   - Features shape: {features_scaled.shape if 'features_scaled' in locals() else 'Not yet scaled'}")
                print(f"   - Expected sequence shape: ({self.sequence_length}, {num_features if 'num_features' in locals() else '?'})")
                print(f"   - Prediction horizon: {self.prediction_horizon}")
                print(f"   - First few target values (if price): {data_clean['close'].values[self.sequence_length : self.sequence_length + 5]}")
            raise ValueError("Не удалось создать ни одной последовательности.")

        if self.verbose: print(f"✅ Создано последовательностей: {len(X)}")
        X_array = np.array(X)
        y_array = np.array(y)
        if self.verbose: print(f"X shape: {X_array.shape}, y shape: {y_array.shape}")
        return X_array, y_array, data_clean

    def build_model(self, input_shape):
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.LSTM(64, return_sequences=False, dropout=0.2), # return_sequences=False for single output
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu') # Final feature extraction layer
        ])
        if self.prediction_type == 'price':
            model.add(layers.Dense(self.prediction_horizon, activation='linear')) # Output for N steps ahead
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        elif self.prediction_type == 'direction' or self.prediction_type == 'breakout':
            model.add(layers.Dense(1, activation='sigmoid')) # Binary classification
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, timeframes: List[str] = ['1H', '1D'], epochs: int = 100, batch_size: int = 32,
              validation_split: float = 0.2, verbose: int = 1, plot_after_train: bool = True):
        if self.verbose: print("🔄 Подготовка данных для обучения...")
        # Get data from the DaysLevel instance, which now has ATRI computed
        data_for_nn = self.dl.data.copy()

        # Ensure all required features are in data_for_nn
        # If any features (like price_change, volatility etc.) were not generated by DaysLevel directly,
        # they must be added to `data_for_nn` BEFORE calling `prepare_sequences`.
        # For this example, let's assume `data_for_nn` already contains all necessary features,
        # perhaps passed in the initial `data` to `DaysLevel` or computed externally.
        # If not, you'd need to add a feature engineering step here.

        X, y, _ = self.prepare_sequences(data_for_nn) # Use the data_for_nn that should have all features
        if self.verbose: print(f"📊 X.shape={X.shape}, y.shape={y.shape}")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=validation_split, shuffle=False) # Use validation_split for test_size here

        self.model = self.build_model((self.sequence_length, X.shape[2]))
        if self.verbose: self.model.summary()

        callbacks = [
            keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.5),
            keras.callbacks.ModelCheckpoint(f'best_model_{self.prediction_type}.h5', save_best_only=True)
        ]

        if self.verbose: print(f"✅ model output shape: {self.model.output_shape}")
        if self.verbose: print(f"✅ y_train shape: {self.y_train.shape}")
        if self.verbose: print("🚀 Начало обучения...")
        self.history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs, batch_size=batch_size, validation_split=0.2, # Use a fixed 0.2 validation split for Keras internal
            callbacks=callbacks, verbose=verbose
        )
        if self.verbose: print("✅ Обучение завершено!")
        self.evaluate_model()

        if plot_after_train:
            self.plot_training_history()
        return self.history

    def evaluate_model(self):
        if self.model is None:
            print("❌ Модель не обучена!")
            return None # Changed to return None for consistency

        # Ensure X_test exists and is not empty before predicting
        if self.X_test is None or len(self.X_test) == 0:
            print("❌ Нет тестовых данных для оценки!")
            return None # Changed to return None for consistency

        y_pred = self.model.predict(self.X_test)
        print("\n📈 ОЦЕНКА МОДЕЛИ:")
        if self.prediction_type == 'price':
            # Inverse transform both predictions and actual values for meaningful MSE/MAE/R2
            if self.prediction_horizon > 1:
                # If y_test is 2D (samples, horizon), flatten for inverse_transform if needed, then reshape back
                y_test_orig = self.price_scaler.inverse_transform(self.y_test.reshape(-1, 1)).reshape(self.y_test.shape)
                y_pred_orig = self.price_scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)
            else: # Single step prediction
                y_test_orig = self.price_scaler.inverse_transform(self.y_test.reshape(-1, 1)).flatten()
                y_pred_orig = self.price_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

            print(f"MSE: {mean_squared_error(y_test_orig, y_pred_orig):.6f}")
            print(f"MAE: {mean_absolute_error(y_test_orig, y_pred_orig):.6f}")
            print(f"R²: {r2_score(y_test_orig, y_pred_orig):.4f}")
            return {'y_test_orig': y_test_orig, 'y_pred_orig': y_pred_orig,
                    'mse': mean_squared_error(y_test_orig, y_pred_orig),
                    'mae': mean_absolute_error(y_test_orig, y_pred_orig),
                    'r2': r2_score(y_test_orig, y_pred_orig)}
        elif self.prediction_type == 'direction' or self.prediction_type == 'breakout':
            y_pred_bin = (y_pred > 0.5).astype(int)
            print(f"Accuracy: {accuracy_score(self.y_test, y_pred_bin):.4f}")
            print(f"Precision: {precision_score(self.y_test, y_pred_bin, zero_division=0):.4f}")
            print(f"Recall: {recall_score(self.y_test, y_pred_bin, zero_division=0):.4f}")
            return {'y_test_bin': self.y_test, 'y_pred_bin': y_pred_bin,
                    'accuracy': accuracy_score(self.y_test, y_pred_bin),
                    'precision': precision_score(self.y_test, y_pred_bin, zero_division=0),
                    'recall': recall_score(self.y_test, y_pred_bin, zero_division=0)}


    def plot_training_history(self):
        import matplotlib.pyplot as plt
        if not self.history:
            print("❌ История обучения недоступна!")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.history.history['loss'], label='Train Loss')
        if 'val_loss' in self.history.history:
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.show()

        if self.prediction_type in ['direction', 'breakout']:
            if 'accuracy' in self.history.history:
                plt.figure(figsize=(10, 6))
                plt.plot(self.history.history['accuracy'], label='Train Accuracy')
                if 'val_accuracy' in self.history.history:
                    plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
                plt.title("Training and Validation Accuracy")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.legend()
                plt.grid(True)
                plt.show()

    def predict(self, sequence: Optional[np.ndarray] = None, steps: int = 1) -> np.ndarray:
        if self.model is None:
            print("❌ Модель не обучена!")
            return None

        if sequence is None:
            if self.X_test is None or len(self.X_test) == 0:
                raise ValueError("Нет тестовых данных для предсказания. Предоставьте аргумент 'sequence'.")
            # Use the last sequence from test data if no sequence is provided
            sequence = self.X_test[-1:]

        # Ensure the input sequence has the correct 3D shape (batch_size, sequence_length, num_features)
        if sequence.ndim == 2:
            sequence = np.expand_dims(sequence, axis=0) # Add batch dimension if missing

        # Validate input shape against model's expected input shape
        if sequence.shape[1:] != self.model.input_shape[1:]:
            raise ValueError(f"Форма входной последовательности {sequence.shape} не соответствует форме ввода модели {self.model.input_shape}. Ожидаемая (batch_size, {self.model.input_shape[1]}, {self.model.input_shape[2]})")

        predictions = []
        current_sequence = sequence.copy()

        for step_idx in range(steps):
            pred_scaled = self.model.predict(current_sequence, verbose=0)

            if self.prediction_type == 'price':
                # Inverse transform the prediction
                pred_orig = self.price_scaler.inverse_transform(pred_scaled.reshape(-1, 1))
                # If model predicts a horizon, take the relevant step's prediction
                # Here, we assume the model predicts `prediction_horizon` steps at once
                # and we're just recording the first predicted step's value for simplicity
                # or the entire predicted horizon for each `current_sequence`.
                # Let's assume we want the horizon for the current sequence
                predictions.append(pred_orig.flatten())

                if steps > 1:
                    # This autoregressive part is a simplification.
                    # To accurately predict the *next* full feature vector, you'd need:
                    # 1. A way to predict/generate all other features (open, high, low, atri, etc.)
                    # 2. Or, a model that directly predicts the next *full feature vector*.
                    # For this example, we will just update the 'close' price within the feature vector
                    # and assume other features remain constant or are derived simply.

                    # Get the index of 'close' in the feature_columns_used
                    try:
                        close_col_idx = self.feature_columns_used.index('close')
                    except ValueError:
                        raise ValueError("Столбец 'close' не найден в используемых признаках (feature_columns_used).")

                    # Take the first predicted price from the horizon for the next step's input
                    predicted_close_price = pred_scaled[0, 0] # Use the first predicted value from the horizon

                    # Create the new feature vector for the next timestep
                    # This is highly simplified: we take the last timestep's features and update 'close'.
                    next_timestep_features_scaled = current_sequence[0, -1, :].copy()
                    next_timestep_features_scaled[close_col_idx] = predicted_close_price

                    # Shift the sequence: remove the oldest timestep, add the new predicted one
                    current_sequence = np.roll(current_sequence, -1, axis=1) # Shift left by one
                    current_sequence[0, -1, :] = next_timestep_features_scaled # Place new features at the end

            elif self.prediction_type in ['direction', 'breakout']:
                predictions.append((pred_scaled[0] > 0.5).astype(int)) # Binary prediction

                # For classification, autoregression is usually not done on the output itself
                # unless you're predicting sequential classifications. If so, you'd need
                # to decide how the classified output influences the next feature set.
                # For this demo, we'll just break if steps > 1 for classification types.
                if steps > 1:
                    print("⚠️ Авторегрессия для прогнозов типа 'direction' или 'breakout' не реализована в данной упрощенной форме. Возвращается только одно предсказание.")
                    break

        # Flatten predictions if they were single-value outputs, otherwise keep as is (e.g., horizon price predictions)
        final_predictions = np.array(predictions)
        if self.prediction_type == 'price' and self.prediction_horizon == 1 and steps == 1:
             return final_predictions.flatten()
        elif self.prediction_type in ['direction', 'breakout'] and steps == 1:
            return final_predictions.flatten()
        else:
            return final_predictions # Return as is for multi-step price predictions or multi-step classification (if implemented)


    def plot_predictions(self, n_samples: int = 100, plot_train_vs_test: bool = True):
        """
        Визуализирует предсказания модели в сравнении с фактическими значениями.

        Args:
            n_samples (int): Количество образцов (последовательностей) из тестового набора для отображения.
                             Для прогнозирования цен будет показана первая точка каждого горизонта.
            plot_train_vs_test (bool): Если True, покажет также небольшой участок из тренировочных данных.
        """
        if self.model is None:
            print("❌ Модель не обучена! Сначала обучите модель.")
            return

        if self.X_test is None or len(self.X_test) == 0:
            print("❌ Нет тестовых данных для построения графиков предсказаний!")
            return

        # Получаем предсказания и фактические значения
        # Обратите внимание: evaluate_model уже возвращает демасштабированные значения для 'price'
        # и бинарные для 'direction'/'breakout'.
        eval_results = self.evaluate_model()
        if eval_results is None:
            return

        if self.prediction_type == 'price':
            y_test_orig = eval_results['y_test_orig']
            y_pred_orig = eval_results['y_pred_orig']

            # Для визуализации берем только первую точку из предсказанного/фактического горизонта
            # Это упрощение, если prediction_horizon > 1
            if self.prediction_horizon > 1:
                y_test_plot = y_test_orig[:n_samples, 0] # Берем только первую точку горизонта
                y_pred_plot = y_pred_orig[:n_samples, 0] # Берем только первую точку горизонта
            else:
                y_test_plot = y_test_orig[:n_samples]
                y_pred_plot = y_pred_orig[:n_samples]

            # Если нужно показать train vs test, то также демасштабируем train
            if plot_train_vs_test and self.X_train is not None and len(self.X_train) > 0:
                # Получаем последние N_samples фактических значений из обучающего набора
                y_train_orig_full = self.price_scaler.inverse_transform(self.y_train.reshape(-1, 1)).reshape(self.y_train.shape)
                y_train_plot = y_train_orig_full[-n_samples:, 0] if self.prediction_horizon > 1 else y_train_orig_full[-n_samples:]

                # Получаем предсказания для части тренировочных данных
                # Убедитесь, что X_train достаточно большой для n_samples
                X_train_subset = self.X_train[-n_samples:] if len(self.X_train) >= n_samples else self.X_train
                y_train_pred_scaled = self.model.predict(X_train_subset, verbose=0)
                y_train_pred_plot_full = self.price_scaler.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).reshape(y_train_pred_scaled.shape)
                y_train_pred_plot = y_train_pred_plot_full[:, 0] if self.prediction_horizon > 1 else y_train_pred_plot_full.flatten()

                fig, axes = plt.subplots(3, 1, figsize=(15, 18))

                # График обучения
                axes[0].plot(self.history.history['loss'], label='Потери на обучении')
                if 'val_loss' in self.history.history:
                    axes[0].plot(self.history.history['val_loss'], label='Потери на валидации')
                axes[0].set_title('История обучения (Потери)')
                axes[0].set_xlabel('Эпоха')
                axes[0].set_ylabel('Потери')
                axes[0].legend()
                axes[0].grid(True)

                # График сравнения Train (последние n_samples)
                axes[1].plot(y_train_plot, label='Реальные (Обучение)', alpha=0.7)
                axes[1].plot(y_train_pred_plot, label='Предсказанные (Обучение)', alpha=0.7)
                axes[1].set_title(f'Сравнение предсказаний и реальных значений (часть тренировочных данных, последние {len(y_train_plot)} точек)')
                axes[1].set_xlabel('Время')
                axes[1].set_ylabel('Цена')
                axes[1].legend()
                axes[1].grid(True)

                # График сравнения Test
                axes[2].plot(y_test_plot, label='Реальные (Тест)', alpha=0.7)
                axes[2].plot(y_pred_plot, label='Предсказанные (Тест)', alpha=0.7)
                axes[2].set_title(f'Сравнение предсказаний и реальных значений (тестовые данные, первые {len(y_test_plot)} точек)')
                axes[2].set_xlabel('Время')
                axes[2].set_ylabel('Цена')
                axes[2].legend()
                axes[2].grid(True)

            else:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

                # График сравнения предсказаний (тест)
                ax1.plot(y_test_plot, label='Реальные значения', alpha=0.7)
                ax1.plot(y_pred_plot, label='Предсказанные значения', alpha=0.7)
                ax1.set_title(f'Сравнение предсказаний и реальных значений (первые {len(y_test_plot)} точек из тестового набора)')
                ax1.set_xlabel('Время')
                ax1.set_ylabel('Цена')
                ax1.legend()
                ax1.grid(True)

                # График ошибок (тест)
                errors = y_test_plot - y_pred_plot
                ax2.plot(errors, color='red', alpha=0.7)
                ax2.axhline(y=0, color='black', linestyle='--')
                ax2.set_title('Ошибки предсказаний (тестовый набор)')
                ax2.set_xlabel('Время')
                ax2.set_ylabel('Ошибка')
                ax2.grid(True)

        elif self.prediction_type in ['direction', 'breakout']:
            y_test_bin = eval_results['y_test_bin']
            y_pred_bin = eval_results['y_pred_bin']

            # Ограничиваем количество выборок для отображения
            y_test_plot = y_test_bin[:n_samples]
            y_pred_plot = y_pred_bin[:n_samples]

            fig, ax = plt.subplots(1, 1, figsize=(15, 6))

            # Сравнение бинарных предсказаний
            time_points = np.arange(len(y_test_plot))
            ax.scatter(time_points, y_test_plot, label='Реальные классы', alpha=0.7, marker='o')
            ax.scatter(time_points, y_pred_plot, label='Предсказанные классы', alpha=0.7, marker='x')
            ax.set_title(f'Сравнение бинарных предсказаний и реальных значений (первые {len(y_test_plot)} точек из тестового набора)')
            ax.set_xlabel('Время')
            ax.set_ylabel('Класс (0 или 1)')
            ax.set_yticks([0, 1])
            ax.legend()
            ax.grid(True)

        else:
            print(f"❌ Визуализация для типа прогноза '{self.prediction_type}' не реализована.")
            return

        plt.tight_layout()
        plt.show()
        # The method should return the figure object for external use if needed,
        # but the prompt asked for `return fig` at the very end, so I'll put it there.
        return fig
    
    
    def save_model(self, filepath: str, save_weights_only: bool = False):
        """
        Сохраняет обученную модель Keras.
        Args:
            filepath (str): Путь к файлу, куда будет сохранена модель.
                            Рекомендуется использовать расширение .keras
                            или .h5 для полной модели.
            save_weights_only (bool): Если True, сохраняет только веса модели.
                                      Если False, сохраняет всю модель (архитектуру + веса).
        """
        if self.model is None:
            print("Модель не обучена. Нечего сохранять.")
            return

        try:
            if save_weights_only:
                self.model.save_weights(filepath)
                print(f"Веса модели НС сохранены в '{filepath}'")
            else:
                self.model.save(filepath)
                print(f"Полная модель НС (архитектура + веса) сохранена в '{filepath}'")
        except Exception as e:
            print(f"Ошибка при сохранении модели: {e}")

    def load_model(self, filepath: str, load_weights_only: bool = False, custom_objects=None):
        """
        Загружает модель Keras из файла.
        Args:
            filepath (str): Путь к файлу модели.
            load_weights_only (bool): Если True, загружает только веса в существующую модель.
                                      Если False, загружает всю модель (архитектуру + веса).
            custom_objects (dict): Словарь пользовательских объектов (слоев, функций и т.д.),
                                   если они использовались в модели.
        Returns:
            tf.keras.Model: Загруженная модель.
        """
        try:
            if load_weights_only:
                if self.model is None:
                    print("Модель не инициализирована. Невозможно загрузить только веса.")
                    return None
                self.model.load_weights(filepath)
                print(f"Веса модели НС загружены из '{filepath}'")
                return self.model
            else:
                loaded_model = keras.models.load_model(filepath, custom_objects=custom_objects)
                self.model = loaded_model # Обновляем текущую модель экземпляра
                print(f"Полная модель НС загружена из '{filepath}'")
                return loaded_model
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            return None

         
    # Example (re-using the dummy data generation from our previous conversation for completeness):

# Пример использования:
if __name__ == "__main__":
    # Предполагаем, что 'dz' (DataFrame с OHLC данными) уже определен
    # и класс DaysLevel также определен
    # Создадим фиктивные данные, если нет реальных
    print(f"Тип индекса df: {type(df.index)}")
    print(f"Первые 5 значений индекса df: {df.index[:5]}")
    print(f"Есть ли столбец 'open_time' в df.columns? {'open_time' in df.columns}")
    df = df.set_index('Date')
    if 'dz' not in globals():
        dates = pd.date_range(start='2025-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'Open': np.random.rand(100) * 100,
            'High': np.random.rand(100) * 100 + 5,
            'Low': np.random.rand(100) * 100 - 5,
            'Close': np.random.rand(100) * 100,
            'Volume': np.random.rand(100) * 1000
        }, index=dates)
        dz = df.copy()
        dz.columns = [col.lower() for col in dz.columns] # Нормализация колонок

    # Создаем экземпляр DaysLevel
    days_level_instance = DaysLevel(dz, symbol='TEST', timeframe='1D')

    # Создаем экземпляр LevelAnalysisNN
    la_nn = LevelAnalysisNN(
        days_level=days_level_instance,
        top_n=5,
        sequence_length=30,
        prediction_horizon=1,
        prediction_type='price'
    )

    
    # Пример подготовки фиктивных данных для обучения (замените на ваш prepare_sequences)
    # Здесь просто создаются случайные данные нужной формы
    input_features = 5 # Количество признаков на шаг
    n_samples = 1000 # Общее количество шагов для обучения
    dummy_X = np.random.rand(n_samples, la_nn.sequence_length, input_features)
    dummy_y = np.random.rand(n_samples, 1) # Для предсказания цены

    la_nn.X_train = dummy_X[:800]
    la_nn.y_train = dummy_y[:800]
    la_nn.X_test = dummy_X[800:]
    la_nn.y_test = dummy_y[800:]

    # Строим модель
    la_nn.build_model(input_shape=(la_nn.sequence_length, input_features))

    # Обучаем модель
    la_nn.train(timeframes=['1D'], epochs=5) # Мало эпох для примера

    # Сохраняем полную модель
    model_filepath_full = 'my_neural_network_model.keras'
    la_nn.save_model(model_filepath_full, save_weights_only=False)

    # Сохраняем только веса
    model_filepath_weights = 'my_neural_network_weights.h5' # Можно использовать .h5 или .weights
    la_nn.save_model(model_filepath_weights, save_weights_only=True)

    print("\n--- Проверка загрузки ---")

    # Загружаем полную модель в новый экземпляр
    new_la_nn_full = LevelAnalysisNN(days_level_instance, top_n=5) # Создаем новый экземпляр
    loaded_model_full = new_la_nn_full.load_model(model_filepath_full)
    if loaded_model_full:
        print(f"Форма загруженной полной модели: {loaded_model_full.input_shape} -> {loaded_model_full.output_shape}")

    # Загружаем только веса в существующую (построенную) модель
    new_la_nn_weights = LevelAnalysisNN(days_level_instance, top_n=5)
    new_la_nn_weights.build_model(input_shape=(la_nn.sequence_length, input_features)) # Нужно сначала построить архитектуру
    loaded_model_weights = new_la_nn_weights.load_model(model_filepath_weights, load_weights_only=True)
    if loaded_model_weights:
        print("Веса успешно загружены в новую модель.")




'''
if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range(start='2025-01-01', periods=500, freq='H')
    data = pd.DataFrame({
        'date': dates,
        'open': np.random.rand(500) * 100 + 100,
        'high': np.random.rand(500) * 10 + 195,
        'low': np.random.rand(500) * 10 + 100,
        'close': np.random.rand(500) * 100 + 100,
        # Add other dummy features to match feature_columns_used in LevelAnalysisNN
        'price_change': np.random.randn(500),
        'volatility': np.random.rand(500) * 5,
        'momentum': np.random.randn(500),
        'momentum_fast': np.random.randn(500),
        'momentum_slow': np.random.randn(500),
        'sma_ratio': np.random.rand(500),
        'sma_ratio_long': np.random.rand(500),
        'rsi': np.random.rand(500) * 100,
        'stoch_k': np.random.rand(500) * 100,
        'stoch_d': np.random.rand(500) * 100,
        'ema_signal': np.random.rand(500),
        'support_distance': np.random.rand(500) * 10,
        'resistance_distance': np.random.rand(500) * 10,
        'support_strength': np.random.rand(500) * 10,
        'resistance_strength': np.random.rand(500) * 10,
        'level_density': np.random.rand(500),
        'level_type_cinching': np.random.randint(0, 2, 500),
        'level_type_mirror': np.random.randint(0, 2, 500),
        'level_type_change': np.random.randint(0, 2, 500),
        'level_type_paranorm': np.random.randint(0, 2, 500),
        'timeframe_weight': np.random.rand(500) * 5,
        'support_touch': np.random.randint(0, 2, 500),
        'resistance_touch': np.random.randint(0, 2, 500),
        'level_bounce': np.random.randint(0, 2, 500),
        'level_break': np.random.randint(0, 2, 500), # Essential for 'breakout' prediction_type
        'false_breakout': np.random.randint(0, 2, 500),
        'time_at_support': np.random.rand(500) * 10,
        'time_at_resistance': np.random.rand(500) * 10,
        'level_cluster_strength': np.random.rand(500) * 5,
        'level_cluster_count': np.random.randint(1, 10, 500),
        'level_cluster_range': np.random.rand(500) * 5,
        'h20_sma': np.random.rand(500) * 100 + 100,
        'h20_volatility': np.random.rand(500) * 5,
        'h20_momentum': np.random.randn(500),
        'price_to_h20_sma': np.random.rand(500),
        'h20_support_strength': np.random.rand(500) * 10,
        'h20_resistance_strength': np.random.rand(500) * 10,
        'h20_level_density': np.random.rand(500),
        'h20_interaction': np.random.rand(500),
        'h20_cluster_strength': np.random.rand(500) * 5
    })
    data = data.set_index('date')

    # Инициализация DaysLevel
    dl_instance = DaysLevel(data.copy(), symbol='TEST_SYMBOL')

    # Инициализация LevelAnalysisNN
    # Для демонстрации 'price' прогноза
    nn_analysis_price = LevelAnalysisNN(dl_instance, top_n=5, sequence_length=30,
                                        prediction_horizon=5, prediction_type='price')

    print("\n--- Обучение и построение графика для прогноза цен ---")
    nn_analysis_price.train(epochs=20, verbose=0, validation_split=0.2)
    nn_analysis_price.plot_predictions(n_samples=50, plot_train_vs_test=True)

    # Демонстрация прогноза 'direction'
    nn_analysis_direction = LevelAnalysisNN(dl_instance, top_n=5, sequence_length=30,
                                            prediction_horizon=5, prediction_type='direction')
    print("\n--- Обучение и построение графика для прогноза направления ---")
    nn_analysis_direction.train(epochs=20, verbose=0, validation_split=0.2)
    nn_analysis_direction.plot_predictions(n_samples=50)

    # Демонстрация прогноза 'breakout'
    nn_analysis_breakout = LevelAnalysisNN(dl_instance, top_n=5, sequence_length=30,
                                           prediction_horizon=5, prediction_type='breakout')
    print("\n--- Обучение и построение графика для прогноза пробоя ---")
    nn_analysis_breakout.train(epochs=20, verbose=0, validation_split=0.2)
    nn_analysis_breakout.plot_predictions(n_samples=50)
'''