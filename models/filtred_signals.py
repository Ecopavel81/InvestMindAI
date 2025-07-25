import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib # Для сохранения скейлеров

# Предполагаем использование TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model # load_model для загрузки

class TradingSignalGenerator:
    """
    Генератор торговых сигналов с определением точек входа/выхода
    и ранжированием по соотношению риск/прибыль
    """
    
    def __init__(self):
        self.tp_ratios = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Ранжирование ТП
        self.min_rr_ratio = 3.0  # Минимальное соотношение прибыль/риск для "горячих" сделок
        
    def calculate_entry_exit_points(self, current_price: float, predicted_price: float, 
                                  atri: float, level_info: Dict) -> Dict:
        """
        Расчет точек входа и выхода на основе предсказания и ATRI
        
        Args:
            current_price: текущая цена
            predicted_price: предсказанная цена
            atri: дневной ATRI
            level_info: информация об уровнях
            
        Returns:
            dict с точками входа/выхода и ранжированием
        """
        direction = 1 if predicted_price > current_price else -1
        price_change = abs(predicted_price - current_price)
        
        # Определение точки входа
        entry_price = current_price
        
        # Стоп-лосс на основе ATRI и близости к уровням
        if direction == 1:  # Long позиция
            # Стоп ниже ближайшей поддержки или на расстоянии ATRI
            support_level = current_price * (1 - level_info['nearest_support_dist'])
            atri_stop = current_price - atri
            stop_loss = min(support_level, atri_stop)
        else:  # Short позиция
            # Стоп выше ближайшего сопротивления или на расстоянии ATRI
            resistance_level = current_price * (1 + level_info['nearest_resistance_dist'])
            atri_stop = current_price + atri
            stop_loss = max(resistance_level, atri_stop)
        
        risk = abs(entry_price - stop_loss)
        
        # Расчет уровней тейк-профита
        take_profits = {}
        hot_deals = []
        
        for ratio in self.tp_ratios:
            if direction == 1:
                tp_price = entry_price + (risk * ratio)
            else:
                tp_price = entry_price - (risk * ratio)
            
            # Проверка достижимости на основе ATRI
            distance_to_tp = abs(tp_price - current_price)
            atri_multiplier = distance_to_tp / atri
            
            # Вероятность достижения ТП (эвристика)
            if atri_multiplier <= 1:
                probability = 0.8
            elif atri_multiplier <= 2:
                probability = 0.6
            elif atri_multiplier <= 3:
                probability = 0.4
            else:
                probability = 0.2
            
            take_profits[f"TP_{ratio}"] = {
                'price': tp_price,
                'ratio': ratio,
                'distance_atri': atri_multiplier,
                'probability': probability,
                'profit': risk * ratio,
                'rr_ratio': ratio
            }
            
            # Определение "горячих" сделок
            if ratio >= self.min_rr_ratio and probability >= 0.4:
                hot_deals.append({
                    'tp_level': f"TP_{ratio}",
                    'rr_ratio': ratio,
                    'probability': probability,
                    'profit_potential': risk * ratio
                })
        
        return {
            'direction': 'LONG' if direction == 1 else 'SHORT',
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'risk': risk,
            'take_profits': take_profits,
            'hot_deals': sorted(hot_deals, key=lambda x: x['rr_ratio'], reverse=True),
            'atri_daily': atri
        }
    
    def rank_signals(self, signals: List[Dict]) -> List[Dict]:
        """
        Ранжирование торговых сигналов по привлекательности
        
        Args:
            signals: список торговых сигналов
            
        Returns:
            отсортированный список сигналов
        """
        def calculate_signal_score(signal):
            # Базовый скор на основе количества горячих сделок
            hot_deals_count = len(signal['hot_deals'])
            base_score = hot_deals_count * 10
            
            # Бонус за высокое соотношение R/R
            if signal['hot_deals']:
                max_rr = max(deal['rr_ratio'] for deal in signal['hot_deals'])
                base_score += max_rr * 5
            
            # Бонус за высокую вероятность
            avg_probability = np.mean([deal['probability'] for deal in signal['hot_deals']]) if signal['hot_deals'] else 0
            base_score += avg_probability * 20
            
            return base_score
        
        for signal in signals:
            signal['score'] = calculate_signal_score(signal)
        
        return sorted(signals, key=lambda x: x['score'], reverse=True)


class CriptoPredictionNN:
    """
    Улучшенная нейросеть для предсказания цены с системой торговых сигналов
    """

    def __init__(self, days_level, lookback_period=20, prediction_horizon=5,
                 prediction_type='price', model_save_dir='models'):
        """
        Инициализация модели предсказания цены

        Args:
            days_level_analyzer: экземпляр класса DaysLevel
            lookback_period: количество периодов для анализа истории
            prediction_horizon: горизонт предсказания (количество периодов вперед)
            prediction_type: тип предсказания ('price' или 'change')
            model_save_dir: директория для сохранения моделей
        """
        self.days_level = days_level
        self.lookback_period = lookback_period
        self.prediction_horizon = prediction_horizon
        self.prediction_type = prediction_type
        self.is_trained = False
        self.model_save_dir = model_save_dir

        # Создаем директорию для моделей
        os.makedirs(model_save_dir, exist_ok=True)

        # Скейлеры для нормализации данных
        self.price_scaler = MinMaxScaler()
        self.feature_scaler = StandardScaler()

        # Модель
        self.model = None

        # История обучения
        self.training_history = None

        # Данные
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Генератор торговых сигналов
        self.signal_generator = TradingSignalGenerator()

    def _calculate_distance_to_levels(self, current_price: float, levels: List[Tuple]) -> Dict:
        """
        Расчет расстояния от текущей цены до различных уровней

        Args:
            current_price: текущая цена
            levels: список уровней [(date, level, strength, timeframe, type, direction)]

        Returns:
            dict с расстояниями и характеристиками уровней
        """
        if not levels:
            return {
                'nearest_support_dist': 0.01,
                'nearest_resistance_dist': 0.01,
                'nearest_support_strength': 5,
                'nearest_resistance_strength': 5,
                'nearest_mirror_dist': 0.02,
                'nearest_mirror_strength': 6,
                'total_levels_above': 3,
                'total_levels_below': 2,
                'weighted_support_strength': 15,
                'weighted_resistance_strength': 12,
            }

        supports = []
        resistances = []
        mirrors = []

        for level_data in levels:
            _, level, strength, timeframe, level_type, direction = level_data

            if level_type == 'mirror':
                mirrors.append((level, strength))
            elif level < current_price:
                supports.append((level, strength))
            else:
                resistances.append((level, strength))

        # Ближайшие уровни
        nearest_support = max(supports, key=lambda x: x[0]) if supports else (0, 0)
        nearest_resistance = min(resistances, key=lambda x: x[0]) if resistances else (999999, 0)
        nearest_mirror = min(mirrors, key=lambda x: abs(x[0] - current_price)) if mirrors else (0, 0)

        # Расчет расстояний
        support_dist = (current_price - nearest_support[0]) / current_price if nearest_support[0] > 0 else 0.05
        resistance_dist = (nearest_resistance[0] - current_price) / current_price if nearest_resistance[0] > 0 else 0.05
        mirror_dist = abs(nearest_mirror[0] - current_price) / current_price if nearest_mirror[0] > 0 else 0.02

        # Взвешенная сила уровней
        weighted_support = sum(s * (1 / (1 + abs(l - current_price))) for l, s in supports) if supports else 0
        weighted_resistance = sum(s * (1 / (1 + abs(l - current_price))) for l, s in resistances) if resistances else 0

        return {
            'nearest_support_dist': support_dist,
            'nearest_resistance_dist': resistance_dist,
            'nearest_support_strength': nearest_support[1],
            'nearest_resistance_strength': nearest_resistance[1],
            'nearest_mirror_dist': mirror_dist,
            'nearest_mirror_strength': nearest_mirror[1],
            'total_levels_above': len(resistances),
            'total_levels_below': len(supports),
            'weighted_support_strength': weighted_support,
            'weighted_resistance_strength': weighted_resistance
        }

    def _calculate_volatility_features(self, data: pd.DataFrame, window: int = 10) -> Dict:
        """
        Расчет характеристик волатильности

        Args:
            data: DataFrame с OHLC данными
            window: окно для расчета

        Returns:
            dict с характеристиками волатильности
        """
        if len(data) < window:
            return {
                'volatility_ratio': 0.03,
                'is_accumulation': False,
                'price_range_ratio': 0.05,
                'volume_trend': 0.01
            }

        # Средний ATRI за период
        avg_atri = data['atri'].rolling(window=window).mean().iloc[-1]
        current_volatility = data['atri'].iloc[-1]

        # Отношение текущей волатильности к средней
        volatility_ratio = current_volatility / avg_atri if avg_atri > 0 else 0

        # Проверка на накопление (волатильность < 0.5 ATRI)
        is_accumulation = volatility_ratio < 0.5

        # Диапазон цен
        price_range = (data['high'].iloc[-1] - data['low'].iloc[-1]) / data['close'].iloc[-1]
        avg_price_range = ((data['high'] - data['low']) / data['close']).rolling(window=window).mean().iloc[-1]
        price_range_ratio = price_range / avg_price_range if avg_price_range > 0 else 0

        # Тренд объема (если есть)
        volume_trend = 0
        if 'volume' in data.columns:
            recent_volume = data['volume'].iloc[-window:].mean()
            prev_volume = data['volume'].iloc[-2*window:-window].mean()
            volume_trend = (recent_volume - prev_volume) / prev_volume if prev_volume > 0 else 0

        return {
            'volatility_ratio': volatility_ratio,
            'is_accumulation': is_accumulation,
            'price_range_ratio': price_range_ratio,
            'volume_trend': volume_trend
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
        nearest_support_dist_atri = level_info['nearest_support_dist'] * current_price / daily_atri
        nearest_resistance_dist_atri = level_info['nearest_resistance_dist'] * current_price / daily_atri
        nearest_mirror_dist_atri = level_info['nearest_mirror_dist'] * current_price / daily_atri

        # Define thresholds in terms of ATRI for "nearness"
        NEAR_LEVEL_ATRI_THRESHOLD = 0.5 # e.g., within 0.5 ATRI
        
        near_support = nearest_support_dist_atri < NEAR_LEVEL_ATRI_THRESHOLD and level_info['nearest_support_strength'] > 0
        near_resistance = nearest_resistance_dist_atri < NEAR_LEVEL_ATRI_THRESHOLD and level_info['nearest_resistance_strength'] > 0
        near_mirror = nearest_mirror_dist_atri < NEAR_LEVEL_ATRI_THRESHOLD and level_info['nearest_mirror_strength'] > 0

        # --- Volatility and Consolidation Checks ---
        # Low volatility: This is already captured by volatility_info['is_accumulation']
        is_low_volatility = volatility_info['is_accumulation'] # True if volatility_ratio < 0.5

        # Check for "Dojis" (small body, long wicks) - Requires adding a feature for candle body size or shape
        # For simplicity, we'll assume `is_doji_pattern_recent` is a feature you might add.
        # is_doji_pattern_recent = self._check_doji_pattern(data, current_index, window=3) # Example
        
        # Low volume - requires a specific check over last 3-4 days
        # This also needs to be a feature calculated and passed in volatility_info or a new 'market_conditions' dict.
        # For now, let's assume `has_low_recent_volume` is a feature.
        has_low_recent_volume = volatility_info['volume_trend'] < 0.005 # Example: low positive or negative trend

        # --- Strategy Logic ---

        # Strategy 1: Price targeting nearest strong level (magnet effect)
        if price_change_abs < daily_atri * 2: # Only consider if predicted change is relatively small
            if near_support and predicted_price > current_price:
                return "Ожидаем отскок от поддержки к ближайшему уровню"
            if near_resistance and predicted_price < current_price:
                return "Ожидаем отскок от сопротивления к ближайшему уровню"

        # Strategy 2: Breakout from Consolidation (Low Volatility, Dojis, Low Volume)
        # Target: More than 3-4 ATRI move, potentially to a weekly level
        BREAKOUT_ATRI_TARGET = 3.5 # Minimum target for a breakout in ATRI multiples

        if is_low_volatility and has_low_recent_volume: # Add `is_doji_pattern_recent` if implemented
            target_level = None
            target_distance_atri = 0

            # Find the next significant weekly level in the predicted direction
            if predicted_price > current_price: # Bullish breakout
                resistances = [lvl for lvl in self.days_level.get_strongest_levels(timeframes=['1W']) if lvl[1] > current_price]
                if resistances:
                    target_level = min(resistances, key=lambda x: x[1]) # Nearest weekly resistance above
                    target_distance_atri = (target_level[1] - current_price) / daily_atri if daily_atri > 0 else 0
            elif predicted_price < current_price: # Bearish breakout
                supports = [lvl for lvl in self.days_level.get_strongest_levels(timeframes=['1W']) if lvl[1] < current_price]
                if supports:
                    target_level = max(supports, key=lambda x: x[1]) # Nearest weekly support below
                    target_distance_atri = (current_price - target_level[1]) / daily_atri if daily_atri > 0 else 0

            if target_level and target_distance_atri >= BREAKOUT_ATRI_TARGET and price_change_abs > BREAKOUT_ATRI_TARGET * daily_atri:
                # Check if predicted price moves towards this target significantly
                if (predicted_price > current_price and predicted_price <= target_level[1]) or \
                  (predicted_price < current_price and predicted_price >= target_level[1]):
                    return f"Прорыв из консолидации к недельному уровню {target_level[1]:.2f} ({target_distance_atri:.1f} ATRI)"

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
    
    
    def predict_with_trading_signals(self, data: pd.DataFrame, current_index: int = None) -> Dict:
        # ... (previous code) ...

        # Ensure ATRI column exists or calculate it for this specific data slice
        if 'atri' not in data.columns or pd.isna(data['atri'].iloc[current_index]):
            temp_data_for_atri = data.iloc[max(0, current_index - self.lookback_period - 14):current_index + 1].copy() # Need more data for ATRI calculation
            # Calculate ATRI on this temp_data_for_atri and assign it back to the original 'data' DataFrame at the current index
            # This is a bit tricky with partial calculation; ideally, ATRI is pre-calculated for the entire 'data'
            # For a robust solution, ensure 'atri' is fully pre-calculated on 'data' before calling this method.
            # As a temporary workaround, if it's missing, we'll try to calculate it for the window.
            if not temp_data_for_atri.empty:
                calculated_atri_series = self._calculate_atri(temp_data_for_atri)
                # Assign the last calculated ATRI value back to the original DataFrame
                if not calculated_atri_series.empty:
                    data.loc[data.index[current_index], 'atri'] = calculated_atri_series.iloc[-1]
                else:
                    data.loc[data.index[current_index], 'atri'] = 0.01 # Fallback
            else:
                data.loc[data.index[current_index], 'atri'] = 0.01 # Fallback


        daily_atri = data['atri'].iloc[current_index] if 'atri' in data.columns and not pd.isna(data['atri'].iloc[current_index]) else 0.01

        # ... (previous code) ...

        # Определение стратегии - NOW PASS daily_atri
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
    
    def _create_features(self, data: pd.DataFrame, index: int) -> np.ndarray:
        """
        Создание признаков для нейросети

        Args:
            data: DataFrame с данными
            index: индекс текущей позиции

        Returns:
            массив признаков
        """
        if index < self.lookback_period:
            return None

        # Исторические цены
        historical_prices = data['close'].iloc[index-self.lookback_period:index].values
        price_returns = np.diff(historical_prices) / historical_prices[:-1]

        # Текущая цена
        current_price = data['close'].iloc[index]

        # Получение уровней
        strongest_levels = self.days_level.get_strongest_levels(timeframes=['1H', '4H', '1D'], top_n=20)

        # Расчет расстояний до уровней
        level_features = self._calculate_distance_to_levels(current_price, strongest_levels)

        # Характеристики волатильности
        volatility_features = self._calculate_volatility_features(
            data.iloc[index-self.lookback_period:index+1]
        )

        # Технические индикаторы
        rsi = self._calculate_rsi(historical_prices)
        macd, macd_signal = self._calculate_macd(historical_prices)

        # Создание финального вектора признаков
        features = np.concatenate([
            # Исторические доходности
            price_returns,

            # Статистики цен
            [np.mean(price_returns), np.std(price_returns), np.min(price_returns), np.max(price_returns)],

            # Характеристики уровней
            [level_features['nearest_support_dist'],
             level_features['nearest_resistance_dist'],
             level_features['nearest_support_strength'],
             level_features['nearest_resistance_strength'],
             level_features['nearest_mirror_dist'],
             level_features['nearest_mirror_strength'],
             level_features['total_levels_above'],
             level_features['total_levels_below'],
             level_features['weighted_support_strength'],
             level_features['weighted_resistance_strength']],

            # Характеристики волатильности
            [volatility_features['volatility_ratio'],
             float(volatility_features['is_accumulation']),
             volatility_features['price_range_ratio'],
             volatility_features['volume_trend']],

            # Технические индикаторы
            [rsi, macd, macd_signal],

            # Дополнительные признаки
            [current_price / np.mean(historical_prices),  # Отношение к средней цене
             (current_price - np.min(historical_prices)) / (np.max(historical_prices) - np.min(historical_prices))]  # Позиция в диапазоне
        ])

        return features

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Расчет RSI"""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float]:
        """Расчет MACD"""
        if len(prices) < slow:
            return 0.0, 0.0

        ema_fast = self._ema(prices, fast)
        ema_slow = self._ema(prices, slow)

        macd = ema_fast - ema_slow

        # Для простоты используем простое скользящее среднее вместо EMA для сигнальной линии
        if len(prices) >= slow + signal:
            macd_line = []
            for i in range(slow-1, len(prices)):
                ema_f = self._ema(prices[:i+1], fast)
                ema_s = self._ema(prices[:i+1], slow)
                macd_line.append(ema_f - ema_s)

            macd_signal = np.mean(macd_line[-signal:])
        else:
            macd_signal = 0.0

        return macd, macd_signal

    def _ema(self, prices: np.ndarray, period: int) -> float:
        """Расчет экспоненциального скользящего среднего"""
        if len(prices) < period:
            return np.mean(prices)

        alpha = 2.0 / (period + 1)
        ema = prices[0]

        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema

        return ema

    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Подготовка данных для обучения

        Args:
            data: DataFrame с OHLC данными

        Returns:
            Tuple[X, y] - признаки и целевые значения
        """
        X, y = [], []

        for i in range(self.lookback_period, len(data) - self.prediction_horizon):
            # Создание признаков для текущей позиции
            features = self._create_features(data, i)
            if features is None:
                continue

            # Целевое значение
            current_price = data['close'].iloc[i]
            future_price = data['close'].iloc[i + self.prediction_horizon]

            if self.prediction_type == 'price':
                target = future_price
            else:  # change
                target = (future_price - current_price) / current_price

            X.append(features)
            y.append(target)

        return np.array(X), np.array(y)

    def create_model(self, input_shape: int) -> keras.Model:
        """
        Создание архитектуры нейросети

        Args:
            input_shape: размерность входного вектора

        Returns:
            модель Keras
        """
        model = keras.Sequential([
            # Входной слой
            layers.Input(shape=(input_shape,)),

            # Полносвязные слои
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            # Скрытые слои
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),

            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),

            layers.Dense(16, activation='relu'),

            # Выходной слой - один выход для одного предсказания
            layers.Dense(1, activation='linear')
        ])

        # Компиляция модели
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model

    def train(self, data: pd.DataFrame, epochs: int = 100, batch_size: int = 32,
              validation_split: float = 0.2, verbose: int = 1):
        """
        Обучение модели

        Args:
            data: DataFrame с данными
            epochs: количество эпох обучения
            batch_size: размер батча
            validation_split: доля данных для валидации
            verbose: уровень детализации вывода
        """
        print("Подготовка данных...")
        X, y = self.prepare_data(data)

        if len(X) == 0:
            raise ValueError("Недостаточно данных для обучения")

        print(f"Размер обучающей выборки: {X.shape}")
        print(f"Размер целевого вектора: {y.shape}")

        # Нормализация целевых значений
        if self.prediction_type == 'price':
            y = y.reshape(-1, 1)
            y_scaled = self.price_scaler.fit_transform(y)
        else:
            y_scaled = y.reshape(-1, 1)

        # Нормализация признаков
        X_scaled = self.feature_scaler.fit_transform(X)

        # Разделение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42, shuffle=False
        )

        # Сохраняем в атрибуты
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

        # Создание модели
        self.model = self.create_model(X.shape[1])

        # Callbacks для обучения
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            )
        ]

        # Обучение
        print("Начало обучения...")
        self.training_history = self.model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            callbacks=callbacks,
            batch_size=batch_size,
            verbose=verbose
        )

        # Оценка на тестовой выборке
        test_loss, test_mae = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nРезультаты на тестовой выборке:")
        print(f"MSE: {test_loss:.6f}")
        print(f"MAE: {test_mae:.6f}")

        # Предсказания на тестовой выборке
        y_pred = self.model.predict(X_test, verbose=0)

        # Дополнительные метрики
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"RMSE: {rmse:.6f}")

        self.is_trained = True

        # Автоматическое сохранение модели
        self.save_model_auto()

        return self.training_history

    def predict_with_trading_signals(self, data: pd.DataFrame, current_index: int = None) -> Dict:
        """
        Предсказание цены с генерацией торговых сигналов

        Args:
            data: DataFrame с данными
            current_index: индекс для предсказания (если None, используется последний)

        Returns:
            dict с предсказанием и торговыми сигналами
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена")

        if current_index is None:
            current_index = len(data) - 1

        # Создание признаков
        features = self._create_features(data, current_index)
        if features is None:
            raise ValueError("Недостаточно данных для создания признаков")

        # Нормализация
        features_scaled = self.feature_scaler.transform(features.reshape(1, -1))

        # Предсказание
        prediction = self.model.predict(features_scaled, verbose=0)[0][0]

        # Текущая цена
        current_price = data['close'].iloc[current_index]
        
        # Дневной ATRI
        daily_atri = data['atri'].iloc[current_index]

        if self.prediction_type == 'price':
            # Обратная нормализация для цены
            prediction_scaled = self.price_scaler.inverse_transform([[prediction]])
            predicted_price = prediction_scaled[0][0]
            predicted_change = (predicted_price - current_price) / current_price
        else:
            # Предсказание изменения
            predicted_change = prediction
            predicted_price = current_price * (1 + predicted_change)

        # Анализ уровней для интерпретации
        strongest_levels = self.days_level.get_strongest_levels(timeframes=['1H', '4H', '1D'], top_n=10)
        level_info = self._calculate_distance_to_levels(current_price, strongest_levels)
        volatility_info = self._calculate_volatility_features(data.iloc[current_index-10:current_index+1])

        # Генерация торговых сигналов
        trading_signals = self.signal_generator.calculate_entry_exit_points(
            current_price, predicted_price, daily_atri, level_info
        )

        # Определение стратегии
        strategy = self._determine_strategy(current_price, predicted_price, level_info, volatility_info)

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
                          level_info: Dict, volatility_info: Dict) -> str:
        """
        Определение торговой стратегии на основе предсказания

        Args:
            current_price: текущая цена
            predicted_price: предсказанная цена
            level_info: информация об уровнях
            volatility_info: информация о волатильности

        Returns:
            строка с описанием стратегии
        """
        price_change = predicted_price - current_price

        # Проверка на близость к уровням
        near_support = level_info['nearest_support_dist'] < 0.02  # В пределах 2%
        near_resistance = level_info['nearest_resistance_dist'] < 0.02
        near_mirror = level_info['nearest_mirror_dist'] < 0.02

        # Стратегия 1: Отбой от сильного уровня
        if near_support and level_info['nearest_support_strength'] > 5:
            if price_change > 0:
                return "Отбой от сильного уровня поддержки - покупка"
            else:
                return "Возможный пробой уровня поддержки - продажа"

        if near_resistance and level_info['nearest_resistance_strength'] > 5:
            if price_change < 0:
                return "Отбой от сильного уровня сопротивления - продажа"
            else:
                return "Возможный пробой уровня сопротивления - покупка"

        # Стратегия 2: Пробой уровня при низкой волатильности
        if volatility_info['is_accumulation']:
            if price_change > 0 and near_resistance:
                return "Пробой сопротивления при низкой волатильности - покупка"
            elif price_change < 0 and near_support:
                return "Пробой поддержки при низкой волатильности - продажа"

        # Стратегия 3: Зеркальный уровень
        if near_mirror and level_info['nearest_mirror_strength'] > 3:
            mirror_distance = level_info['nearest_mirror_dist'] * current_price
            if price_change > 0:
                return f"Отработка зеркального уровня вверх - цель +{mirror_distance:.2f}"
            else:
                return f"Отработка зеркального уровня вниз - цель -{mirror_distance:.2f}"

        # Общая стратегия
        if abs(price_change) > 0.02:  # Значительное движение > 2%
            direction = "покупка" if price_change > 0 else "продажа"
            return f"Значительное движение - {direction}"

        return "Боковое движение - ожидание"

    def _calculate_confidence(self, features_scaled: np.ndarray) -> float:
        """
        Расчет уверенности в предсказании (простая эвристика)

        Args:
            features_scaled: нормализованные признаки

        Returns:
            уровень уверенности от 0 до 1
        """
        # Для простоты используем обратную дисперсию признаков
        feature_variance = np.var(features_scaled)
        confidence = 1 / (1 + feature_variance)

        return min(max(confidence, 0.1), 0.9)  # Ограничиваем от 0.1 до 0.9

    def evaluate_model(self):
        """Оценка качества модели"""
        if self.model is None:
            print("❌ Модель не обучена!")
            return None

        # Предсказания на тестовой выборке
        y_pred = self.model.predict(self.X_test)
        y_test = self.y_test

        # Обратная нормализация только для цены
        if self.prediction_type == 'price':
            y_test_original = self.price_scaler.inverse_transform(y_test)
            y_pred_original = self.price_scaler.inverse_transform(y_pred)
        else:
            y_test_original = y_test
            y_pred_original = y_pred

        # Метрики
        mse = mean_squared_error(y_test_original, y_pred_original)
        mae = mean_absolute_error(y_test_original, y_pred_original)
        r2 = r2_score(y_test_original, y_pred_original)

        print("📈 ОЦЕНКА МОДЕЛИ:")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"R²: {r2:.4f}")

        # Направленность предсказаний
        if self.prediction_type == 'change':
            actual_directions = np.sign(y_test_original.flatten())
            pred_directions = np.sign(y_pred_original.flatten())
            direction_accuracy = np.mean(actual_directions == pred_directions)
            print(f"Точность направления: {direction_accuracy:.4f}")

        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'y_test': y_test_original,
            'y_pred': y_pred_original
        }

    def generate_trading_report(self, data: pd.DataFrame, last_n_predictions: int = 10) -> Dict:
        """
        Генерация отчета по торговым сигналам за последние N предсказаний
        
        Args:
            data: DataFrame с данными
            last_n_predictions: количество последних предсказаний для анализа
            
        Returns:
            dict с отчетом по торговым сигналам
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена")
        
        signals = []
        
        # Генерируем сигналы для последних N периодов
        start_idx = max(self.lookback_period, len(data) - last_n_predictions)
        
        for i in range(start_idx, len(data)):
            try:
                prediction_result = self.predict_with_trading_signals(data, i)
                prediction_result['timestamp'] = data.index[i] if hasattr(data, 'index') else i
                # When passing to rank_signals, we need the actual trading_signals dict
                # not the whole prediction_result if rank_signals expects 'hot_deals' at top level
                # As per previous fix, we need to pass the trading_signals part.
                
                # Make a copy to avoid modifying the original prediction_result if needed elsewhere
                signal_for_ranking = prediction_result['trading_signals'].copy() 
                signal_for_ranking['timestamp'] = prediction_result['timestamp'] # Add timestamp to it
                signals.append(signal_for_ranking) # Append the correct dict for ranking
                
            except Exception as e: # Catch specific exceptions or just general for debugging
                # Print the error for debugging, but continue
                print(f"Error generating signal for index {i}: {e}")
                continue
        
        # Ранжируем сигналы
        ranked_signals = self.signal_generator.rank_signals(signals)
        
        # Анализируем горячие сделки
        hot_deals_summary = {}
        total_hot_deals = 0
        
        for signal in ranked_signals: # 'signal' here is now the 'trading_signals' dict
            # Make sure 'hot_deals' exists, it might be empty
            hot_deals = signal.get('hot_deals', []) # Use .get with a default empty list
            total_hot_deals += len(hot_deals)
            
            for deal in hot_deals:
                rr_ratio = deal['rr_ratio']
                if rr_ratio not in hot_deals_summary:
                    hot_deals_summary[rr_ratio] = {
                        'count': 0,
                        'avg_probability': 0,
                        'total_profit_potential': 0
                    }
                
                hot_deals_summary[rr_ratio]['count'] += 1
                hot_deals_summary[rr_ratio]['avg_probability'] += deal['probability']
                hot_deals_summary[rr_ratio]['total_profit_potential'] += deal['profit_potential']
        
        # Вычисляем средние значения
        for rr_ratio in hot_deals_summary:
            count = hot_deals_summary[rr_ratio]['count']
            if count > 0: # Avoid division by zero
                hot_deals_summary[rr_ratio]['avg_probability'] /= count
        
        return {
            'total_signals': len(signals),
            'ranked_signals': ranked_signals,
            'hot_deals_summary': hot_deals_summary,
            'total_hot_deals': total_hot_deals,
            'avg_score': np.mean([s['score'] for s in ranked_signals]) if ranked_signals else 0
        }

    #Отчет
    def print_trading_report(self, data: pd.DataFrame, last_n_predictions: int = 10):
        """
        Генерирует и печатает отчет по торговым сигналам.
        
        Args:
            data: DataFrame с данными
            last_n_predictions: количество последних предсказаний для анализа
        """
        report = self.generate_trading_report(data, last_n_predictions)

        print("\n--- Trading Report ---")
        print(f"Total Signals Generated: {report['total_signals']}")
        print(f"Average Signal Score: {report['avg_score']:.2f}")

        print("\n--- Ranked Signals (Top 5) ---")
        # Ensure 'ranked_signals' exists and is iterable
        if 'ranked_signals' in report and report['ranked_signals']:
            for i, signal in enumerate(report['ranked_signals'][:5]):
                print(f"  Signal {i+1} (Score: {signal.get('score', 0):.2f}):")
                print(f"    Timestamp: {signal.get('timestamp', 'N/A')}")
                # Accessing directly from 'signal' now, since it's the trading_signals dict
                print(f"    Direction: {signal.get('direction', 'N/A')}")
                print(f"    Entry Price: {signal.get('entry_price', 'N/A'):.4f}")
                print(f"    Stop Loss: {signal.get('stop_loss', 'N/A'):.4f}")
                print(f"    Risk: {signal.get('risk', 'N/A'):.4f}")
                
                hot_deals = signal.get('hot_deals', [])
                if hot_deals:
                    print("    Hot Deals:")
                    for deal in hot_deals:
                        print(f"      - TP Level: {deal.get('tp_level', 'N/A')}, R/R: {deal.get('rr_ratio', 'N/A'):.2f}, Probability: {deal.get('probability', 'N/A'):.2f}, Profit Potential: {deal.get('profit_potential', 'N/A'):.4f}")
                else:
                    print("    No Hot Deals found for this signal.")
        else:
            print("  No ranked signals to display.")

        print("\n--- Hot Deals Summary ---")
        if 'hot_deals_summary' in report and report['hot_deals_summary']:
            for rr_ratio, summary in sorted(report['hot_deals_summary'].items()):
                print(f"  R/R Ratio {rr_ratio}:")
                print(f"    Count: {summary.get('count', 0)}")
                print(f"    Average Probability: {summary.get('avg_probability', 0):.2f}")
                print(f"    Total Profit Potential: {summary.get('total_profit_potential', 0):.4f}")
        else:
            print("  No hot deals summary to display.")

        print(f"\nTotal Hot Deals Identified: {report.get('total_hot_deals', 0)}")    

    def plot_trading_signals(self, data: pd.DataFrame, last_n_days: int = 30):
        """
        Визуализация торговых сигналов на графике цены
        
        Args:
            data: DataFrame с данными
            last_n_days: количество последних дней для отображения
        """
        if not self.is_trained:
            print("❌ Модель не обучена!")
            return
        
        # Получаем данные за последние N дней
        plot_data = data.iloc[-last_n_days:].copy()
        
        # Генерируем сигналы
        signals = []
        signal_indices = []
        
        for i in range(len(plot_data)):
            try:
                idx = len(data) - last_n_days + i
                if idx >= self.lookback_period:
                    signal = self.predict_with_trading_signals(data, idx)
                    signals.append(signal)
                    signal_indices.append(i)
            except:
                continue
        
        # Создаем график
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # График цены с сигналами
        ax1.plot(plot_data.index, plot_data['close'], label='Цена закрытия', linewidth=2)
        
        # Отмечаем точки входа
        for i, signal in enumerate(signals):
            idx = signal_indices[i]
            trading_signal = signal['trading_signals']
            
            if trading_signal['direction'] == 'LONG':
                ax1.scatter(plot_data.index[idx], signal['current_price'], 
                           color='green', marker='^', s=100, alpha=0.7)
                # Стоп-лосс
                ax1.scatter(plot_data.index[idx], trading_signal['stop_loss'], 
                           color='red', marker='v', s=50, alpha=0.7)
            else:
                ax1.scatter(plot_data.index[idx], signal['current_price'], 
                           color='red', marker='v', s=100, alpha=0.7)
                # Стоп-лосс
                ax1.scatter(plot_data.index[idx], trading_signal['stop_loss'], 
                           color='green', marker='^', s=50, alpha=0.7)
        
        ax1.set_title('Торговые сигналы на графике цены')
        ax1.set_ylabel('Цена USD')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # График ATRI
        ax2.plot(plot_data.index, plot_data['atri'], label='ATRI', color='orange')
        ax2.set_title('Дневной ATRI')
        ax2.set_ylabel('ATRI')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # График соотношения R/R для горячих сделок
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
        
        ax3.bar(timestamps, hot_deals_data, alpha=0.7, color='purple')
        ax3.set_title('Максимальное соотношение R/R для горячих сделок')
        ax3.set_ylabel('R/R Ratio')
        ax3.axhline(y=3, color='red', linestyle='--', alpha=0.7, label='Минимум для горячих сделок')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig

    def plot_training_history(self):
        """Визуализация истории обучения"""
        if self.training_history is None:
            print("Модель не обучена")
            return

        plt.figure(figsize=(12, 4))

        # График потерь
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history.history['loss'], label='Training Loss')
        plt.plot(self.training_history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # График MAE
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
        """Визуализация предсказаний"""
        if self.model is None:
            print("❌ Модель не обучена!")
            return

        # Получаем предсказания
        eval_results = self.evaluate_model()
        y_test = eval_results['y_test'][:n_samples]
        y_pred = eval_results['y_pred'][:n_samples]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # График сравнения предсказаний
        ax1.plot(y_test.flatten(), label='Actual', alpha=0.7)
        ax1.plot(y_pred.flatten(), label='Predicted', alpha=0.7)
        ax1.set_title(f'Сравнение предсказаний и реальных значений (первые {n_samples} точек)')
        ax1.set_xlabel('Время')
        ax1.set_ylabel('Цена')
        ax1.legend()
        ax1.grid(True)

        # График ошибок
        errors = y_test.flatten() - y_pred.flatten()
        ax2.plot(errors, color='red', alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='--')
        ax2.set_title('Ошибки предсказаний')
        ax2.set_xlabel('Время')
        ax2.set_ylabel('Ошибка')
        ax2.grid(True)

        plt.tight_layout()
        return fig

    def save_model_auto(self):
        """Автоматическое сохранение модели с временной меткой"""
        if self.model is None:
            raise ValueError("Модель не создана")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"crypto_model_{timestamp}"
        
        self.save_model(model_name)

    def save_model(self, model_name: str):
        """
        Сохранение модели и всех связанных компонентов
        
        Args:
            model_name: имя модели для сохранения
        """
        if self.model is None:
            raise ValueError("Модель не создана")

        model_dir = os.path.join(self.model_save_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        # Сохраняем Keras модель
        keras_path = os.path.join(model_dir, "keras_model.h5")
        self.model.save(keras_path)

        # Сохраняем скейлеры
        scalers_path = os.path.join(model_dir, "scalers.pkl")
        scalers_data = {
            'price_scaler': self.price_scaler,
            'feature_scaler': self.feature_scaler
        }
        with open(scalers_path, 'wb') as f:
            pickle.dump(scalers_data, f)

        # Сохраняем параметры модели
        params_path = os.path.join(model_dir, "model_params.pkl")
        model_params = {
            'lookback_period': self.lookback_period,
            'prediction_horizon': self.prediction_horizon,
            'prediction_type': self.prediction_type,
            'is_trained': self.is_trained
        }
        with open(params_path, 'wb') as f:
            pickle.dump(model_params, f)

        # Сохраняем историю обучения (если есть)
        if self.training_history is not None:
            history_path = os.path.join(model_dir, "training_history.pkl")
            with open(history_path, 'wb') as f:
                pickle.dump(self.training_history.history, f)

        print(f"✅ Модель сохранена в {model_dir}")
        return model_dir

    def load_model(self, model_name: str):
        """
        Загрузка модели и всех связанных компонентов
        
        Args:
            model_name: имя модели для загрузки
        """
        model_dir = os.path.join(self.model_save_dir, model_name)
        
        if not os.path.exists(model_dir):
            raise ValueError(f"Модель {model_name} не найдена в {self.model_save_dir}")

        # Загружаем Keras модель
        keras_path = os.path.join(model_dir, "keras_model.h5")
        self.model = keras.models.load_model(keras_path)

        # Загружаем скейлеры
        scalers_path = os.path.join(model_dir, "scalers.pkl")
        with open(scalers_path, 'rb') as f:
            scalers_data = pickle.load(f)
            self.price_scaler = scalers_data['price_scaler']
            self.feature_scaler = scalers_data['feature_scaler']

        # Загружаем параметры модели
        params_path = os.path.join(model_dir, "model_params.pkl")
        with open(params_path, 'rb') as f:
            model_params = pickle.load(f)
            self.lookback_period = model_params['lookback_period']
            self.prediction_horizon = model_params['prediction_horizon']
            self.prediction_type = model_params['prediction_type']
            self.is_trained = model_params['is_trained']

        # Загружаем историю обучения (если есть)
        history_path = os.path.join(model_dir, "training_history.pkl")
        if os.path.exists(history_path):
            with open(history_path, 'rb') as f:
                history_dict = pickle.load(f)
                # Создаем объект истории (упрощенная версия)
                class HistoryWrapper:
                    def __init__(self, history_dict):
                        self.history = history_dict
                self.training_history = HistoryWrapper(history_dict)

        print(f"✅ Модель загружена из {model_dir}")

    def list_saved_models(self) -> List[str]:
        """
        Получение списка сохраненных моделей
        
        Returns:
            список имен сохраненных моделей
        """
        if not os.path.exists(self.model_save_dir):
            return []
        
        models = []
        for item in os.listdir(self.model_save_dir):
            model_path = os.path.join(self.model_save_dir, item)
            if os.path.isdir(model_path):
                # Проверяем, что это действительно сохраненная модель
                required_files = ["keras_model.h5", "scalers.pkl", "model_params.pkl"]
                if all(os.path.exists(os.path.join(model_path, f)) for f in required_files):
                    models.append(item)
        
        return sorted(models)

    def generate_trading_report(self, data: pd.DataFrame, last_n_predictions: int = 10) -> Dict:
        """
        Генерация отчета по торговым сигналам за последние N предсказаний
        
        Args:
            data: DataFrame с данными
            last_n_predictions: количество последних предсказаний для анализа
            
        Returns:
            dict с отчетом по торговым сигналам
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена")
        
        signals_for_ranking = [] # This will store the actual trading_signals dictionaries
        all_prediction_results = [] # Keep this if you need the full results for other parts

        # Генерируем сигналы для последних N периодов
        start_idx = max(self.lookback_period, len(data) - last_n_predictions)
        
        for i in range(start_idx, len(data)):
            try:
                prediction_result = self.predict_with_trading_signals(data, i)
                # Ensure the timestamp is added to the full result if needed elsewhere
                prediction_result['timestamp'] = data.index[i] if hasattr(data, 'index') else i
                all_prediction_results.append(prediction_result)

                # Extract the 'trading_signals' part and add timestamp to it for ranking if needed
                # The rank_signals function seems to expect 'hot_deals' directly,
                # so we should pass the 'trading_signals' dictionary.
                trading_signal_data = prediction_result['trading_signals']
                trading_signal_data['timestamp'] = prediction_result['timestamp'] # Add timestamp to the signal data
                trading_signal_data['score'] = 0 # Initialize score, will be updated by rank_signals
                signals_for_ranking.append(trading_signal_data)

            except Exception as e: # Catch specific exceptions or just general for debugging
                print(f"Error generating signal for index {i}: {e}")
                continue
        
        # Ранжируем сигналы. Now signals_for_ranking contains dictionaries directly consumable by rank_signals
        ranked_signals = self.signal_generator.rank_signals(signals_for_ranking)
        
        # Анализируем горячие сделки
        hot_deals_summary = {}
        total_hot_deals = 0
        
        for signal in ranked_signals: # These are the 'trading_signals' dicts from before
            hot_deals = signal['hot_deals'] # Now 'hot_deals' is directly accessible
            total_hot_deals += len(hot_deals)
            
            for deal in hot_deals:
                rr_ratio = deal['rr_ratio']
                if rr_ratio not in hot_deals_summary:
                    hot_deals_summary[rr_ratio] = {
                        'count': 0,
                        'avg_probability': 0,
                        'total_profit_potential': 0
                    }
                
                hot_deals_summary[rr_ratio]['count'] += 1
                hot_deals_summary[rr_ratio]['avg_probability'] += deal['probability']
                hot_deals_summary[rr_ratio]['total_profit_potential'] += deal['profit_potential']
        
        # Вычисляем средние значения
        for rr_ratio in hot_deals_summary:
            count = hot_deals_summary[rr_ratio]['count']
            if count > 0: # Avoid division by zero
                hot_deals_summary[rr_ratio]['avg_probability'] /= count
            
        return {
            'total_signals': len(signals_for_ranking), # Use the count of signals actually ranked
            'ranked_signals': ranked_signals,
            'hot_deals_summary': hot_deals_summary,
            'total_hot_deals': total_hot_deals,
            'avg_score': np.mean([s['score'] for s in ranked_signals]) if ranked_signals else 0
        }


# == Заменитель DaysLevel для тестирования ==
class MockDaysLevel:
    def __init__(self, data):
        self.data = data
        self.timeframe = '1D'

    def get_strongest_levels(self, timeframes=None, top_n=10):
        # Возвращаем фиктивные уровни на основе исторических максимумов/минимумов
        current_price = self.data['close'].iloc[-1]
        
        # Получаем последние 50 свечей для анализа
        recent_data = self.data.iloc[-50:]
        
        levels = []
        
        # Добавляем уровни на основе локальных максимумов и минимумов
        highs = recent_data['high'].rolling(5).max()
        lows = recent_data['low'].rolling(5).min()
        
        # Находим значимые уровни
        for i in range(len(recent_data)):
            if i < 2 or i >= len(recent_data) - 2:
                continue
                
            price = recent_data['close'].iloc[i]
            high = recent_data['high'].iloc[i]
            low = recent_data['low'].iloc[i]
            
            # Проверяем, является ли это локальным максимумом/минимумом
            is_high = (high >= recent_data['high'].iloc[i-2:i+3].max())
            is_low = (low <= recent_data['low'].iloc[i-2:i+3].min())
            
            if is_high:
                strength = np.random.uniform(3, 8)
                levels.append((
                    recent_data.index[i],
                    high,
                    strength,
                    '1H',
                    'resistance',
                    'up'
                ))
            
            if is_low:
                strength = np.random.uniform(3, 8)
                levels.append((
                    recent_data.index[i],
                    low,
                    strength,
                    '1H',
                    'support',
                    'down'
                ))
        
        # Добавляем зеркальные уровни
        for _ in range(3):
            level = current_price * (1 + np.random.normal(0, 0.03))
            strength = np.random.uniform(4, 7)
            levels.append((
                recent_data.index[-1],
                level,
                strength,
                '4H',
                'mirror',
                'neutral'
            ))
        
        # Сортируем по силе и возвращаем топ N
        levels.sort(key=lambda x: x[2], reverse=True)
        return levels[:top_n]


# == Пример использования улучшенной системы ==
def example_usage():
    """
    Пример использования улучшенной системы предсказания цены с торговыми сигналами
    """
    print("🚀 Запуск улучшенной системы торговых сигналов...")
    
    # Загрузка данных (замените на ваш путь)
    csv_file_path = '/mnt/c/Users/ecopa/Desktop/Proekts/Trader bot/kandles.csv'
    
    try:
        # Загружаем реальные данные
        def load_data_from_csv(csv_file_path, start_date=None, end_date=None):
            data = pd.read_csv(csv_file_path, parse_dates=['open_time'])
            data.rename(columns={'open_time': 'Date', 'close': 'Close'}, inplace=True)
            data.sort_values('Date', inplace=True)
            
            if start_date:
                data = data[data['Date'] >= pd.to_datetime(start_date)]
            if end_date:
                data = data[data['Date'] <= pd.to_datetime(end_date)]
            
            return data
        
        df = load_data_from_csv(csv_file_path, start_date='2024-11-01', end_date='2025-06-05')
        data = df.copy()
        
    except:
        print("⚠️ Не удалось загрузить данные, создаем синтетические...")
        # Создание тестовых данных
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=1000, freq='1D')

        # Генерация синтетических OHLC данных
        price = 45000  # Начальная цена BTC
        prices = [price]

        for i in range(999):
            change = np.random.normal(0, 0.025)  # 2.5% стандартное отклонение
            price *= (1 + change)
            prices.append(price)

        # Создание OHLC данных
        data = pd.DataFrame(index=dates)
        data['close'] = prices
        data['open'] = data['close'].shift(1) * (1 + np.random.normal(0, 0.005, len(data)))
        data['high'] = np.maximum(data['open'], data['close']) * (1 + np.abs(np.random.normal(0, 0.015, len(data))))
        data['low'] = np.minimum(data['open'], data['close']) * (1 - np.abs(np.random.normal(0, 0.015, len(data))))
        data['volume'] = np.random.uniform(1000, 10000, len(data))
        data = data.dropna()

    # Добавляем ATRI (Average True Range Index)
    def calculate_atri(data, period=14):
        high_low = data['high'] - data['low']
        high_close_prev = np.abs(data['high'] - data['close'].shift(1))
        low_close_prev = np.abs(data['low'] - data['close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        atri = true_range.rolling(window=period).mean()
        
        return atri
    
    data['atri'] = calculate_atri(data)
    data = data.dropna()

    print(f"📊 Загружено {len(data)} записей данных")

    # Создание анализатора уровней
    days_level = MockDaysLevel(data)

    print("🧠 Создание модели предсказания...")
    predictor = CriptoPredictionNN(
        days_level=days_level,
        lookback_period=30,
        prediction_horizon=5,
        prediction_type='price',
        model_save_dir='crypto_models'
    )

    print("📚 Обучение модели...")
    try:
        history = predictor.train(data, epochs=100, batch_size=32, verbose=1)

        print("\n🔮 Генерация торговых сигналов...")
        
        # Получаем предсказание с торговыми сигналами
        prediction = predictor.predict_with_trading_signals(data)

        print("\n" + "="*70)
        print("📈 РЕЗУЛЬТАТ АНАЛИЗА")
        print("="*70)
        print(f"💰 Текущая цена: ${prediction['current_price']:.2f}")
        print(f"🎯 Предсказанная цена: ${prediction['predicted_price']:.2f}")
        print(f"📊 Ожидаемое изменение: {prediction['predicted_change_percent']:.2f}%")
        print(f"📋 Стратегия: {prediction['strategy']}")
        print(f"🎲 Уверенность: {prediction['confidence']:.2f}")
        print(f"📏 Дневной ATRI: ${prediction['daily_atri']:.2f}")

        # Торговые сигналы
        ts = prediction['trading_signals']
        print(f"\n🎯 ТОРГОВЫЕ СИГНАЛЫ:")
        print(f"   Направление: {ts['direction']}")
        print(f"   Точка входа: ${ts['entry_price']:.2f}")
        print(f"   Стоп-лосс: ${ts['stop_loss']:.2f}")
        print(f"   Риск: ${ts['risk']:.2f}")

        print(f"\n🔥 ГОРЯЧИЕ СДЕЛКИ (R/R ≥ 3:1):")
        for i, deal in enumerate(ts['hot_deals'][:5]):  # Показываем топ-5
            print(f"   {i+1}. {deal['tp_level']}: R/R {deal['rr_ratio']}:1, "
                  f"вероятность: {deal['probability']:.2f}, "
                  f"потенциал: ${deal['profit_potential']:.2f}")

        print(f"\n📊 ВСЕ УРОВНИ ТЕЙК-ПРОФИТА:")
        for tp_name, tp_data in ts['take_profits'].items():
            print(f"   {tp_name}: ${tp_data['price']:.2f} "
                  f"(R/R {tp_data['ratio']}:1, "
                  f"расстояние: {tp_data['distance_atri']:.1f} ATRI, "
                  f"вероятность: {tp_data['probability']:.2f})")

        # Генерируем отчет по торговым сигналам
        predictor.print_trading_report(data, last_n_predictions=20)

        # Визуализация
        print("\n📊 Создание графиков...")
        
        # История обучения
        predictor.plot_training_history()
        
        # Предсказания vs реальность
        predictor.plot_predictions(n_samples=200)
        
        # Торговые сигналы на графике
        predictor.plot_trading_signals(data, last_n_days=60)

        # Список сохраненных моделей
        saved_models = predictor.list_saved_models()
        print(f"\n💾 Сохраненные модели: {saved_models}")

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    example_usage()