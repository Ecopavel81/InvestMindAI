import time
import functools
import logging
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from tf.keras.models import Sequential
from tf.keras.layers import LSTM, Dense, Dropout
import warnings
import pickle
from datetime import datetime
warnings.filterwarnings('ignore')

# Импортируем рефакторингованный TradingSignalGenerator
# from your_module import TradingSignalGenerator, StopLossCalculator, TakeProfitCalculator, SignalRanker

# Для демонстрации, используем заглушки для внешних зависимостей
class DaysLevel:
    def get_strongest_levels(self, timeframes: List[str], top_n: int) -> List[Tuple]:
        # Заглушка для получения уровней
        # return [(date, level, strength, timeframe, type, direction)]
        return [
            (pd.Timestamp.now(), 1000, 7, '1D', 'support', 'up'),
            (pd.Timestamp.now(), 1050, 8, '1D', 'resistance', 'down'),
            (pd.Timestamp.now(), 1025, 6, '4H', 'mirror', 'both')
        ]

# Заглушки для рефакторингованных классов TradingSignalGenerator
class StopLossCalculator:
    def calculate_stop_loss(self, current_price: float, direction: int, atri: float, level_info: Dict) -> float:
        return current_price * 0.98 if direction == 1 else current_price * 1.02 # Пример
class TakeProfitCalculator:
    def __init__(self, tp_ratios: List[int], min_rr_ratio: float): pass
    def calculate_take_profits(self, entry_price: float, risk: float, direction: int, atri: float) -> Tuple[Dict, List[Dict]]:
        return {}, [] # Пример
class SignalRanker:
    def calculate_signal_score(self, signal: Dict) -> float: return 0.0 # Пример
    def rank_signals(self, signals: List[Dict]) -> List[Dict]: return signals # Пример

class TradingSignalGenerator:
    def __init__(self, tp_ratios: List[int] = None, min_rr_ratio: float = 3.0):
        self.stop_loss_calculator = StopLossCalculator()
        self.take_profit_calculator = TakeProfitCalculator(tp_ratios, min_rr_ratio)
        self.signal_ranker = SignalRanker()
    def generate_signal(self, current_price: float, predicted_price: float, atri: float, level_info: Dict) -> Dict:
        # Упрощенная заглушка
        return {'direction': 'LONG', 'entry_price': current_price, 'predicted_price': predicted_price, 'stop_loss': 0, 'risk': 0, 'take_profits': {}, 'hot_deals': [], 'atri_daily': atri}
    def rank_signals(self, signals: List[Dict]) -> List[Dict]:
        return self.signal_ranker.rank_signals(signals)

# ---
## 1. Вспомогательные классы для расчетов точек
# ---

class StopLossCalculator:
    """
    Класс для расчета стоп-лосса на основе текущих данных.
    """
    def calculate_stop_loss(self, current_price: float, direction: int, atri: float, level_info: Dict) -> float:
        """
        Рассчитывает цену стоп-лосса.

        Args:
            current_price: Текущая цена.
            direction: Направление сделки (1 для LONG, -1 для SHORT).
            atri: Дневной ATRI.
            level_info: Информация об уровнях поддержки/сопротивления.

        Returns:
            Цена стоп-лосса.
        """
        if direction == 1:  # Long позиция
            # Стоп ниже ближайшей поддержки или на расстоянии ATRI
            support_level = current_price * (1 - level_info.get('nearest_support_dist', 0))
            atri_stop = current_price - atri
            stop_loss = min(support_level, atri_stop)
        else:  # Short позиция
            # Стоп выше ближайшего сопротивления или на расстоянии ATRI
            resistance_level = current_price * (1 + level_info.get('nearest_resistance_dist', 0))
            atri_stop = current_price + atri
            stop_loss = max(resistance_level, atri_stop)
        return stop_loss

class TakeProfitCalculator:
    """
    Класс для расчета уровней тейк-профита и их вероятностей.
    """
    def __init__(self, tp_ratios: List[int], min_rr_ratio: float):
        self.tp_ratios = tp_ratios
        self.min_rr_ratio = min_rr_ratio

    def calculate_take_profits(self, entry_price: float, risk: float, direction: int, atri: float) -> Tuple[Dict, List[Dict]]:
        """
        Рассчитывает уровни тейк-профита и определяет "горячие" сделки.

        Args:
            entry_price: Цена входа.
            risk: Размер риска (расстояние до стоп-лосса).
            direction: Направление сделки (1 для LONG, -1 для SHORT).
            atri: Дневной ATRI.

        Returns:
            Кортеж из словаря take_profits и списка hot_deals.
        """
        take_profits = {}
        hot_deals = []

        for ratio in self.tp_ratios:
            if direction == 1:
                tp_price = entry_price + (risk * ratio)
            else:
                tp_price = entry_price - (risk * ratio)

            distance_to_tp = abs(tp_price - entry_price) # Изменил с current_price на entry_price
            
            # Избегаем деления на ноль, если ATRI очень мал или 0
            atri_multiplier = distance_to_tp / atri if atri > 0 else float('inf')

            # Вероятность достижения ТП (эвристика) - можно вынести в отдельную функцию или стратегию
            probability = 0.0
            if atri_multiplier <= 1:
                probability = 0.8
            elif atri_multiplier <= 2:
                probability = 0.6
            elif atri_multiplier <= 3:
                probability = 0.4
            else:
                probability = 0.2
            
            # Если TP цена находится за предсказанной ценой, уменьшаем вероятность.
            # Эта логика должна быть дополнительно продумана, это просто пример
            # if direction == 1 and tp_price > predicted_price:
            #     probability *= 0.5 # Штраф за выход за предсказанный диапазон
            # elif direction == -1 and tp_price < predicted_price:
            #     probability *= 0.5 # Штраф за выход за предсказанный диапазон


            take_profits[f"TP_{ratio}"] = {
                'price': tp_price,
                'ratio': ratio,
                'distance_atri': atri_multiplier,
                'probability': probability,
                'profit': risk * ratio,
                'rr_ratio': ratio
            }

            if ratio >= self.min_rr_ratio and probability >= 0.4:
                hot_deals.append({
                    'tp_level': f"TP_{ratio}",
                    'rr_ratio': ratio,
                    'probability': probability,
                    'profit_potential': risk * ratio
                })
        return take_profits, sorted(hot_deals, key=lambda x: x['rr_ratio'], reverse=True)


# ---
## 2. Класс для ранжирования сигналов
# ---

class SignalRanker:
    """
    Класс для ранжирования торговых сигналов по привлекательности.
    """
    def calculate_signal_score(self, signal: Dict) -> float:
        """
        Вычисляет скор для отдельного торгового сигнала.
        """
        # Базовый скор на основе количества горячих сделок
        hot_deals_count = len(signal.get('hot_deals', []))
        base_score = hot_deals_count * 10

        # Бонус за высокое соотношение R/R
        if signal.get('hot_deals'):
            # Убедимся, что hot_deals не пустой перед вызовом max()
            max_rr = max(deal['rr_ratio'] for deal in signal['hot_deals'])
            base_score += max_rr * 5

        # Бонус за высокую вероятность
        avg_probability = np.mean([deal['probability'] for deal in signal.get('hot_deals', [])]) if signal.get('hot_deals') else 0
        base_score += avg_probability * 20

        # Добавим небольшой штраф за низкий ATRI, если это нежелательно (пример)
        # if signal['atri_daily'] < some_threshold:
        #     base_score -= 5

        return base_score

    def rank_signals(self, signals: List[Dict]) -> List[Dict]:
        """
        Ранжирует список торговых сигналов.

        Args:
            signals: Список торговых сигналов.

        Returns:
            Отсортированный список сигналов по убыванию скора.
        """
        for signal in signals:
            signal['score'] = self.calculate_signal_score(signal)

        return sorted(signals, key=lambda x: x['score'], reverse=True)


# ---
## 3. Обновленный класс TradingSignalGenerator (Композиция)
# ---

class TradingSignalGenerator:
    """
    Генератор торговых сигналов с определением точек входа/выхода
    и ранжированием по соотношению риск/прибыль.
    Использует композицию для делегирования ответственности.
    """

    def __init__(self, tp_ratios: List[int] = None, min_rr_ratio: float = 3.0):
        self.tp_ratios = tp_ratios if tp_ratios is not None else [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.min_rr_ratio = min_rr_ratio
        
        # Композиция: делегируем расчеты
        self.stop_loss_calculator = StopLossCalculator()
        self.take_profit_calculator = TakeProfitCalculator(self.tp_ratios, self.min_rr_ratio)
        self.signal_ranker = SignalRanker() # Класс для ранжирования

    def generate_signal(self, current_price: float, predicted_price: float,
                        atri: float, level_info: Dict) -> Dict:
        """
        Генерирует один торговый сигнал.

        Args:
            current_price: текущая цена
            predicted_price: предсказанная цена
            atri: дневной ATRI
            level_info: информация об уровнях

        Returns:
            dict с точками входа/выхода и ранжированием
        """
        direction = 1 if predicted_price > current_price else -1
        # price_change = abs(predicted_price - current_price) # Не используется, можно удалить

        entry_price = current_price

        # Делегируем расчет стоп-лосса
        stop_loss = self.stop_loss_calculator.calculate_stop_loss(current_price, direction, atri, level_info)
        risk = abs(entry_price - stop_loss)
        
        # Если риск равен 0 (например, stop_loss совпал с entry_price),
        # это может вызвать проблемы при расчете take_profits.
        # Необходимо обработать этот краевой случай.
        if risk == 0:
            # Можно вернуть пустой сигнал или поднять исключение,
            # или установить минимальный риск, если это имеет смысл в вашей логике.
            print("Предупреждение: Риск равен нулю. Невозможно рассчитать TP.")
            return {
                'direction': 'LONG' if direction == 1 else 'SHORT',
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'risk': risk,
                'take_profits': {},
                'hot_deals': [],
                'atri_daily': atri,
                'score': 0 # Добавляем скор, даже если пустой сигнал
            }


        # Делегируем расчет тейк-профитов и "горячих" сделок
        take_profits, hot_deals = self.take_profit_calculator.calculate_take_profits(
            entry_price, risk, direction, atri
        )

        signal_data = {
            'direction': 'LONG' if direction == 1 else 'SHORT',
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'risk': risk,
            'take_profits': take_profits,
            'hot_deals': hot_deals,
            'atri_daily': atri
        }
        
        # Рассчитываем скор для этого конкретного сигнала, чтобы он был доступен сразу
        signal_data['score'] = self.signal_ranker.calculate_signal_score(signal_data)

        return signal_data

    def rank_signals(self, signals: List[Dict]) -> List[Dict]:
        """
        Ранжирует список торговых сигналов, используя SignalRanker.
        """
        return self.signal_ranker.rank_signals(signals)


# ---
## 4. Feature Calculators
---

class LevelFeatureCalculator:
    """
    Отвечает за расчет признаков, связанных с уровнями поддержки/сопротивления.
    """
    def __init__(self, days_level_analyzer: DaysLevel):
        self.days_level = days_level_analyzer

    def calculate_distance_to_levels(self, current_price: float) -> Dict:
        """
        Расчет расстояния от текущей цены до различных уровней.
        Теперь не принимает levels, а сам их запрашивает у days_level.
        """
        strongest_levels = self.days_level.get_strongest_levels(timeframes=['1H', '4H', '1D'], top_n=20)

        if not strongest_levels:
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

        for level_data in strongest_levels:
            _, level, strength, _, level_type, _ = level_data

            if level_type == 'mirror':
                mirrors.append((level, strength))
            elif level < current_price:
                supports.append((level, strength))
            else:
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


class VolatilityFeatureCalculator:
    """
    Отвечает за расчет характеристик волатильности.
    """
    def calculate_volatility_features(self, data_slice: pd.DataFrame, window: int = 10) -> Dict:
        """
        Расчет характеристик волатильности для заданной нарезки данных.
        """
        if len(data_slice) < window:
            return {
                'volatility_ratio': 0.03, 'is_accumulation': False,
                'price_range_ratio': 0.05, 'volume_trend': 0.01
            }

        avg_atri = data_slice['atri'].rolling(window=window).mean().iloc[-1] if 'atri' in data_slice.columns and not data_slice['atri'].empty else 0.01
        current_volatility = data_slice['atri'].iloc[-1] if 'atri' in data_slice.columns and not data_slice['atri'].empty else 0.01

        volatility_ratio = current_volatility / avg_atri if avg_atri > 0 else 0

        is_accumulation = volatility_ratio < 0.5

        price_range = (data_slice['high'].iloc[-1] - data_slice['low'].iloc[-1]) / data_slice['close'].iloc[-1]
        avg_price_range = ((data_slice['high'] - data_slice['low']) / data_slice['close']).rolling(window=window).mean().iloc[-1]
        price_range_ratio = price_range / avg_price_range if avg_price_range > 0 else 0

        volume_trend = 0
        if 'volume' in data_slice.columns:
            recent_volume = data_slice['volume'].iloc[-window:].mean()
            prev_volume = data_slice['volume'].iloc[-2*window:-window].mean()
            volume_trend = (recent_volume - prev_volume) / prev_volume if prev_volume > 0 else 0

        return {
            'volatility_ratio': volatility_ratio,
            'is_accumulation': is_accumulation,
            'price_range_ratio': price_range_ratio,
            'volume_trend': volume_trend
        }


class TechnicalIndicatorCalculator:
    """
    Отвечает за расчет технических индикаторов.
    """
    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Расчет RSI."""
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

    def _ema(self, prices: np.ndarray, period: int) -> float:
        """Расчет экспоненциального скользящего среднего."""
        if len(prices) < period:
            return np.mean(prices)

        alpha = 2.0 / (period + 1)
        ema = prices[0]

        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema

    def calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float]:
        """Расчет MACD."""
        if len(prices) < slow:
            return 0.0, 0.0

        ema_fast = self._ema(prices, fast)
        ema_slow = self._ema(prices, slow)

        macd = ema_fast - ema_slow

        if len(prices) >= slow + signal:
            macd_line = []
            for i in range(slow-1, len(prices)): # Придется пересчитывать EMA для каждого шага
                ema_f = self._ema(prices[:i+1], fast)
                ema_s = self._ema(prices[:i+1], slow)
                macd_line.append(ema_f - ema_s)
            macd_signal = np.mean(macd_line[-signal:])
        else:
            macd_signal = 0.0

        return macd, macd_signal

# ---
## 5. Data Processor
---

class DataProcessor:
    """
    Отвечает за подготовку данных для обучения и предсказания,
    включая масштабирование и агрегацию признаков.
    """
    def __init__(self, lookback_period: int, prediction_horizon: int, prediction_type: str,
                 level_calculator: LevelFeatureCalculator,
                 volatility_calculator: VolatilityFeatureCalculator,
                 tech_indicator_calculator: TechnicalIndicatorCalculator):
        self.lookback_period = lookback_period
        self.prediction_horizon = prediction_horizon
        self.prediction_type = prediction_type
        self.price_scaler = MinMaxScaler()
        self.feature_scaler = StandardScaler()

        # Композиция: используем калькуляторы признаков
        self.level_calculator = level_calculator
        self.volatility_calculator = volatility_calculator
        self.tech_indicator_calculator = tech_indicator_calculator

    def _create_features(self, data: pd.DataFrame, index: int) -> np.ndarray:
        """
        Создание признаков для нейросети из заданной нарезки данных.
        Использует делегирование для расчета отдельных групп признаков.
        """
        if index < self.lookback_period:
            return None

        historical_prices = data['close'].iloc[index-self.lookback_period:index].values
        price_returns = np.diff(historical_prices) / historical_prices[:-1]

        current_price = data['close'].iloc[index]

        # Делегируем расчеты
        level_features = self.level_calculator.calculate_distance_to_levels(current_price)
        volatility_features = self.volatility_calculator.calculate_volatility_features(
            data.iloc[index-self.lookback_period:index+1]
        )
        rsi = self.tech_indicator_calculator.calculate_rsi(historical_prices)
        macd, macd_signal = self.tech_indicator_calculator.calculate_macd(historical_prices)

        features = np.concatenate([
            price_returns,
            [np.mean(price_returns), np.std(price_returns), np.min(price_returns), np.max(price_returns)],
            [level_features['nearest_support_dist'], level_features['nearest_resistance_dist'],
             level_features['nearest_support_strength'], level_features['nearest_resistance_strength'],
             level_features['nearest_mirror_dist'], level_features['nearest_mirror_strength'],
             level_features['total_levels_above'], level_features['total_levels_below'],
             level_features['weighted_support_strength'], level_features['weighted_resistance_strength']],
            [volatility_features['volatility_ratio'],
             float(volatility_features['is_accumulation']),
             volatility_features['price_range_ratio'],
             volatility_features['volume_trend']],
            [rsi, macd, macd_signal],
            [current_price / np.mean(historical_prices),
             (current_price - np.min(historical_prices)) / (np.max(historical_prices) - np.min(historical_prices))]
        ])
        return features

    def prepare_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Подготовка данных для обучения модели.
        Масштабирует данные и целевые значения.
        """
        X, y = [], []

        for i in range(self.lookback_period, len(data) - self.prediction_horizon):
            features = self._create_features(data, i)
            if features is None:
                continue

            current_price = data['close'].iloc[i]
            future_price = data['close'].iloc[i + self.prediction_horizon]

            if self.prediction_type == 'price':
                target = future_price
            else: # 'change'
                target = (future_price - current_price) / current_price

            X.append(features)
            y.append(target)

        X = np.array(X)
        y = np.array(y).reshape(-1, 1) # Reshape for scaler

        # Масштабирование признаков и целевых значений
        X_scaled = self.feature_scaler.fit_transform(X)
        y_scaled = self.price_scaler.fit_transform(y) # Используем price_scaler для целевых значений

        return X_scaled, y_scaled

    def get_prediction_features(self, data: pd.DataFrame, current_index: int) -> Tuple[np.ndarray, Dict, Dict]:
        """
        Генерирует признаки для предсказания на текущем шаге.
        Возвращает также исходные словари признаков для дальнейшего использования.
        """
        features_raw = self._create_features(data, current_index)
        if features_raw is None:
            return None, {}, {}

        features_scaled = self.feature_scaler.transform(features_raw.reshape(1, -1))

        # Пересчитываем эти для передачи в determine_strategy
        current_price = data['close'].iloc[current_index]
        level_info = self.level_calculator.calculate_distance_to_levels(current_price)
        volatility_info = self.volatility_calculator.calculate_volatility_features(
            data.iloc[max(0, current_index - self.lookback_period):current_index+1]
        )

        return features_scaled, level_info, volatility_info

# ---
## 6. Model Manager
---

class CryptoPricePredictor:
    """
    Отвечает за создание, обучение, сохранение/загрузку
    и выполнение предсказаний нейронной сети.
    """
    def __init__(self, input_shape: Tuple[int], prediction_type: str, model_save_dir: str = 'models'):
        self.input_shape = input_shape
        self.prediction_type = prediction_type
        self.model_save_dir = model_save_dir
        self.model = None
        self.is_trained = False

        os.makedirs(model_save_dir, exist_ok=True)

    def _build_model(self):
        """Создает архитектуру нейронной сети."""
        model = Sequential([
            LSTM(units=100, return_sequences=True, input_shape=self.input_shape),
            Dropout(0.3),
            LSTM(units=100, return_sequences=False),
            Dropout(0.3),
            Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray,
                    epochs: int = 50, batch_size: int = 32):
        """Обучает модель."""
        if self.model is None:
            self._build_model()

        self.training_history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )
        self.is_trained = True

    def predict(self, features_scaled: np.ndarray) -> float:
        """Делает предсказание."""
        if not self.is_trained or self.model is None:
            raise ValueError("Модель не обучена или не загружена.")
        
        # LSTM требует 3D вход: (samples, timesteps, features)
        # Если features_scaled уже 2D (1, num_features), то timesteps = 1
        # Если features_scaled уже 3D, то все ок.
        if features_scaled.ndim == 2:
            # Преобразуем (1, num_features) в (1, 1, num_features)
            features_reshaped = features_scaled.reshape(1, 1, -1)
        else: # Предполагаем, что это уже (1, lookback_period, num_features_per_timestep)
            features_reshaped = features_scaled

        predicted_scaled = self.model.predict(features_reshaped)[0][0]
        return predicted_scaled

    def save_model(self, model_name: str = "crypto_model.h5"):
        """Сохраняет обученную модель."""
        if self.model:
            path = os.path.join(self.model_save_dir, model_name)
            self.model.save(path)
            print(f"Модель сохранена в {path}")

    def load_model(self, model_name: str = "crypto_model.h5"):
        """Загружает модель."""
        path = os.path.join(self.model_save_dir, model_name)
        if os.path.exists(path):
            from tensorflow.keras.models import load_model # Импорт внутри для ленивой загрузки
            self.model = load_model(path)
            self.is_trained = True
            print(f"Модель загружена из {path}")
        else:
            print(f"Модель не найдена по пути: {path}")


# ---
## 7. Strategy Advisor
---

class StrategyAdvisor:
    """
    Класс для определения торговой стратегии на основе предсказания,
    информации об уровнях и волатильности.
    """
    def __init__(self, days_level_analyzer: DaysLevel):
        self.days_level = days_level_analyzer

    def determine_strategy(self, current_price: float, predicted_price: float,
                           level_info: Dict, volatility_info: Dict, daily_atri: float) -> str:
        """
        Определение торговой стратегии.
        """
        price_change_abs = abs(predicted_price - current_price)
        price_change_percent = price_change_abs / current_price

        # --- Level Proximity ---
        # Convert absolute distances to ATRI multiples for better context
        # Используем .get() с дефолтами на случай отсутствия ключа
        nearest_support_dist_atri = level_info.get('nearest_support_dist', 0.0) * current_price / daily_atri if daily_atri > 0 else float('inf')
        nearest_resistance_dist_atri = level_info.get('nearest_resistance_dist', 0.0) * current_price / daily_atri if daily_atri > 0 else float('inf')
        nearest_mirror_dist_atri = level_info.get('nearest_mirror_dist', 0.0) * current_price / daily_atri if daily_atri > 0 else float('inf')

        NEAR_LEVEL_ATRI_THRESHOLD = 0.5

        near_support = nearest_support_dist_atri < NEAR_LEVEL_ATRI_THRESHOLD and level_info.get('nearest_support_strength', 0) > 0
        near_resistance = nearest_resistance_dist_atri < NEAR_LEVEL_ATRI_THRESHOLD and level_info.get('nearest_resistance_strength', 0) > 0
        near_mirror = nearest_mirror_dist_atri < NEAR_LEVEL_ATRI_THRESHOLD and level_info.get('nearest_mirror_strength', 0) > 0

        # --- Volatility and Consolidation Checks ---
        is_low_volatility = volatility_info.get('is_accumulation', False)
        has_low_recent_volume = volatility_info.get('volume_trend', 0) < 0.005 # Пример

        # --- Strategy Logic ---

        # Strategy 1: Price targeting nearest strong level (magnet effect)
        if daily_atri > 0 and price_change_abs < daily_atri * 2: # Only consider if predicted change is relatively small
            if near_support and predicted_price > current_price:
                return "Ожидаем отскок от поддержки к ближайшему уровню"
            if near_resistance and predicted_price < current_price:
                return "Ожидаем отскок от сопротивления к ближайшему уровню"

        # Strategy 2: Breakout from Consolidation
        BREAKOUT_ATRI_TARGET = 3.5

        if is_low_volatility and has_low_recent_volume:
            target_level = None
            target_distance_atri = 0

            if predicted_price > current_price:
                resistances = [lvl for lvl in self.days_level.get_strongest_levels(timeframes=['1W']) if lvl[1] > current_price]
                if resistances:
                    target_level_data = min(resistances, key=lambda x: x[1]) # Nearest weekly resistance above
                    target_level_price = target_level_data[1]
                    target_distance_atri = (target_level_price - current_price) / daily_atri if daily_atri > 0 else 0
            elif predicted_price < current_price:
                supports = [lvl for lvl in self.days_level.get_strongest_levels(timeframes=['1W']) if lvl[1] < current_price]
                if supports:
                    target_level_data = max(supports, key=lambda x: x[1]) # Nearest weekly support below
                    target_level_price = target_level_data[1]
                    target_distance_atri = (current_price - target_level_price) / daily_atri if daily_atri > 0 else 0

            if target_level_data and target_distance_atri >= BREAKOUT_ATRI_TARGET and price_change_abs > BREAKOUT_ATRI_TARGET * daily_atri:
                if (predicted_price > current_price and predicted_price <= target_level_price) or \
                   (predicted_price < current_price and predicted_price >= target_level_price):
                    return f"Прорыв из консолидации к недельному уровню {target_level_price:.2f} ({target_distance_atri:.1f} ATRI)"

        # --- Existing Strategies ---
        if near_support and level_info.get('nearest_support_strength', 0) > 5:
            return "Отбой от сильного уровня поддержки - покупка" if predicted_price > current_price else "Возможный пробой уровня поддержки - продажа"

        if near_resistance and level_info.get('nearest_resistance_strength', 0) > 5:
            return "Отбой от сильного уровня сопротивления - продажа" if predicted_price < current_price else "Возможный пробой уровня сопротивления - покупка"

        if near_mirror and level_info.get('nearest_mirror_strength', 0) > 3:
            mirror_distance = level_info.get('nearest_mirror_dist', 0) * current_price
            return f"Отработка зеркального уровня вверх - цель +{mirror_distance:.2f}" if predicted_price > current_price else f"Отработка зеркального уровня вниз - цель -{mirror_distance:.2f}"

        if price_change_percent > 0.02 or price_change_percent < -0.02:
            direction = "покупка" if predicted_price > current_price else "продажа"
            return f"Значительное движение - {direction}"

        return "Боковое движение - ожидание"


# ---
## 8. Обновленный класс CryptoPredictionNN (Оркестратор)
---

class CryptoPredictionNN:
    """
    Система для предсказания цены криптовалют и генерации торговых сигналов.
    Является оркестратором, который координирует работу специализированных компонентов.
    """

    def __init__(self, days_level: DaysLevel, lookback_period: int = 20, prediction_horizon: int = 5,
                 prediction_type: str = 'price', model_save_dir: str = 'models'):
        """
        Инициализация системы предсказания цены.
        """
        # Инициализация компонент
        self.level_calculator = LevelFeatureCalculator(days_level)
        self.volatility_calculator = VolatilityFeatureCalculator()
        self.tech_indicator_calculator = TechnicalIndicatorCalculator()
        
        self.data_processor = DataProcessor(
            lookback_period, prediction_horizon, prediction_type,
            self.level_calculator, self.volatility_calculator, self.tech_indicator_calculator
        )
        
        # Определяем input_shape для модели. Это количество признаков, которые мы генерируем.
        # Для этого нужно сгенерировать один набор признаков.
        # Предполагаем, что _create_features генерирует плоский массив признаков
        # и LSTM-слой будет работать с `timesteps=1` и `input_dim=num_features`.
        # Реальный input_shape может быть (lookback_period, num_features_per_timestep)
        # Если ваш LSTM слой ожидает sequence, то X в prepare_training_data
        # должен быть 3D-массивом (samples, timesteps, features_per_timestep).
        # В данном случае, _create_features возвращает плоский массив, что подразумевает
        # timesteps=1 для LSTM.
        
        # Для корректного определения input_shape, нам нужно знать количество признаков
        # после _create_features. В реальном коде, можно вызвать _create_features
        # на dummy-данных или рассчитать это статически.
        
        # Заглушка для input_shape, в реальности нужно вычислить!
        dummy_data = pd.DataFrame(np.random.rand(self.data_processor.lookback_period + 10, 5),
                                   columns=['close', 'high', 'low', 'volume', 'atri'])
        dummy_features = self.data_processor._create_features(dummy_data, self.data_processor.lookback_period)
        if dummy_features is None:
             raise ValueError("Не удалось сгенерировать dummy-признаки для определения input_shape.")
        input_dim = len(dummy_features) # Количество признаков
        
        self.price_predictor = CryptoPricePredictor(input_shape=(1, input_dim), prediction_type=prediction_type, model_save_dir=model_save_dir)
        self.strategy_advisor = StrategyAdvisor(days_level)
        self.signal_generator = TradingSignalGenerator() # Используем рефакторингованный класс

    def train(self, data: pd.DataFrame, epochs: int = 50, batch_size: int = 32):
        """
        Обучает нейронную сеть.
        """
        X_scaled, y_scaled = self.data_processor.prepare_training_data(data)
        
        # Разделение на тренировочную и тестовую выборки
        train_size = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
        y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]

        # LSTM ожидает 3D вход (samples, timesteps, features)
        # Если features_scaled - это (samples, num_features), то timesteps = 1
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

        self.price_predictor.train_model(X_train, y_train, X_test, y_test, epochs, batch_size)

    def predict_and_generate_signals(self, data: pd.DataFrame, current_index: int) -> Dict:
        """
        Выполняет предсказание и генерирует торговые сигналы для текущей ситуации.
        """
        current_price = data['close'].iloc[current_index]
        
        # Получаем масштабированные признаки и исходные инфо-словари
        features_scaled, level_info, volatility_info = self.data_processor.get_prediction_features(data, current_index)
        
        if features_scaled is None:
            # Обработка случая, когда недостаточно данных для признаков
            return {
                'current_price': current_price,
                'predicted_price': None,
                'predicted_change': None,
                'predicted_change_percent': None,
                'strategy': "Недостаточно данных для предсказания",
                'level_info': level_info,
                'volatility_info': volatility_info,
                'confidence': 0,
                'trading_signals': {},
                'daily_atri': 0.01
            }

        # Делаем предсказание
        predicted_scaled = self.price_predictor.predict(features_scaled)
        
        # Обратное масштабирование
        predicted_price_unscaled = self.data_processor.price_scaler.inverse_transform(predicted_scaled.reshape(1, -1))[0][0]
        
        if self.data_processor.prediction_type == 'change':
            predicted_price = current_price * (1 + predicted_price_unscaled)
            predicted_change = predicted_price_unscaled
        else: # 'price'
            predicted_price = predicted_price_unscaled
            predicted_change = (predicted_price - current_price) / current_price
            
        predicted_change_percent = predicted_change * 100

        # Получаем ATRI, убеждаясь, что он есть
        daily_atri = data['atri'].iloc[current_index] if 'atri' in data.columns and not pd.isna(data['atri'].iloc[current_index]) else 0.01

        # Определяем стратегию
        strategy = self.strategy_advisor.determine_strategy(
            current_price, predicted_price, level_info, volatility_info, daily_atri
        )

        # Генерируем торговые сигналы
        trading_signals = self.signal_generator.generate_signal(
            current_price, predicted_price, daily_atri, level_info
        )

        # Оценка уверенности модели (можно оставить в этом классе или вынести)
        confidence = self._calculate_confidence(features_scaled) # Пример, можно перенести в CryptoPricePredictor

        return {
            'current_price': current_price,
            'predicted_price': predicted_price,
            'predicted_change': predicted_change,
            'predicted_change_percent': predicted_change_percent,
            'strategy': strategy,
            'level_info': level_info,
            'volatility_info': volatility_info,
            'confidence': confidence,
            'trading_signals': trading_signals,
            'daily_atri': daily_atri
        }

    def evaluate_model(self):
        """Оценка качества модели"""
        # Делегируем вызов в price_predictor, передавая price_scaler
        return self.price_predictor.evaluate_model(self.data_processor.price_scaler, self.prediction_type)

    def generate_trading_report(self, data: pd.DataFrame, last_n_predictions: int = 10) -> Dict:
        """
        Генерация отчета по торговым сигналам за последние N предсказаний
        """
        if not self.price_predictor.is_trained:
            raise ValueError("Модель не обучена для генерации отчета.")
        
        signals = []
        
        # Генерируем сигналы для последних N периодов
        # Убедимся, что у нас достаточно данных для создания признаков и предсказания
        start_idx = max(self.lookback_period + self.prediction_horizon, len(data) - last_n_predictions)
        
        print(f"Генерация торговых сигналов для отчета (последние {len(data) - start_idx} периодов)...")

        for i in range(start_idx, len(data)):
            try:
                prediction_result = self.predict_with_trading_signals(data, i)
                # Убедимся, что предсказание было успешным (predicted_price не None)
                if prediction_result['predicted_price'] is not None:
                    # 'signal_for_ranking' - это по сути 'trading_signals' из prediction_result
                    signal_for_ranking = prediction_result['trading_signals'].copy() 
                    signal_for_ranking['timestamp'] = data.index[i] if hasattr(data, 'index') else i
                    # В 'signal_for_ranking' может не быть 'score' если generate_signal его не добавил
                    # или hot_deals пуст. rank_signals должен уметь это обрабатывать.
                    signals.append(signal_for_ranking)
                else:
                    print(f"Пропущено предсказание для индекса {i} из-за недостатка данных или ошибки.")
            except Exception as e:
                print(f"Ошибка при генерации сигнала для индекса {i}: {e}")
                continue
        
        print(f"Сгенерировано {len(signals)} сигналов.")

        if not signals:
            return {
                'total_signals': 0,
                'ranked_signals': [],
                'hot_deals_summary': {},
                'total_hot_deals': 0,
                'avg_score': 0
            }

        # Ранжируем сигналы
        ranked_signals = self.signal_generator.rank_signals(signals)
        
        # Анализируем горячие сделки
        hot_deals_summary = {}
        total_hot_deals = 0
        
        for signal_data in ranked_signals: # 'signal_data' здесь - это полный словарь сигнала из generate_signal
            hot_deals = signal_data.get('hot_deals', [])
            total_hot_deals += len(hot_deals)
            
            for deal in hot_deals:
                rr_ratio_key = round(deal['rr_ratio'], 1) # Округляем для группировки
                if rr_ratio_key not in hot_deals_summary:
                    hot_deals_summary[rr_ratio_key] = {
                        'count': 0,
                        'avg_probability': 0,
                        'total_profit_potential': 0
                    }
                
                hot_deals_summary[rr_ratio_key]['count'] += 1
                hot_deals_summary[rr_ratio_key]['avg_probability'] += deal.get('probability', 0)
                hot_deals_summary[rr_ratio_key]['total_profit_potential'] += deal.get('profit_potential', 0)
        
        # Вычисляем средние значения
        for rr_ratio in hot_deals_summary:
            count = hot_deals_summary[rr_ratio]['count']
            if count > 0:
                hot_deals_summary[rr_ratio]['avg_probability'] /= count
        
        # Вычисляем средний скор из ранжированных сигналов
        avg_score = np.mean([s.get('score', 0) for s in ranked_signals]) if ranked_signals else 0
        
        return {
            'total_signals': len(signals),
            'ranked_signals': ranked_signals,
            'hot_deals_summary': hot_deals_summary,
            'total_hot_deals': total_hot_deals,
            'avg_score': avg_score
        }
    
    def _calculate_confidence(self, features_scaled: np.ndarray) -> float:
        """
        Заглушка для расчета уверенности модели.
        В реальной системе может быть основана на метриках модели,
        или на предсказании диапазона/распределения.
        """
        # Пример: более высокая уверенность, если предсказание не экстремальное
        # или если модель хорошо себя показывает на валидационной выборке
        return 0.75 # Placeholder

    def save_system_components(self, system_name: str = "crypto_system"):
        """Сохраняет необходимые компоненты системы (модель и скейлеры)."""
        self.price_predictor.save_model(f"{system_name}_model.h5")
        import joblib # Используем joblib для сохранения скейлеров
        joblib.dump(self.data_processor.price_scaler, os.path.join(self.price_predictor.model_save_dir, f"{system_name}_price_scaler.pkl"))
        joblib.dump(self.data_processor.feature_scaler, os.path.join(self.price_predictor.model_save_dir, f"{system_name}_feature_scaler.pkl"))
        print(f"Скейлеры сохранены для {system_name}")

    def load_system_components(self, system_name: str = "crypto_system"):
        """Загружает необходимые компоненты системы."""
        self.price_predictor.load_model(f"{system_name}_model.h5")
        import joblib
        scaler_dir = self.price_predictor.model_save_dir
        try:
            self.data_processor.price_scaler = joblib.load(os.path.join(scaler_dir, f"{system_name}_price_scaler.pkl"))
            self.data_processor.feature_scaler = joblib.load(os.path.join(scaler_dir, f"{system_name}_feature_scaler.pkl"))
            print(f"Скейлеры загружены для {system_name}")
        except FileNotFoundError:
            print(f"Скейлеры не найдены для {system_name}")
            # Возможно, потребуется обучить скейлеры заново, если они не найдены



# Настройка базового логирования
# Это должно быть выполнено один раз в приложении,
# например, в главном скрипте или файле конфигурации.
# Для простоты примера, настроим здесь.
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("logs/crypto_app.log"),
                        logging.StreamHandler()
                    ])

logger = logging.getLogger(__name__)

def log_method_calls(func):
    """
    Декоратор для логирования вызовов методов класса, их аргументов и результатов.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        instance = args[0] # Первый аргумент - это всегда self для методов класса
        class_name = instance.__class__.__name__
        method_name = func.__name__

        logger.info(f"Вызов метода {class_name}.{method_name} с аргументами: args={args[1:]}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"Метод {class_name}.{method_name} успешно завершен. Результат: {result if len(str(result)) < 200 else '(...)'}")
            return result
        except Exception as e:
            logger.error(f"Ошибка в методе {class_name}.{method_name}: {e}", exc_info=True)
            raise
    return wrapper

def measure_execution_time(func):
    """
    Декоратор для измерения времени выполнения метода.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        instance = args[0]
        class_name = instance.__class__.__name__
        method_name = func.__name__

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Метод {class_name}.{method_name} выполнен за {execution_time:.4f} секунд.")
        return result
    return wrapper


class PaymentProcessor:
    def process_payment(self, amount):
        # Логика обработки платежа
        pass

class ShippingService:
    def arrange_shipping(self, address):
        # Логика организации доставки
        pass

class NotificationSender:
    def send_email(self, recipient, message):
        # Логика отправки email
        pass

class OrderProcessor:
    def __init__(self):
        self.payment_processor = PaymentProcessor()
        self.shipping_service = ShippingService()
        self.notification_sender = NotificationSender()

    def process_order(self, order_details):
        self.payment_processor.process_payment(order_details["amount"])
        self.shipping_service.arrange_shipping(order_details["address"])
        self.notification_sender.send_email(order_details["customer_email"], "Ваш заказ обработан!")