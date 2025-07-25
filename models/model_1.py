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
import warnings
warnings.filterwarnings('ignore')

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
data = df.copy()


class CriptoPredictionNN:
    """
    Нейросеть для предсказания цены актива на основе анализа уровней поддержки и сопротивления.

    Основные стратегии:
    1) Отбой от сильного уровня (сверху или снизу)
    2) Пробой уровня: наблюдается накопление, волатильность < 0.5 ATRI
    3) Зеркальный уровень: цена пройдет такое же расстояние, как до зеркального уровня
    """

    def __init__(self, days_level, lookback_period=20, prediction_horizon=5,
                 prediction_type='price'):
        """
        Инициализация модели предсказания цены

        Args:
            days_level_analyzer: экземпляр класса DaysLevel
            lookback_period: количество периодов для анализа истории
            prediction_horizon: горизонт предсказания (количество периодов вперед)
            prediction_type: тип предсказания ('price' или 'change')
        """
        self.days_level = days_level
        self.lookback_period = lookback_period
        self.prediction_horizon = prediction_horizon
        self.prediction_type = prediction_type
        self.is_trained = False

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
        nearest_resistance = min(resistances, key=lambda x: x[0]) if resistances else (0, 0)
        nearest_mirror = min(mirrors, key=lambda x: abs(x[0] - current_price)) if mirrors else (0, 0)

        # Расчет расстояний
        support_dist = (current_price - nearest_support[0]) / current_price if nearest_support[0] > 0 else 0
        resistance_dist = (nearest_resistance[0] - current_price) / current_price if nearest_resistance[0] > 0 else 0
        mirror_dist = abs(nearest_mirror[0] - current_price) / current_price if nearest_mirror[0] > 0 else 0

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
        if len(df) < window:
            return {
                'volatility_ratio': 0.03,
                'is_accumulation': False,
                'price_range_ratio': 0.05,
                'volume_trend': 0.01
            }

        # Средний ATRI за период
        avg_atri = df['atri'].rolling(window=window).mean().iloc[-1]
        current_volatility = df['atri'].iloc[-1]

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

    def build_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Построение матрицы признаков на основе _create_features

        Args:
          data: DataFrame с OHLCV данными

        Returns:
          np.ndarray: матрица признаков
        """
        features = []

        for i in range(len(data)):
            f = self._create_features(data, i)
            if f is not None:
                features.append(f)

        return np.array(features)

    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Подготовка данных для обучения

        Args:
            data: DataFrame с OHLC данными

        Returns:
            Tuple[X, y] - признаки и целевые значения
        """
        prices = data['close'].values.reshape(-1, 1)

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

        return self.training_history

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

    def predict(self, data: pd.DataFrame, current_index: int = None) -> Dict:
        """
        Предсказание цены

        Args:
            data: DataFrame с данными
            current_index: индекс для предсказания (если None, используется последний)

        Returns:
            dict с предсказанием и дополнительной информацией
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
            'confidence': self._calculate_confidence(features_scaled)
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
        near_support = level_info['nearest_support_dist'] < 0.01  # В пределах 1%
        near_resistance = level_info['nearest_resistance_dist'] < 0.01
        near_mirror = level_info['nearest_mirror_dist'] < 0.01

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

    def save_model(self, filepath: str):
        """Сохранение модели"""
        if self.model is None:
            raise ValueError("Модель не создана")

        self.model.save(filepath)
        print(f"Модель сохранена в {filepath}")

    def load_model(self, filepath: str):
        """Загрузка модели"""
        self.model = keras.models.load_model(filepath)
        self.is_trained = True
        print(f"Модель загружена из {filepath}")
    
    
    
        
    # ==Пример использования==
    def example_usage():
        """
        Пример использования системы предсказания цены
        """
        # Создание тестовых данных
        np.random.seed(42)
        dates = pd.date_range('2025-01-01', periods=1000, freq='1D')

        # Генерация синтетических OHLC данных
        price = 100
        prices = [price]

        for i in range(999):
            change = np.random.normal(0, 0.02)  # 2% стандартное отклонение
            price *= (1 + change)
            prices.append(price)

        # Создание OHLC данных
        data = pd.DataFrame(index=dates)
        data['close'] = prices
        data['open'] = data['close'].shift(1) * (1 + np.random.normal(0, 0.005, len(data)))
        data['high'] = np.maximum(data['open'], data['close']) * (1 + np.abs(np.random.normal(0, 0.01, len(data))))
        data['low'] = np.minimum(data['open'], data['close']) * (1 - np.abs(np.random.normal(0, 0.01, len(data))))
        data['volume'] = np.random.uniform(100, 1000, len(data))  # если требуется
        data = data.dropna()

        print("Создание анализатора уровней...")

        # Заменитель DaysLevel
        class MockDaysLevel:
            def __init__(self, data):
                self.data = data
                self.timeframe = '1D'

            def get_strongest_levels(self, timeframes=None, top_n=10):
                # Возвращаем фиктивные уровни
                current_price = self.data['close'].iloc[-1]
                levels = []
                for i in range(top_n):
                    level = current_price * (1 + np.random.normal(0, 0.05))
                    strength = np.random.uniform(1, 10)
                    levels.append((
                        self.data.index[-1],
                        level,
                        strength,
                        '1H',
                        np.random.choice(['support', 'resistance', 'mirror']),
                        np.random.choice(['up', 'down', 'neutral'])
                    ))
                return levels

        # Добавляем ATRI (заглушка)
        data['atri'] = data['close'].rolling(14).std()

        days_level = MockDaysLevel(data)

        print("Создание модели предсказания...")
        predictor = CriptoPredictionNN(
            days_level=days_level,
            lookback_period=20,
            prediction_horizon=5,
            prediction_type = 'price'
        )

        print("Обучение модели...")
        try:
            history = predictor.train(data, epochs=50, batch_size=16, verbose=1)

            print("\nПредсказание для последних данных...")
            prediction = predictor.predict(data)

            print("\n" + "="*50)
            print("РЕЗУЛЬТАТ ПРЕДСКАЗАНИЯ")
            print("="*50)
            print(f"Текущая цена: {prediction['current_price']:.4f}")
            print(f"Предсказанная цена: {prediction['predicted_price']:.4f}")
            print(f"Ожидаемое изменение: {prediction['predicted_change_percent']:.2f}%")
            print(f"Стратегия: {prediction['strategy']}")
            print(f"Уверенность: {prediction['confidence']:.2f}")

            # Визуализация истории обучения
            predictor.plot_training_history()

            # Визуализация предсказаний
            predictor.plot_predictions()

        except Exception as e:
            print(f"Ошибка: {e}")
            import traceback
            traceback.print_exc()

    if __name__ == "__main__":
        example_usage()

    
