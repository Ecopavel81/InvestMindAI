import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, FancyBboxPatch
from collections import defaultdict
import seaborn as sns

class DaysLevel:
    """Класс для поиска сильных уровней поддержки и сопротивления с ранжированием по временным периодам"""

    def __init__(self, dz, symbol=None, atr_period=14, timeframe='1D',
             min_atri_multiplier=2.0, max_atri_multiplier=5.0):
        """
        Инициализация класса

        Args:
            data: DataFrame с OHLC данными
            symbol: символ инструмента (опционально)
            atr_period: период для расчета ATR
            timeframe: текущий временной период данных ('1H', '4H', '1D', '1W', '1M')
        """
        self.dz = data
        self.data = data.copy()
        self.symbol = symbol
        self.atr_period = atr_period
        self.timeframe = timeframe
        self.min_atri_multiplier = min_atri_multiplier
        self.max_atri_multiplier = max_atri_multiplier
        self.indicators = {}

        # Определение весов для временных периодов (чем больше значение, тем сильнее уровень)
        self.timeframe_weights = {
            '1M': 10,    # Месячный - максимальная сила
            '1W': 8,     # Недельный
            '1D': 6,     # Дневной
            '4H': 4,     # 4-часовой
            '1H': 2,     # Часовой
            '30m': 1,    # 30-минутный
            '15m': 0.5   # 15-минутный
        }

        # Нормализация названий колонок
        self.data.columns = [col.lower() for col in self.data.columns]

        # Проверка наличия необходимых колонок
        required_cols = ['high', 'low', 'close']
        if not all(col in self.data.columns for col in required_cols):
            raise ValueError(f"Отсутствуют необходимые колонки: {required_cols}")

        # Убеждаемся, что индекс - это datetime
        if not isinstance(self.data.index, pd.DatetimeIndex):
            if 'date' in self.data.columns or 'datetime' in self.data.columns:
                date_col = 'date' if 'date' in self.data.columns else 'datetime'
                self.data.index = pd.to_datetime(self.data[date_col])
                self.data.drop(columns=[date_col], inplace=True)
            else:
                self.data.index = pd.to_datetime(self.data.index)

        # Расчет ATRI при инициализации
        self._calculate_atri()

    def _calculate_atri(self):
        """Расчет модифицированного ATR (ATRI)"""
        high = self.data['high']
        low = self.data['low']
        close = self.data['close']

        # Расчёт True Range с несколькими вариантами
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr4 = abs(close - close.shift(1))
        tr5 = abs(high - low.shift(1))  # дополнительный вариант

        true_range_i = pd.concat([tr1, tr2, tr3, tr4, tr5], axis=1).max(axis=1)

        # Скользящее среднее по true range
        atri_raw = true_range_i.rolling(window=self.atr_period).mean()

        # Вычисляем среднее значение
        atri_mean = atri_raw.mean()

        # Фильтрация: исключить слишком высокие и слишком низкие значения
        atri_filtered = atri_raw.where(
            (atri_raw < 1.5 * atri_mean) & (atri_raw > 0.6 * atri_mean)
        )

        # Сохраняем ATRI
        self.data['atri'] = atri_filtered
        self.indicators['ATRI'] = atri_filtered

    def _convert_to_daily(self):
        """Конвертация данных в дневные для анализа суточных уровней"""
        daily_data = self.data.resample('D').agg({
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'atri': 'mean'
        }).dropna()

        return daily_data

    def _convert_to_timeframe(self, target_timeframe):
        """Конвертация данных в указанный временной период"""
        freq_map = {
            '15m': '15T',
            '30m': '30T',
            '1H': '1H',
            '4H': '4H',
            '1D': '1D',
            '1W': '1W',
            '1M': '1M'
        }

        if target_timeframe not in freq_map:
            raise ValueError(f"Неподдерживаемый временной период: {target_timeframe}")

        resampled_data = self.data.resample(freq_map[target_timeframe]).agg({
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'atri': 'mean'
        }).dropna()

        return resampled_data

    def _get_level_strength(self, level_type, timeframe):
        """Расчет силы уровня на основе типа и временного периода"""
        type_multipliers = {
            'cinching': 1.0,
            'mirror': 1.2,
            'change': 1.5,
            'paranorm': 2.0
        }

        timeframe_weight = self.timeframe_weights.get(timeframe, 1)
        type_multiplier = type_multipliers.get(level_type, 1)

        return timeframe_weight * type_multiplier

    def level_cinching(self, timeframe=None):
        """
        Поиск уровней поджатия вверх (cinching) на основе high.

        Args:
            timeframe: временной период для анализа (если None, используется текущий)

        Returns:
            список: [(дата, уровень, сила, временной_период)]
        """
        if timeframe is None:
            timeframe = self.timeframe

        # Получаем данные для указанного временного периода
        if timeframe != self.timeframe:
            data = self._convert_to_timeframe(timeframe)
        else:
            data = self.data

        highs = data['high']
        cinch_levels = []
        strength = self._get_level_strength('cinching', timeframe)

        for i in range(4, len(highs)):
            h1 = highs.iloc[i - 4]
            h2 = highs.iloc[i - 3]
            h3 = highs.iloc[i - 2]
            h4 = highs.iloc[i - 1]
            h5 = highs.iloc[i]  # текущий бар

            if (h4 == h3) and (h3 > h2) and (h2 > h1):
                level = h4
                cinch_levels.append((data.index[i], level, strength, timeframe))

        return cinch_levels

    def level_mirror(self, tolerance=0.002, timeframe=None):
        """
        Поиск зеркальных уровней (mirror) — уровни, где High и Low совпадают с небольшим допуском.

        Args:
            tolerance: допуск на отклонение уровня (например, 0.002 = 0.2%)
            timeframe: временной период для анализа

        Returns:
            список: [(дата, уровень, сила, временной_период)]
        """
        if timeframe is None:
            timeframe = self.timeframe

        # Получаем данные для указанного временного периода
        if timeframe != self.timeframe:
            data = self._convert_to_timeframe(timeframe)
        else:
            data = self.data

        highs = data['high']
        lows = data['low']
        mirror_levels = []
        strength = self._get_level_strength('mirror', timeframe)

        for i in range(4, len(data)):
            h1 = highs.iloc[i - 4]
            h2 = highs.iloc[i - 3]
            l1 = lows.iloc[i - 2]
            l2 = lows.iloc[i - 1]

            # Средний уровень
            avg_level = np.mean([h1, h2, l1, l2])

            # Проверяем, что все близки к средней с допуском
            if all(abs(x - avg_level) / avg_level < tolerance for x in [h1, h2, l1, l2]):
                mirror_levels.append((data.index[i], avg_level, strength, timeframe))

        return mirror_levels

    def level_change(self, atr_multiplier=1.0, timeframe=None):
        """
        Поиск излома тренда (change) — локальные/глобальные экстремумы по High и Low
        с учётом среднего ATRI.

        Args:
            atr_multiplier: множитель для ATRI при определении значимости движения
            timeframe: временной период для анализа

        Returns:
            список: [(дата, уровень, сила, временной_период, 'up'/'down')]
        """
        if timeframe is None:
            timeframe = self.timeframe

        # Получаем данные для указанного временного периода
        if timeframe != self.timeframe:
            data = self._convert_to_timeframe(timeframe)
        else:
            data = self.data

        highs = data['high']
        lows = data['low']
        atri = data.get('atri')
        change_levels = []
        strength = self._get_level_strength('change', timeframe)

        if atri is None or atri.isnull().all():
            print(f"⚠️ ATRI не найден или содержит только NaN для {timeframe}")
            return []

        atri_mean = atri.mean()

        for i in range(5, len(highs)):
            # === Потенциальный разворот вверх ===
            h1 = highs.iloc[i - 5]
            h2 = highs.iloc[i - 4]
            h3 = highs.iloc[i - 3]
            h4 = highs.iloc[i - 2]
            h5 = highs.iloc[i - 1]

            if (h4 > h2) and (h3 > h2) and (h2 > h1):
                if (h3 - h2 > atri_mean * atr_multiplier) and (h4 - h2 > atri_mean * atr_multiplier):
                    level = h1
                    change_levels.append((data.index[i], level, strength, timeframe, 'up'))

            # === Потенциальный разворот вниз ===
            l1 = lows.iloc[i - 5]
            l2 = lows.iloc[i - 4]
            l3 = lows.iloc[i - 3]
            l4 = lows.iloc[i - 2]
            l5 = lows.iloc[i - 1]

            if (l4 < l2) and (l3 < l2) and (l2 < l1):
                if (l2 - l3 > atri_mean * atr_multiplier) and (l2 - l4 > atri_mean * atr_multiplier):
                    level = l1
                    change_levels.append((data.index[i], level, strength, timeframe, 'down'))

        return change_levels

    def level_paranorm(self, atr_paranorm=1.5, timeframe=None):
        """
        Поиск уровня по паранормальному бару (paranorm) — локальные/глобальные экстремумы по High и Low
        с учётом среднего ATRI.

        Args:
            atr_paranorm: множитель для ATRI при определении паранормального движения
            timeframe: временной период для анализа

        Returns:
            список: [(дата, уровень, сила, временной_период, 'up'/'down')]
        """
        if timeframe is None:
            timeframe = self.timeframe

        # Получаем данные для указанного временного периода
        if timeframe != self.timeframe:
            data = self._convert_to_timeframe(timeframe)
        else:
            data = self.data

        highs = data['high']
        lows = data['low']
        atri = data.get('atri')
        paranorm_levels = []
        strength = self._get_level_strength('paranorm', timeframe)

        if atri is None or atri.isnull().all():
            print(f"⚠️ ATRI не найден или содержит только NaN для {timeframe}")
            return []

        atri_mean = atri.mean()

        for i in range(5, len(highs)):
            # === Потенциальный вверх ===
            h1 = highs.iloc[i - 5]
            h2 = highs.iloc[i - 4]
            h3 = highs.iloc[i - 3]
            h4 = highs.iloc[i - 2]
            h5 = highs.iloc[i - 1]

            if (h5 > h4) and (h4 > h3) and (h3 > h2) and (h2 > h1):
                if (h5 - h4 > atri_mean * atr_paranorm):
                    level = h4
                    paranorm_levels.append((data.index[i], level, strength, timeframe, 'up'))

            # === Потенциальный низ ===
            l1 = lows.iloc[i - 5]
            l2 = lows.iloc[i - 4]
            l3 = lows.iloc[i - 3]
            l4 = lows.iloc[i - 2]
            l5 = lows.iloc[i - 1]

            if (l5 < l4) and (l4 < l3) and (l3 < l2) and (l2 < l1):
                if (l4 - l5 > atri_mean * atr_paranorm):
                    level = l5
                    paranorm_levels.append((data.index[i], level, strength, timeframe, 'down'))

        return paranorm_levels

    def get_all_levels(self, timeframes=None):
        """
        Получить все найденные уровни для указанных временных периодов

        Args:
            timeframes: список временных периодов для анализа (по умолчанию ['1H', '1D'])

        Returns:
            dict с различными типами уровней, отсортированными по силе
        """
        if timeframes is None:
            timeframes = ['1H', '1D'] if self.timeframe == '1H' else [self.timeframe, '1D']

        all_levels = {}

        for tf in timeframes:
            try:
                levels = {
                    'cinching': self.level_cinching(tf),
                    'mirror': self.level_mirror(timeframe=tf),
                    'change': self.level_change(timeframe=tf),
                    'paranorm': self.level_paranorm(timeframe=tf)
                }
                all_levels[tf] = levels
            except Exception as e:
                print(f"⚠️ Ошибка при анализе {tf}: {e}")
                continue

        return all_levels

    def get_strongest_levels(self, timeframes=None, top_n=15):
        """
        Получить самые сильные уровни, отсортированные по силе

        Args:
            timeframes: список временных периодов для анализа
            top_n: количество самых сильных уровней для возврата

        Returns:
            список кортежей: [(дата, уровень, сила, временной_период, тип, направление)]
        """
        all_levels = self.get_all_levels(timeframes)
        combined_levels = []

        for tf, level_types in all_levels.items():
            for level_type, levels in level_types.items():
                for level_data in levels:
                    if len(level_data) >= 4:  # Минимальная структура: дата, уровень, сила, временной_период
                        date, level, strength, timeframe = level_data[:4]
                        direction = level_data[4] if len(level_data) > 4 else 'neutral'
                        combined_levels.append((date, level, strength, timeframe, level_type, direction))

        # Сортировка по силе (убывание)
        combined_levels.sort(key=lambda x: x[2], reverse=True)

        return combined_levels[:top_n]

    def print_levels_summary(self, timeframes=None):
        """Вывод сводки по найденным уровням с ранжированием"""
        all_levels = self.get_all_levels(timeframes)
        strongest_levels = self.get_strongest_levels(timeframes)

        print(f"📊 Анализ уровней для {self.symbol or 'инструмента'}:")
        print("=" * 60)

        for tf, levels in all_levels.items():
            tf_weight = self.timeframe_weights.get(tf, 1)
            print(f"\n⏰ Временной период: {tf} (вес: {tf_weight})")
            print(f"🟩 Уровни поджатия (cinching): {len(levels['cinching'])}")
            print(f"🔄 Зеркальные уровни (mirror): {len(levels['mirror'])}")
            print(f"📈 Уровни излома (change): {len(levels['change'])}")
            print(f"⚡ Паранормальные уровни (paranorm): {len(levels['paranorm'])}")

        print(f"\n🏆 ТОП-{len(strongest_levels)} самых сильных уровней:")
        print("-" * 60)
        for i, (date, level, strength, tf, level_type, direction) in enumerate(strongest_levels, 1):
            direction_emoji = "📈" if direction == 'up' else "📉" if direction == 'down' else "⚖️"
            print(f"{i:2d}. {date.strftime('%Y-%m-%d %H:%M')} | {level:.5f} | "
                  f"💪{strength:.1f} | {tf} | {level_type} {direction_emoji}")

        return all_levels

# Пример использования:
"""
# Создание экземпляра класса с указанием текущего временного периода
strongest_levels = DaysLevel(data, symbol='BTCUSDT', timeframe='1H')

# Анализ уровней на разных временных периодах
all_levels = strong_levels.get_all_levels(['1H', '1D'])

# Получение самых сильных уровней
strongest = strong_levels.get_strongest_levels(['1H', '1D'], top_n=15)

# Вывод сводки с ранжированием
strong_levels.print_levels_summary(['1H', '1D'])

# Анализ только дневных уровней (самые сильные)
daily_levels = strong_levels.get_all_levels(['1D'])

# Создание экземпляра DaysLevel
mark_levels = DaysLevel(daily_levels, min_atri_multiplier=3.0, max_atri_multiplier=4.0)
"""