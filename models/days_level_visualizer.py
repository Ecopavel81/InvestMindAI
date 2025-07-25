import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, FancyBboxPatch
from collections import defaultdict
import seaborn as sns


class DaysLevelVisualizer:
    """Класс для визуализации истинных уровней"""

    def __init__(self, true_level_instance, figsize=(16, 12)):
        self.tl = true_level_instance
        self.figsize = figsize
        self.colors = {
            'support': '#00FF00',      # зеленый для поддержки
            'resistance': '#FF0000',   # красный для сопротивления
            'neutral': '#FFA500',      # оранжевый для нейтральных
            'candle_up': '#00CC00',
            'candle_down': '#CC0000',
            'background': '#F0F0F0'
        }

    def plot_true_levels(self, start_date=None, end_date=None, show_top_n=10,
                         show_scores=True, show_zones=True):
        data = self.tl.data.copy()

        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]

        top_levels = self.tl.get_strongest_levels(top_n=show_top_n)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize,
                                       height_ratios=[4, 1], sharex=True)

        self._plot_candlesticks(ax1, data)
        self._plot_mark_levels(ax1, top_levels, data, show_scores, show_zones)
        self._plot_level_scores(ax2, top_levels)

        ax1.set_title(f'Истинные уровни - {self.tl.symbol or "Инструмент"}',
                      fontsize=16, fontweight='bold')
        ax1.set_ylabel('Цена', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')

        ax2.set_xlabel('Дата', fontsize=12)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(data)//15)))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        return fig

    def _plot_candlesticks(self, ax, data):
        for i, (timestamp, row) in enumerate(data.iterrows()):
            open_price = row.get('open', row['close'])
            high_price = row['high']
            low_price = row['low']
            close_price = row['close']
            color = self.colors['candle_up'] if close_price >= open_price else self.colors['candle_down']

            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)
            ax.add_patch(Rectangle((i - 0.4, body_bottom), 0.8, body_height, facecolor=color, alpha=0.7))
            ax.plot([i, i], [low_price, high_price], color='black', linewidth=1)

    def _plot_mark_levels(self, ax, levels, data, show_scores, show_zones):
        if not levels:
            ax.text(0.5, 0.5, 'Истинные уровни не найдены',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=14, fontweight='bold')
            return

        max_score = max(level[2] for level in levels)

        for i, (date, level_value, score, timeframe, level_type, direction) in enumerate(levels):
            if direction == 'up':
                color = self.colors['support']
            elif direction == 'down':
                color = self.colors['resistance']
            else:
                color = self.colors['neutral']

            alpha = 0.4 + 0.6 * (score / max_score)
            linewidth = 2 + 3 * (score / max_score)

            ax.axhline(y=level_value, color=color, linewidth=linewidth, alpha=alpha,
                       label=f'Уровень {i+1}' if i < 5 else "")

            if show_zones:
                zone_size = self.tl.data['atri'].mean() * 0.5
                zone_top = level_value + zone_size
                zone_bottom = level_value - zone_size
                ax.fill_between(range(len(data)), zone_bottom, zone_top,
                                color=color, alpha=0.1)

            if show_scores:
                ax.text(len(data) * 0.02, level_value,
                        f'#{i+1} ({score:.2f})',
                        verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                        fontsize=10, fontweight='bold', color='white')

        ax.text(0.98, 0.98, f"Найдено уровней: {len(levels)}", transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9),
                fontsize=12, fontweight='bold')

    def _plot_level_scores(self, ax, strongest_levels):
        if not strongest_levels:
            return

        scores = [level[2] for level in strongest_levels]
        level_names = [f"Уровень {i+1}" for i in range(len(strongest_levels))]

        bars = ax.bar(level_names, scores, color='skyblue', alpha=0.7, edgecolor='navy')

        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(scores) * 0.01,
                    f'{score:.2f}', ha='center', va='bottom', fontweight='bold')

        ax.set_title('Скоры истинных уровней', fontweight='bold')
        ax.set_ylabel('Скор важности')
        ax.grid(True, alpha=0.3)
        if len(strongest_levels) > 10:
            plt.setp(ax.get_xticklabels(), rotation=45)

    def plot_level_analysis(self):
        levels = self.tl.true_levels
        if not levels:
            print("Нет данных для анализа")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        scores = [level['total_score'] for level in levels]
        ax1.hist(scores, bins='fd', alpha=0.7, color='lightblue', edgecolor='black')
        ax1.axvline(np.mean(scores), color='red', linestyle='--', linewidth=2,
                    label=f'Среднее: {np.mean(scores):.2f}')

        ax1.set_ylim(top=np.percentile(scores, 95) * 1.2) # Ограничим Y (если нужно):
        ax1.set_title('Распределение скоров', fontweight='bold')
        ax1.set_xlabel('Скор важности')
        ax1.set_ylabel('Количество')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        type_counts = {}
        for level in levels:
            level_type = level['type']
            type_counts[level_type] = type_counts.get(level_type, 0) + 1

        if type_counts:
            ax2.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%', startangle=90)
            ax2.set_title('Распределение типов', fontweight='bold')

        cluster_sizes = [level.get('cluster_size', 1) for level in levels]
        ax3.scatter(cluster_sizes, scores, alpha=0.7, s=60)
        ax3.set_xlabel('Размер кластера')
        ax3.set_ylabel('Скор важности')
        ax3.set_title('Скор vs Размер кластера', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        if len(cluster_sizes) > 1:
            z = np.polyfit(cluster_sizes, scores, 1)
            p = np.poly1d(z)
            ax3.plot(cluster_sizes, p(cluster_sizes), "r--", alpha=0.8)

        size_scores = [level.get('size_score', 1) for level in levels]
        ax4.bar(range(len(size_scores)), size_scores, alpha=0.7, color='orange')
        ax4.axhline(self.tl.min_atri_multiplier, color='red', linestyle='--',
                    label=f'Мин. размер ({self.tl.min_atri_multiplier} ATRI)')
        ax4.axhline(self.tl.max_atri_multiplier, color='green', linestyle='--',
                    label=f'Макс. размер ({self.tl.max_atri_multiplier} ATRI)')
        ax4.set_title('Размеры уровней (в ATRI)', fontweight='bold')
        ax4.set_xlabel('Номер уровня')
        ax4.set_ylabel('Размер (x ATRI)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


"""
# == Пример использования:==
# Анализ только дневных уровней (самые сильные)
daily_levels = strongest_levels.get_all_levels(['1D'])

# Создание экземпляра DaysLevel
mark_levels = DaysLevel(data, symbol='BTCUSDT', timeframe='1D', # Или любой другой символ/таймфрейм
                        min_atri_multiplier=3.0,
                        max_atri_multiplier=4.0)

# Создание визуализатора
visualizer_2 = DaysLevelVisualizer(mark_levels)

# Основной график
fig3 = visualizer_2.plot_true_levels(show_top_n=15)
plt.show()

# Детальный анализ
fig4 = visualizer_2.plot_level_analysis()
plt.show()
"""