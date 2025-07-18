import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import PatchCollection
import numpy as np
import matplotlib.patches as mpatches
import os
import gc
import json
from datetime import datetime
import pickle

class CryptoWatchlistAnalyzer:
    def __init__(self):
        self.watchlist = []
        self.analysis_results = {}
        self.models = {}  # Здесь будут храниться обученные модели для каждого тикера
        
    def load_watchlist_from_input(self):
        """Загружает watchlist из пользовательского ввода"""
        print("=== ЗАГРУЗКА WATCHLIST ===")
        print("Введите тикеры через запятую или по одному на строке.")
        print("Примеры: BTCUSDT, ETHUSDT, SOLUUSDT")
        print("Для завершения ввода нажмите Enter на пустой строке")
        print("-" * 50)
        
        tickers = []
        
        # Вариант 1: Ввод через запятую
        input_line = input("Введите тикеры (через запятую): ").strip()
        if input_line:
            tickers.extend([t.strip().upper() for t in input_line.split(',')])
        else:
            # Вариант 2: Построчный ввод
            print("Вводите тикеры по одному (Enter для завершения):")
            while True:
                ticker = input("Тикер: ").strip().upper()
                if not ticker:
                    break
                tickers.append(ticker)
        
        # Убираем дубликаты и фильтруем
        self.watchlist = list(set(filter(None, tickers)))
        print(f"\nВаш watchlist ({len(self.watchlist)} тикеров): {', '.join(self.watchlist)}")
        
    def save_watchlist(self, filename="my_watchlist.txt"):
        """Сохраняет watchlist в файл"""
        with open(filename, 'w') as f:
            for ticker in self.watchlist:
                f.write(f"{ticker}\n")
        print(f"Watchlist сохранен в файл: {filename}")
        
    def load_watchlist_from_file(self, filename="my_watchlist.txt"):
        """Загружает watchlist из файла"""
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                self.watchlist = [line.strip().upper() for line in f if line.strip()]
            print(f"Watchlist загружен из файла: {filename}")
            print(f"Тикеры ({len(self.watchlist)}): {', '.join(self.watchlist)}")
        else:
            print(f"Файл {filename} не найден")
            
    def analyze_ticker_patterns(self, ticker):
        """Анализирует паттерны для одного тикера"""
        csv_filename = f"{ticker}_kandles.csv"
        
        if not os.path.exists(csv_filename):
            return {
                'status': 'error',
                'message': f"Файл {csv_filename} не найден",
                'recommendation': 'SKIP'
            }
        
        try:
            df = pd.read_csv(csv_filename, parse_dates=["open_time"])
            
            # Базовая статистика
            total_candles = len(df)
            if total_candles < 20:
                return {
                    'status': 'error',
                    'message': f"Недостаточно данных ({total_candles} свечей)",
                    'recommendation': 'SKIP'
                }
            
            # Анализ последних 10 свечей для предсказания
            recent_data = df.tail(10)
            
            # Вычисляем технические индикаторы
            df['price_change'] = df['close'].pct_change()
            df['volume_ma'] = df['volume'].rolling(window=5).mean()
            df['price_ma'] = df['close'].rolling(window=5).mean()
            
            # Анализ тренда
            recent_trend = self.analyze_trend(recent_data)
            
            # Анализ волатильности
            volatility = df['price_change'].std() * 100
            
            # Анализ объема
            volume_trend = self.analyze_volume_trend(df.tail(20))
            
            # Простая модель предсказания (можно заменить на ML)
            prediction = self.simple_prediction_model(df.tail(20))
            
            # Генерируем рекомендацию
            recommendation = self.generate_recommendation(
                recent_trend, volatility, volume_trend, prediction
            )
            
            return {
                'status': 'success',
                'ticker': ticker,
                'total_candles': total_candles,
                'recent_trend': recent_trend,
                'volatility': round(volatility, 2),
                'volume_trend': volume_trend,
                'prediction': prediction,
                'recommendation': recommendation,
                'last_price': df['close'].iloc[-1],
                'price_change_24h': round(df['price_change'].iloc[-1] * 100, 2)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Ошибка анализа: {str(e)}",
                'recommendation': 'SKIP'
            }
    
    def analyze_trend(self, data):
        """Анализирует тренд в данных"""
        if len(data) < 3:
            return 'UNKNOWN'
        
        start_price = data['close'].iloc[0]
        end_price = data['close'].iloc[-1]
        change = (end_price - start_price) / start_price * 100
        
        if change > 2:
            return 'STRONG_UP'
        elif change > 0.5:
            return 'UP'
        elif change < -2:
            return 'STRONG_DOWN'
        elif change < -0.5:
            return 'DOWN'
        else:
            return 'SIDEWAYS'
    
    def analyze_volume_trend(self, data):
        """Анализирует тренд объема"""
        recent_volume = data['volume'].tail(5).mean()
        older_volume = data['volume'].head(15).mean()
        
        if recent_volume > older_volume * 1.2:
            return 'INCREASING'
        elif recent_volume < older_volume * 0.8:
            return 'DECREASING'
        else:
            return 'STABLE'
    
    def simple_prediction_model(self, data):
        """Простая модель предсказания (замените на ML)"""
        # Простая логика на основе технических индикаторов
        price_trend = self.analyze_trend(data)
        volume_trend = self.analyze_volume_trend(data)
        
        score = 0
        
        # Scoring based on trends
        if price_trend in ['UP', 'STRONG_UP']:
            score += 1
        elif price_trend in ['DOWN', 'STRONG_DOWN']:
            score -= 1
            
        if volume_trend == 'INCREASING':
            score += 0.5
        elif volume_trend == 'DECREASING':
            score -= 0.5
        
        # Анализ RSI-подобного индикатора
        recent_changes = data['close'].pct_change().dropna()
        if len(recent_changes) > 0:
            avg_change = recent_changes.mean()
            if avg_change > 0.01:  # 1% средний рост
                score += 0.5
            elif avg_change < -0.01:  # 1% средний спад
                score -= 0.5
        
        if score > 0.5:
            return 'BUY'
        elif score < -0.5:
            return 'SELL'
        else:
            return 'HOLD'
    
    def generate_recommendation(self, trend, volatility, volume_trend, prediction):
        """Генерирует финальную рекомендацию"""
        if prediction == 'BUY' and trend in ['UP', 'STRONG_UP'] and volume_trend == 'INCREASING':
            return 'STRONG_BUY'
        elif prediction == 'BUY':
            return 'BUY'
        elif prediction == 'SELL' and trend in ['DOWN', 'STRONG_DOWN']:
            return 'STRONG_SELL'
        elif prediction == 'SELL':
            return 'SELL'
        else:
            return 'HOLD'
    
    def analyze_watchlist(self):
        """Анализирует весь watchlist"""
        print("\n=== АНАЛИЗ WATCHLIST ===")
        print("Анализируем тикеры...")
        
        results = []
        
        for ticker in self.watchlist:
            print(f"Анализ {ticker}...", end=" ")
            result = self.analyze_ticker_patterns(ticker)
            results.append(result)
            print(f"✓ {result['recommendation']}")
        
        self.analysis_results = results
        return results
    
    def display_recommendations(self):
        """Отображает рекомендации в удобном формате"""
        if not self.analysis_results:
            print("Сначала выполните анализ watchlist")
            return
        
        print("\n" + "="*80)
        print("📊 РЕКОМЕНДАЦИИ ПО ВАШЕМУ WATCHLIST")
        print("="*80)
        
        # Группируем по рекомендациям
        recommendations = {}
        for result in self.analysis_results:
            if result['status'] == 'success':
                rec = result['recommendation']
                if rec not in recommendations:
                    recommendations[rec] = []
                recommendations[rec].append(result)
        
        # Отображаем по приоритету
        priority_order = ['STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL']
        
        for rec_type in priority_order:
            if rec_type in recommendations:
                print(f"\n🎯 {rec_type} ({len(recommendations[rec_type])} тикеров):")
                print("-" * 50)
                
                for result in recommendations[rec_type]:
                    emoji = self.get_recommendation_emoji(rec_type)
                    print(f"{emoji} {result['ticker']:<12} | "
                          f"Цена: ${result['last_price']:<8.4f} | "
                          f"Изм: {result['price_change_24h']:>6.2f}% | "
                          f"Тренд: {result['recent_trend']:<10} | "
                          f"Волат: {result['volatility']:<5.2f}%")
        
        # Показываем ошибки отдельно
        errors = [r for r in self.analysis_results if r['status'] == 'error']
        if errors:
            print(f"\n❌ ОШИБКИ АНАЛИЗА ({len(errors)} тикеров):")
            print("-" * 50)
            for error in errors:
                print(f"⚠️  {error.get('ticker', 'Unknown')}: {error['message']}")
    
    def get_recommendation_emoji(self, rec_type):
        """Возвращает эмодзи для типа рекомендации"""
        emojis = {
            'STRONG_BUY': '🚀',
            'BUY': '📈',
            'HOLD': '⏸️',
            'SELL': '📉',
            'STRONG_SELL': '🔻'
        }
        return emojis.get(rec_type, '❓')
    
    def save_analysis_report(self, filename=None):
        """Сохраняет отчет анализа"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"watchlist_analysis_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        
        print(f"\nОтчет сохранен в файл: {filename}")
    
    def run_interactive_mode(self):
        """Запускает интерактивный режим"""
        print("🚀 CRYPTO WATCHLIST ANALYZER")
        print("=" * 50)
        
        while True:
            print("\nВыберите действие:")
            print("1. Ввести новый watchlist")
            print("2. Загрузить watchlist из файла")
            print("3. Показать текущий watchlist")
            print("4. Анализировать watchlist")
            print("5. Показать рекомендации")
            print("6. Сохранить watchlist")
            print("7. Сохранить отчет анализа")
            print("0. Выход")
            
            choice = input("\nВаш выбор: ").strip()
            
            if choice == '1':
                self.load_watchlist_from_input()
            elif choice == '2':
                filename = input("Имя файла (Enter для my_watchlist.txt): ").strip()
                if not filename:
                    filename = "my_watchlist.txt"
                self.load_watchlist_from_file(filename)
            elif choice == '3':
                if self.watchlist:
                    print(f"\nТекущий watchlist ({len(self.watchlist)} тикеров):")
                    for i, ticker in enumerate(self.watchlist, 1):
                        print(f"{i}. {ticker}")
                else:
                    print("Watchlist пуст")
            elif choice == '4':
                if self.watchlist:
                    self.analyze_watchlist()
                else:
                    print("Сначала загрузите watchlist")
            elif choice == '5':
                self.display_recommendations()
            elif choice == '6':
                if self.watchlist:
                    filename = input("Имя файла (Enter для my_watchlist.txt): ").strip()
                    if not filename:
                        filename = "my_watchlist.txt"
                    self.save_watchlist(filename)
                else:
                    print("Watchlist пуст")
            elif choice == '7':
                if self.analysis_results:
                    filename = input("Имя файла (Enter для автоматического): ").strip()
                    self.save_analysis_report(filename if filename else None)
                else:
                    print("Сначала выполните анализ")
            elif choice == '0':
                print("До свидания!")
                break
            else:
                print("Неверный выбор, попробуйте еще раз")

# Пример использования
if __name__ == "__main__":
    analyzer = CryptoWatchlistAnalyzer()
    analyzer.run_interactive_mode()