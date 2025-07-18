import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import PatchCollection
import numpy as np
import matplotlib.patches as mpatches
import os
import gc  # Импортируем сборщик мусора

def load_tickers_from_file(filename):
    """Загружает тикеры из текстового файла"""
    tickers = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            # Пропускаем пустые строки и комментарии
            if line and not line.startswith('Выбрано') and not line.startswith('#'):
                # Убираем комментарии после тикера
                ticker = line.split(' - ')[0].strip()
                if ticker:
                    tickers.append(ticker)
    return tickers

def plot_candlestick(data, start_idx, end_idx, filename):
    """Создает график свечей"""
    # Создаем фигуру и оси
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Создаем свечи
    green_patches = []
    red_patches = []
    
    for i in range(len(data)):
        x = start_idx + i
        open_price = data.iloc[i]["open"]
        close_price = data.iloc[i]["close"]
        low_price = data.iloc[i]["low"]
        high_price = data.iloc[i]["high"]
        
        if close_price >= open_price:
            color = "green"
            height = close_price - open_price
            bottom = open_price
            green_patches.append(mpatches.Rectangle((x - 0.2, bottom), 0.4, height))
        else:
            color = "red"
            height = open_price - close_price
            bottom = close_price
            red_patches.append(mpatches.Rectangle((x - 0.2, bottom), 0.4, height))
        
        # Добавляем тени свечей
        ax.plot([x, x], [low_price, high_price], color=color)
    
    # Добавляем тела свечей
    pc_green = PatchCollection(green_patches, facecolor="green", edgecolor="black")
    pc_red = PatchCollection(red_patches, facecolor="red", edgecolor="black")
    ax.add_collection(pc_green)
    ax.add_collection(pc_red)
    
    # Убираем все элементы оформления
    ax.axis("off")  # Убираем оси полностью
    plt.box(False)  # Убираем рамку
    
    # Сохраняем график
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    
    # Очищаем память
    plt.close(fig)
    del fig, ax, green_patches, red_patches, pc_green, pc_red
    gc.collect()  # Запускаем сборщик мусора

def process_ticker_data(ticker):
    """Обрабатывает данные для одного тикера"""
    # Предполагаем, что файлы с данными имеют формат: {ticker}_kandles.csv
    csv_filename = f"{ticker}_kandles.csv"
    
    if not os.path.exists(csv_filename):
        print(f"Файл {csv_filename} не найден, пропускаем тикер {ticker}")
        return
    
    try:
        # Читаем данные из CSV
        df = pd.read_csv(csv_filename, parse_dates=["open_time"])
        print(f"Обрабатываем тикер {ticker}, найдено {len(df)} свечей")
        
        # Создаем папки для текущего тикера
        ticker_folder = f"charts_{ticker}"
        os.makedirs(f"{ticker_folder}/label_0", exist_ok=True)
        os.makedirs(f"{ticker_folder}/label_1", exist_ok=True)
        
        # Создаем графики для каждого диапазона по 10 свечей
        total_candles = len(df)
        processed_count = 0
        
        for i in range(0, total_candles - 10):  # -10 чтобы избежать выхода за пределы
            current_data = df.iloc[i : i + 10]
            
            # Определяем метку на основе следующей свечи
            next_open = df.iloc[i + 10]["open"]
            next_close = df.iloc[i + 10]["close"]
            label = 1 if next_close >= next_open else 0
            
            start_time = current_data.iloc[0]["open_time"].strftime("%Y%m%d_%H%M")
            end_time = current_data.iloc[-1]["open_time"].strftime("%Y%m%d_%H%M")
            
            # Формируем путь сохранения файла в зависимости от метки
            folder = f"{ticker_folder}/label_1" if label == 1 else f"{ticker_folder}/label_0"
            filename = f"{folder}/candlestick_{ticker}_{start_time}_to_{end_time}_label_{label}.png"
            
            plot_candlestick(current_data, i, i + 10, filename)
            processed_count += 1
            
            # Периодическая очистка памяти
            if processed_count % 100 == 0:  # Каждые 100 итераций
                gc.collect()
                print(f"Обработано {processed_count} графиков для {ticker}")
        
        print(f"Для тикера {ticker} создано {processed_count} графиков")
        
    except Exception as e:
        print(f"Ошибка при обработке тикера {ticker}: {e}")

def main():
    """Основная функция"""
    # Загружаем тикеры из файла
    tickers_file = "Тикеры крипты топ.txt"
    
    if not os.path.exists(tickers_file):
        print(f"Файл {tickers_file} не найден!")
        return
    
    tickers = load_tickers_from_file(tickers_file)
    print(f"Загружено {len(tickers)} тикеров: {tickers[:5]}..." if len(tickers) > 5 else f"Загружено {len(tickers)} тикеров: {tickers}")
    
    # Обрабатываем каждый тикер
    for ticker in tickers:
        # Пропускаем тикеры с пометкой "лучше не брать"
        if ticker in ['DOGEUSDT', 'TRUMPUSDT']:
            print(f"Пропускаем тикер {ticker} (помечен как нежелательный)")
            continue
            
        print(f"\nНачинаем обработку тикера: {ticker}")
        process_ticker_data(ticker)
        
        # Очистка памяти между тикерами
        gc.collect()
    
    print("\nВсе тикеры обработаны!")

if __name__ == "__main__":
    main()