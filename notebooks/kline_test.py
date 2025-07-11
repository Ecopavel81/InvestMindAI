import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import PatchCollection
import numpy as np
import matplotlib.patches as mpatches
import os
import gc  # Импортируем сборщик мусора

# Создаем папки для сохранения графиков
os.makedirs("label_0", exist_ok=True)
os.makedirs("label_1", exist_ok=True)

# Читаем данные из CSV
df = pd.read_csv("kandles.csv", parse_dates=["open_time"])


def plot_candlestick(data, start_idx, end_idx, filename):
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


# Создаем графики для каждого диапазона по 10 свечей
total_candles = len(df)
for i in range(0, total_candles - 10):  # -10 чтобы избежать выхода за пределы
    current_data = df.iloc[i : i + 10]

    # Определяем метку на основе следующей свечи
    next_open = df.iloc[i + 10]["open"]
    next_close = df.iloc[i + 10]["close"]
    label = 1 if next_close >= next_open else 0

    start_time = current_data.iloc[0]["open_time"].strftime("%Y%m%d_%H%M")
    end_time = current_data.iloc[-1]["open_time"].strftime("%Y%m%d_%H%M")

    # Формируем путь сохранения файла в зависимости от метки
    folder = "label_1" if label == 1 else "label_0"
    filename = f"{folder}/candlestick_{start_time}_to_{end_time}_label_{label}.png"

    plot_candlestick(current_data, i, i + 10, filename)

    # Периодическая очистка памяти
    if i % 100 == 0:  # Каждые 100 итераций
        gc.collect()

print("Графики успешно сохранены в папки label_0 и label_1")
