import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Установка опций для Matplotlib, чтобы избежать повторных ошибок (если они связаны с базовыми настройками)
plt.rcParams['patch.linewidth'] = 0.5
plt.rcParams['axes.linewidth'] = 0.8

def calculate_volume_movement(df: pd.DataFrame) -> pd.DataFrame:
    if 'Volume' not in df.columns or 'Close' not in df.columns:
        raise ValueError("DataFrame должен содержать колонки 'Volume' и 'Close'.")

    df['Volume_Change_Pct'] = df['Volume'].pct_change()
    df['Close_Change_Pct'] = df['Close'].pct_change()
    df['Volume_Movement'] = df['Volume_Change_Pct'] * df['Close_Change_Pct']
    return df

def normalize_data(df: pd.DataFrame, columns_to_normalize: list) -> pd.DataFrame:
    scaler = MinMaxScaler()
    df_result = df.copy() # Работаем с копией, которая будет возвращена

    for col in columns_to_normalize:
        if col in df_result.columns:
            if df_result[col].dropna().empty: 
                print(f"Предупреждение: Колонка '{col}' содержит только NaN значения или пустая после dropna, нормализация невозможна.")
                df_result[col + '_Normalized'] = np.nan 
            else:
                df_result[col + '_Normalized'] = scaler.fit_transform(df_result[[col]])
        else:
            print(f"Предупреждение: Колонка '{col}' не найдена для нормализации.")
    return df_result # Возвращаем эту измененную копию

def fetch_and_analyze_btc_data(ticker_symbol: str = "BTC-USD", period: str = "1y"):
    print(f"Загрузка данных для {ticker_symbol} за период {period}...")
    try:
        data = yf.download(ticker_symbol, period=period)
        if data.empty:
            print(f"Не удалось загрузить данные для {ticker_symbol}. Проверьте символ или период.")
            return
    except Exception as e:
        print(f"Произошла ошибка при загрузке данных: {e}")
        return

    if 'Volume' not in data.columns:
        print(f"Ошибка: Колонка 'Volume' не найдена в данных для {ticker_symbol}.")
        print(f"Доступные колонки: {data.columns.tolist()}")
        return
    if 'Close' not in data.columns:
        print(f"Ошибка: Колонка 'Close' не найдена в данных для {ticker_symbol}.")
        print(f"Доступные колонки: {data.columns.tolist()}")
        return

    print("Расчет движения объема...")
    data = calculate_volume_movement(data)

    print("Нормализация данных...")
    cols_to_norm = ['Close', 'Volume', 'Volume_Movement']
    # Важно: data = normalize_data(data, cols_to_norm) гарантирует,
    # что data теперь ссылается на новую, измененную копию DataFrame.
    data = normalize_data(data, cols_to_norm) 

    if 'Volume_Movement_Normalized' not in data.columns:
        print(f"Критическая ошибка: Колонка 'Volume_Movement_Normalized' не была создана после нормализации.")
        print(f"Доступные колонки после нормализации: {data.columns.tolist()}")
        print("\nПервые 5 значений 'Volume_Movement':\n", data['Volume_Movement'].head())
        print("\nПоследние 5 значений 'Volume_Movement':\n", data['Volume_Movement'].tail())
        print("\nКоличество NaN в 'Volume_Movement':", data['Volume_Movement'].isna().sum())
        return

    print("Построение графиков...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(data.index, data['Close_Normalized'], label='Цена закрытия (Normalized)', color='blue')
    axes[0].set_ylabel('Normalized Value')
    axes[0].set_title(f'Динамика нормализованной цены и объема для {ticker_symbol}')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].bar(data.index, data['Volume_Normalized'], label='Объем (Normalized)', color='grey', alpha=0.7)
    axes[1].set_ylabel('Normalized Volume')
    axes[1].legend()
    axes[1].grid(True)

    # --- КРИТИЧЕСКОЕ ИЗМЕНЕНИЕ: Явное создание DataFrame для графика
    # Сначала убедимся, что есть все колонки, затем делаем dropna
    # и используем этот новый DataFrame для построения графика.
    plot_data_for_volume_movement = data[['Volume_Movement_Normalized']].dropna()

    print("\n--- Проверка данных для третьего графика ---")
    print(f"Количество NaN в 'Volume_Movement_Normalized' до dropna: {data['Volume_Movement_Normalized'].isna().sum()}")
    print(f"Размер DataFrame после dropna: {plot_data_for_volume_movement.shape}")
    print(f"Пуст ли plot_data_for_volume_movement: {plot_data_for_volume_movement.empty}")
    
    if plot_data_for_volume_movement.empty:
        print("Недостаточно данных для построения графика 'Движение объема (Normalized)' после удаления NaN.")
        print("Третий график не будет построен.")
    else:
        x_data = plot_data_for_volume_movement.index.to_numpy()
        y_data = plot_data_for_volume_movement['Volume_Movement_Normalized'].to_numpy()
        
        print(f"Длина y_data (данных для оси Y): {len(y_data)}")
        print(f"Первые 5 значений y_data: {y_data[:5]}")
        print(f"Последние 5 значений y_data: {y_data[-5:]}")
        print(f"Есть ли NaN в y_data после dropna: {np.isnan(y_data).any()}")

        bars = axes[2].bar(x_data,
                           y_data,
                           label='Движение объема (Normalized)',
                           color='gray',
                           alpha=0.8,
                           edgecolor='none',
                           linewidth=0)

        for bar, val in zip(bars, y_data):
            if pd.isna(val):
                bar.set_color('lightgray')
            elif val > 0:
                bar.set_color('green')
            else:
                bar.set_color('red')

        axes[2].set_ylabel('Движение объема (Normalized)')
        axes[2].set_xlabel('Дата')
        axes[2].legend()
        axes[2].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    fetch_and_analyze_btc_data(ticker_symbol="BTC-USD", period="6mo")