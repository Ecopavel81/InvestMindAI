import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
import os
import time
from datetime import datetime

# --- Настройка API ключей ---
# РЕКОМЕНДУЕТСЯ: Использовать переменные окружения для безопасности
# export BINANCE_API_KEY="ВАШ_API_КЛЮЧ"
# export BINANCE_SECRET_KEY="ВАШ_SECRET_КЛЮЧ"
#API_KEY = os.environ.get('BINANCE_API_KEY')
#SECRET_KEY = os.environ.get('BINANCE_SECRET_KEY')

# Если переменные окружения не установлены, можно указать здесь (не рекомендуется для продакшена!)
API_KEY = ""
SECRET_KEY = ""

if not API_KEY or not SECRET_KEY:
    print("Ошибка: API_KEY или SECRET_KEY не установлены. Пожалуйста, установите их как переменные окружения.")
    print("Например: export BINANCE_API_KEY='...'")
    exit()

client = Client(API_KEY, SECRET_KEY)

# --- Список монет для парсинга ---
SYMBOLS = [
    'BTCUSDT',
    'ETHUSDT',
    'BNBUSDT',
    'SOLUSDT',
    'BNBUSDT',
    '1000PEPEUSDT',
    'XRPUSDT',
    'TRXUSDT',
    'ADAUSDT',
    'HYPEUSDT',
    'BCHUSDT',
    'SUIUSDT',
    'LINKUSDT',
    'LEOUSDT',
    'XLMUSDT',
    'AVAXUSDT',
    'TONUSDT',
    'SHIBUSDT',
    'LTCUSDT',
    'FUNUSDT',
    'FARTCOINUSDT',
    'UNIUSDT',
    'BIDUSDT',
    'TUSDT',
    'AAVEUSDT',
    'WIFUSDT',
    'SPKUSDT',
    'FDUSDUSDT',
    'SEIUSDT',
    'AERGOUSDT',
    'IPUUSDT',
    'KAIAUSDT',
    'ENAUSDT',
    'WLDUSDT',
    'MAGICUSDT',
    'PNUTUSDT',
    'GPSUSDT',
    'ZKJUSDT',
    'DOTUSDT',
    'MYXUSDT',
    'VIRTUALUSDT',
    'RESOLVUSDT'
    'MOODENGUSDT',
    'NEIROUSDT',
    'FILUSDT',
    'TAOUSDT',
    'OPUUSDT',
    'NEARUSDT',
    'AEROUSDT',
    'INJUSDT',
    'RVNUSDT',
    'CRVUSDT',
    'WCTUSDT',
    'TIAUSDT',
    'FDUSDUSDC',
    'MKRUSDT',
    'NXPCUSDT',
    'ALTUSDT',
    'ARBUSDT',
    'ONDOUSDT',
    'FETUDT',
    'LAUSDT',
    'TRBUSDT',
    'SYRUPUSDT',
    'HOMEUDT',
    'HUMAUSDT',
    'ETCUSDT',
    'KAITOUSDT',
    'POPCATUSDT',
    'JTOUSDT'
    'BMTUSDT',
    'RAYSOLUSDT',
    '1000BOBUSDT',
    'EIFGEUSDT',
    # Добавьте другие монеты, которые вам нужны
]

# --- Параметры парсинга ---
INTERVAL = Client.KLINE_INTERVAL_1HOUR  # Интервал 1 час
START_DATE = "1 Sep, 2024"  # Дата начала загрузки данных (формат: "День Месяц, Год")
END_DATE = "now"            # Дата окончания загрузки данных ("now" для текущей даты)
OUTPUT_CSV_FILE = 'binance_data_1h.csv' # Имя выходного CSV файла

def get_klines_data(symbol, interval, start_str, end_str=None):
    """
    Загружает исторические данные (свечи) для указанной торговой пары.
    Обрабатывает ограничения по лимиту загрузки, загружая данные частями.
    """
    all_klines = []
    print(f"Начинаю загрузку данных для {symbol} с {start_str} до {end_str if end_str else 'текущего момента'}...")

    try:
        # Binance API может возвращать не более 1000 свечей за один запрос.
        # Если период очень большой, потребуется несколько запросов.
        # klines = client.get_historical_klines(symbol, interval, start_str, end_str)
        # Для больших периодов используем get_historical_klines_generator
        
        # Вместо get_historical_klines используем итератор для больших объемов данных
        # Это автоматически обрабатывает пагинацию и лимиты
        klines_generator = client.get_historical_klines_generator(symbol, interval, start_str, end_str)
        
        for kline in klines_generator:
            all_klines.append(kline)
            # Добавляем небольшую задержку, чтобы избежать превышения лимитов
            time.sleep(0.01) # 10 миллисекунд

    except BinanceAPIException as e:
        print(f"Ошибка Binance API при получении данных для {symbol}: {e}")
        return pd.DataFrame() # Возвращаем пустой DataFrame в случае ошибки
    except Exception as e:
        print(f"Произошла непредвиденная ошибка при получении данных для {symbol}: {e}")
        return pd.DataFrame()

    if not all_klines:
        print(f"Нет данных для {symbol} за указанный период.")
        return pd.DataFrame()

    # Преобразование данных в DataFrame
    df = pd.DataFrame(all_klines, columns=[
        'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close time', 'Quote asset volume', 'Number of trades',
        'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
    ])

    # Преобразование времени из миллисекунд в удобочитаемый формат
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
    df['Close time'] = pd.to_datetime(df['Close time'], unit='ms')

    # Преобразование числовых столбцов в float
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
                    'Quote asset volume', 'Taker buy base asset volume',
                    'Taker buy quote asset volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Добавляем столбец с символом
    df['Symbol'] = symbol

    # Выбираем необходимые столбцы и переупорядочиваем их
    df = df[['Symbol', 'Open time', 'Close time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Number of trades']]
    
    # Сортировка по времени открытия, чтобы данные были последовательными
    df = df.sort_values(by='Open time')

    print(f"Загружено {len(df)} записей для {symbol}.")
    return df

def main():
    all_data_frames = []
    
    for symbol in SYMBOLS:
        df_symbol = get_klines_data(symbol, INTERVAL, START_DATE, END_DATE)
        if not df_symbol.empty:
            all_data_frames.append(df_symbol)
        
        # Задержка между запросами для разных символов, чтобы избежать превышения лимитов
        time.sleep(1) # 1 секунда

    if all_data_frames:
        # Объединяем все DataFrame в один
        final_df = pd.concat(all_data_frames, ignore_index=True)
        
        # Сортировка по дате и символу
        final_df = final_df.sort_values(by=['Symbol', 'Open time'])
        
        # Сохранение в CSV файл
        final_df.to_csv(OUTPUT_CSV_FILE, index=False)
        print(f"\nДанные успешно сохранены в файл: {OUTPUT_CSV_FILE}")
        print(f"Общее количество записей: {len(final_df)}")
    else:
        print("Не удалось загрузить данные ни для одной из указанных монет.")

if __name__ == "__main__":
    main()