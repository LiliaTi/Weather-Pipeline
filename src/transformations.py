import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
from constants import COLS_TO_CELSIUS, COLS_TO_M_PER_S, COLS_TO_M , COLS_TO_MM
from constants import COLS_TO_AVG, COLS_TO_TOTAL
from constants import COLS_TO_ISO
from constants import COLS_TO_DROP, RESULT_COLS
from validators import validate_dataframe_columns

def parse_unix_time(unix_time: Union[int, float], utc_offset: int = 0) -> datetime:
    """
    Преобразует Unix-время (в секундах) в datetime с учетом смещения UTC.
    """
    if not isinstance(unix_time, (int, float)):
        raise TypeError(f"unix_time должен быть int или float, получен {type(unix_time)}")
    
    if not isinstance(utc_offset, int):
        raise TypeError(f"utc_offset должен быть int, получен {type(utc_offset)}")
    
    # Максимальное значение для Unix time в секундах (2100 год)
    max_seconds = 4102444800
    
    if unix_time < 0 or unix_time > max_seconds:
        raise ValueError(f"Недопустимое значение unix_time (ожидаются секунды): {unix_time}")
    
    try:
        return datetime.utcfromtimestamp(unix_time + utc_offset)
    except (OverflowError, OSError) as e:
        raise ValueError(f"Ошибка конвертации времени: {e}")

def validate_timestamps(df: pd.DataFrame, df_name: str, utc_offset: int) -> None:
    """
    Проверяет корректность временных меток в DataFrame.
    
    Args:
        df: DataFrame для проверки
        df_name: Название DataFrame для сообщений об ошибках
        utc_offset: Смещение временной зоны в секундах
        
    Raises:
        ValueError: Если обнаружены некорректные временные метки
    """
    if 'time' not in df.columns:
        raise ValueError(f"Отсутствует колонка 'time' в {df_name} данных")
    
    try:
        for timestamp in df['time']:
            parse_unix_time(timestamp, utc_offset)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Ошибка в {df_name} данных: {str(e)}")

def extract_weather_data(api_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Извлекает и преобразует данные о погоде из ответа API.

    Args:
        api_response: Словарь с данными от API (результат response.json()).
                     Должен содержать ключи 'daily' и 'hourly' с данными,
                     и опционально 'utc_offset_seconds'.

    Returns:
        Словарь с:
        - 'daily': DataFrame с дневными прогнозами
        - 'hourly': DataFrame с почасовыми прогнозами
        - 'utc_offset_seconds': Смещение временной зоны (0 по умолчанию)

    Raises:
        KeyError: Если отсутствуют обязательные ключи
        ValueError: Если данные имеют некорректный формат
        TypeError: Если входные данные не являются словарем
    """
    if not isinstance(api_response, dict):
        raise TypeError(f"Ожидался словарь, получен {type(api_response)}")

    required_keys = {'daily', 'hourly'}
    missing_keys = required_keys - api_response.keys()
    if missing_keys:
        raise KeyError(f"Отсутствуют обязательные ключи: {missing_keys}")

    daily_data = api_response['daily']
    hourly_data = api_response['hourly']
    utc_offset = 0 if api_response.get('utc_offset_seconds') is None else api_response['utc_offset_seconds']

    if not isinstance(utc_offset, int):
        raise TypeError("utc_offset_seconds должен быть целым числом или None")

    daily_df = pd.DataFrame(daily_data)
    hourly_df = pd.DataFrame(hourly_data)

    validate_timestamps(daily_df, 'daily', utc_offset)
    validate_timestamps(hourly_df, 'hourly', utc_offset)

    return {
        'daily': daily_df,
        'hourly': hourly_df,
        'utc_offset_seconds': utc_offset
    }

def add_date_column(df: pd.DataFrame, time_col: str = 'time', utc_offset: int = 25_200) -> pd.DataFrame:
    """
    Добавляет колонку с датой (без времени) на основе временной метки.
    
    Args:
        df: Исходный DataFrame
        time_col: Название колонки с временными метками
        
    Returns:
        DataFrame с добавленной колонкой 'date'
        
    Raises:
        ValueError: Если колонка time_col отсутствует или содержит некорректные значения
    """
    if time_col not in df.columns:
        raise ValueError(f"Колонка {time_col} не найдена в DataFrame")
    
    try:
        df['date'] = pd.to_datetime(df['time'] + utc_offset, unit='s', utc=True).dt.date
    except Exception as e:
        raise ValueError(f"Ошибка преобразования временных меток: {str(e)}")
    
    return df

def add_human_readable_datetime(df: pd.DataFrame, time_col: str = 'time', 
                               utc_offset: int = 25_200) -> pd.DataFrame:
    """
    Добавляет колонку с человекопонятной датой и временем.
    
    Args:
        df: Исходный DataFrame
        time_col: Название колонки с временными метками
        utc_offset: Смещение временной зоны в секундах
        
    Returns:
        DataFrame с добавленной колонкой 'datetime'
    """
    try:
        timestamps = pd.to_datetime(df['time'] + utc_offset, unit='s', utc=True)
        df['datetime'] = timestamps.dt.strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        raise ValueError(f"Ошибка создания человекочитаемого времени: {str(e)}")
    
    return df

def merge_daily_hourly(hourly_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Объединяет daily и hourly данные по колонке 'date'.
    
    Args:
        daily_df: DataFrame с дневными данными
        hourly_df: DataFrame с почасовыми данными
        
    Returns:
        Объединенный DataFrame
        
    Raises:
        ValueError: Если колонка 'date' отсутствует в одном из DataFrame
    """
    required_cols = {'date'}
    for df, name in [(daily_df, 'daily'), (hourly_df, 'hourly')]:
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Колонка 'date' отсутствует в {name} DataFrame")
    
    return pd.merge(
        hourly_df,
        daily_df,
        on='date',
        how='left',
        suffixes=('_hourly', '_daily')
    )

def check_numeric_column(df: pd.DataFrame, col: str) -> None:
    """Проверяет, что указанная колонка содержит числовые данные"""
    if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
        raise ValueError(f"Колонка {col} должна содержать числовые значения")

def convert_weather_units(df: pd.DataFrame) -> pd.DataFrame:
    """
    Конвертирует единицы измерения в погодных данных.
    
    Преобразования:
    - Температура: °F → °C
    - Скорость ветра: узлы → м/с 
    - Видимость: футы → метры
    - Осадки: дюймы → мм
    
    Args:
        df: Исходный DataFrame с погодными данными (будет изменен!)
        
    Returns:
        Модифицированный DataFrame с добавленными колонками в новых единицах измерения
        
    Raises:
        TypeError: Если входные данные не являются DataFrame
        ValueError: Если DataFrame пустой или содержит некорректные значения
    """
    # Базовые проверки
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Ожидается pandas DataFrame")
    if df.empty:
        raise ValueError("Передан пустой DataFrame")
    
    # Проверка всех колонок, которые будем конвертировать
    for col in (COLS_TO_CELSIUS + COLS_TO_M_PER_S + COLS_TO_M + COLS_TO_MM):
        check_numeric_column(df, col)
    
    # Основная логика конвертации
    for col in COLS_TO_CELSIUS:
        if col in df.columns:
            df[f"{col}_celsius"] = (df[col] - 32) * 5/9
    
    for col in COLS_TO_M_PER_S:
        if col in df.columns:
            df[f"{col}_m_per_s"] = df[col] * 0.514444
    
    for col in COLS_TO_M:
        if col in df.columns:
            df[f"{col}_m"] = df[col] * 0.3048
    
    for col in COLS_TO_MM:
        if col in df.columns:
            df[f"{col}_mm"] = df[col] * 25.4
    
    return df

def remove_original_units(df: pd.DataFrame, cols_to_drop=COLS_TO_DROP) -> pd.DataFrame:
    """
    Удаляет исходные колонки с единицами измерения, которые были преобразованы.
    Оставляет только колонки Итоговой таблицы.
    """
    
    # Для температурных колонок
    cols_to_drop.extend(col for col in COLS_TO_CELSIUS if col in df.columns)
    
    # Для скорости ветра
    cols_to_drop.extend(col for col in COLS_TO_M_PER_S if col in df.columns)
    
    # Для видимости
    cols_to_drop.extend(col for col in COLS_TO_M if col in df.columns)
    
    # Для осадков
    cols_to_drop.extend(col for col in COLS_TO_MM if col in df.columns)

    # Для времени unixtime
    cols_to_drop.extend(col for col in COLS_TO_ISO if col in df.columns)
    
    return df.drop(columns=cols_to_drop)

def add_daylight_flag(
    df: pd.DataFrame,
    time_col: str = 'time_hourly',
    sunrise_col: str = 'sunrise',
    sunset_col: str = 'sunset',
    result_col: str = 'is_daylight'
) -> pd.DataFrame:
    """
    Добавляет флаг daylight (True если время между sunrise и sunset).
    
    Args:
        df: Исходный DataFrame (будет изменен!)
        time_col: Название колонки с временем
        sunrise_col: Название колонки с временем восхода
        sunset_col: Название колонки с временем заката
        result_col: Название для новой колонки
        
    Returns:
        Модифицированный DataFrame с добавленной колонкой is_daylight
        
    Raises:
        ValueError: Если отсутствуют необходимые колонки
    """
    # Проверка наличия необходимых колонок
    required_cols = {time_col, sunrise_col, sunset_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Отсутствуют колонки: {missing_cols}")

    # Добавляем флаг daylight
    df[result_col] = df[time_col].between(df[sunrise_col], df[sunset_col])
    
    return df

def add_daylight_hours(
    df: pd.DataFrame,
    daylight_duration_col: str = 'daylight_duration',
    result_col: str = 'daylight_hours'
) -> pd.DataFrame:
    """
    Добавляет колонку daylight_hours (Разница в часах между sunrise и sunset).
    
    Args:
        df: Исходный DataFrame (будет изменен)
        daylight_duration_col: Название колонки с продолжительностью светового дня в секундах
        result_col: Название для новой колонки
        
    Returns:
        Модифицированный DataFrame с добавленной колонкой daylight_hours (в часах)
        
    Raises:
        ValueError: Если отсутствуют необходимые колонки
    """
    # Проверка наличия необходимых колонок
    required_cols = {daylight_duration_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Отсутствуют колонки: {missing_cols}")

    # Вычисляем разницу в секундах и конвертируем в часы
    df[result_col] = df[daylight_duration_col] / 3600
    
    return df

def remove_unit_suffix(column_name: str) -> str:
    """
    Удаляет суффиксы единиц измерения из названия колонки.
    Обрабатывает: '_celsius', '_m_per_s', '_m', '_mm'.
    
    Args:
        column_name: Исходное название колонки (например, 'temperature_2m_celsius')
        
    Returns:
        Название без суффикса (например, 'temperature_2m')
    """
    suffixes = ['_celsius', '_m_per_s', '_m', '_mm']
    for suffix in suffixes:
        if column_name.endswith(suffix):
            return column_name[:-len(suffix)]
    return column_name

def calculate_24H_avg(
    df: pd.DataFrame,
    cols_to_avg: List[str] = COLS_TO_AVG,
    time_col: str = 'datetime',
) -> pd.DataFrame:
    """
    Добавляет колонки со средними значениями за 24 часа.
    
    Args:
        df: Исходный DataFrame
        cols_to_avg: Список колонок для расчёта средних
        time_col: Колонка с временными метками (должна быть datetime)
        
    Returns:
        DataFrame с добавленными колонками avg_*_24h
        
    Raises:
        ValueError: Если колонки не найдены или time_col не datetime
    """
    # Проверка наличия колонок
    missing_cols = [col for col in cols_to_avg + [time_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Отсутствуют колонки: {missing_cols}")
    
    # Проверка типа временной колонки
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        try:
            df[time_col] = pd.to_datetime(df[time_col])
        except Exception as e:
            raise ValueError(f"Невозможно преобразовать {time_col} в datetime: {e}")
    
    # Группировка и расчёт
    daily_grouper = pd.Grouper(key=time_col, freq='D')
    
    for col in cols_to_avg:
        df[f'avg_{remove_unit_suffix(col)}_24h'] = df.groupby(daily_grouper)[col].transform('mean')
    
    return df
    

def calculate_daylight_avg(
    df: pd.DataFrame,
    cols_to_avg: List[str] = COLS_TO_AVG,
    time_col: str = 'datetime',
    is_daylight: str = 'is_daylight'
) -> pd.DataFrame:
    """
    Добавляет колонки со средними значениями за световой день.
    
    Args:
        df: Исходный DataFrame
        cols_to_avg: Список колонок для расчёта средних
        time_col: Колонка с временными метками (datetime)
        is_daylight: Колонка с флагом daylight
        
    Returns:
        DataFrame с добавленными колонками avg_*_daylight
        
    Raises:
        ValueError: Если отсутствуют необходимые колонки
    """
    # Проверка наличия колонок
    required_cols = cols_to_avg + [time_col, is_daylight]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Отсутствуют колонки: {missing_cols}")
    
    # Проверка типа временной колонки
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Группировка для агрегации
    daily_grouper = pd.Grouper(key=time_col, freq='D')
    daylight_df = df[df[is_daylight]]
    
    # Для каждой колонки рассчитываем среднее за световой день
    for col in cols_to_avg:
        # Вычисляем средние значения
        daylight_avg = daylight_df.groupby(daily_grouper)[col].mean().reset_index()
        daylight_avg.columns = [time_col, f'avg_{remove_unit_suffix(col)}_daylight']
        
        # Преобразуем дату для корректного merge
        daylight_avg['date'] = daylight_avg[time_col].dt.date
        
        # Объединяем с исходным DataFrame
        df = df.merge(
            daylight_avg[['date', f'avg_{remove_unit_suffix(col)}_daylight']],
            on='date',
            how='left'
        )
    
    return df

def calculate_24H_total(
    df: pd.DataFrame,
    cols_to_total: List[str] = COLS_TO_TOTAL,
    time_col: str = 'datetime',
) -> pd.DataFrame:
    """
    Добавляет колонки с суммарными значениями за 24 часа.
    
    Args:
        df: Исходный DataFrame
        cols_to_total: Список колонок для расчёта сумм
        time_col: Колонка с временными метками (должна быть datetime)
        
    Returns:
        DataFrame с добавленными колонками total_*_24h
        
    Raises:
        ValueError: Если колонки не найдены или time_col не datetime
    """
    # Проверка наличия колонок
    missing_cols = [col for col in cols_to_total + [time_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Отсутствуют колонки: {missing_cols}")
    
    # Проверка типа временной колонки
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        try:
            df[time_col] = pd.to_datetime(df[time_col])
        except Exception as e:
            raise ValueError(f"Невозможно преобразовать {time_col} в datetime: {e}")
    
    # Группировка и расчёт
    daily_grouper = pd.Grouper(key=time_col, freq='D')
    
    for col in cols_to_total:
        df[f'total_{remove_unit_suffix(col)}_24h'] = df.groupby(daily_grouper)[col].transform('sum')
    
    return df
    

def calculate_daylight_total(
    df: pd.DataFrame,
    cols_to_total: List[str] = COLS_TO_TOTAL,
    time_col: str = 'datetime',
    is_daylight: str = 'is_daylight'
) -> pd.DataFrame:
    """
    Добавляет колонки с суммарными значениями за световой день.
    
    Args:
        df: Исходный DataFrame
        cols_to_total: Список колонок для расчёта сумм
        time_col: Колонка с временными метками (datetime)
        is_daylight: Колонка с флагом daylight
        
    Returns:
        DataFrame с добавленными колонками total_*_daylight
        
    Raises:
        ValueError: Если отсутствуют необходимые колонки
    """
    # Проверка наличия колонок
    required_cols = cols_to_total + [time_col, is_daylight]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Отсутствуют колонки: {missing_cols}")
    
    # Проверка типа временной колонки
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Группировка для агрегации
    daily_grouper = pd.Grouper(key=time_col, freq='D')
    daylight_df = df[df[is_daylight]]
    
    # Для каждой колонки рассчитываем сумму за световой день
    for col in cols_to_total:
        # Вычисляем суммарные значения
        daylight_total = daylight_df.groupby(daily_grouper)[col].sum().reset_index()
        daylight_total.columns = [time_col, f'total_{remove_unit_suffix(col)}_daylight']
        
        # Преобразуем дату для корректного merge
        daylight_total['date'] = daylight_total[time_col].dt.date
        
        # Объединяем с исходным DataFrame
        df = df.merge(
            daylight_total[['date', f'total_{remove_unit_suffix(col)}_daylight']],
            on='date',
            how='left'
        )
    
    return df
    
def convert_to_iso_format(
    df: pd.DataFrame,
    cols_to_iso: List[str] = COLS_TO_ISO
) -> pd.DataFrame:
    """
    Конвертирует колонки с Unix time (в секундах) в строковый формат ISO 8601.
    
    Args:
        df: Исходный DataFrame
        cols_to_iso: Список колонок с Unix time (в секундах) для преобразования
        
    Returns:
        DataFrame с преобразованными колонками (оригинальные колонки будут перезаписаны)
        
    Raises:
        ValueError: Если отсутствуют указанные колонки
    """
    # Проверка наличия колонок
    missing_cols = [col for col in cols_to_iso if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Отсутствуют колонки: {missing_cols}")

    # Конвертация каждой колонки из Unix time (секунды) в ISO 8601
    for col in cols_to_iso:
        df[col + '_iso'] = pd.to_datetime(df[col], unit='s').dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    return df

def reorder_columns(
    df: pd.DataFrame,
    column_order: List[str] = RESULT_COLS,
    required_cols: List[str] = RESULT_COLS
) -> pd.DataFrame:
    """
    Упорядочивает колонки DataFrame согласно переданному списку после валидации.
    Требует точного соответствия column_order и required_cols.

    Args:
        df: Исходный DataFrame
        column_order: Желаемый порядок колонок (должен содержать все required_cols)
        required_cols: Список обязательных колонок для валидации

    Returns:
        DataFrame с измененным порядком колонок

    Raises:
        SystemExit: Если валидация колонок не пройдена
    """
    # 1. Обязательная валидация (вызовет SystemExit при ошибке)
    validate_dataframe_columns(df, required_cols)
    
    # 2. Возвращаем DataFrame с новым порядком
    return df[column_order]
    
def process_weather_data(daily_df: pd.DataFrame, 
                        hourly_df: pd.DataFrame,
                        utc_offset: int = 0,
                        cols_to_drop=[]) -> pd.DataFrame:
    """
    Основная функция обработки данных:
    1. Добавляет колонки с датой
    2. Добавляет человекопонятное время
    3. Объединяет данные
    
    Args:
        daily_df: DataFrame с дневными данными
        hourly_df: DataFrame с почасовыми данными
        utc_offset: Смещение временной зоны в секундах
        
    Returns:
        Обработанный объединенный DataFrame
    """
    # Добавляем колонки с датой
    daily_with_date = add_date_column(daily_df.copy())
    hourly_with_date = add_date_column(hourly_df.copy())
    
    # Добавляем человекопонятное время
    hourly_with_datetime = add_human_readable_datetime(hourly_with_date, utc_offset=utc_offset)
    
    # Объединяем данные
    merged_df = merge_daily_hourly(hourly_with_datetime, daily_with_date)
    
    # Конвертируем единицы измерения
    converted_df = convert_weather_units(merged_df)

    transformed_df = add_daylight_flag(converted_df)
    transformed_df = add_daylight_hours(transformed_df)
    transformed_df = convert_to_iso_format(transformed_df)
    transformed_df = calculate_24H_avg(df=transformed_df)
    transformed_df = calculate_daylight_avg(df=transformed_df)
    transformed_df = calculate_24H_total(df=transformed_df)
    transformed_df = calculate_daylight_total(df=transformed_df)
    transformed_df = remove_original_units(transformed_df)
    result = reorder_columns(transformed_df)
    
    return result
   