from pathlib import Path
PROJECT_ROOT = Path.cwd().parent
from constants import COLS_TO_AVG, RESULT_COLS
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch
from transformations import (  
    parse_unix_time,
    validate_timestamps,
    extract_weather_data,
    add_date_column,
    add_human_readable_datetime,
    merge_daily_hourly,
    convert_weather_units,
    add_daylight_flag,
    add_daylight_hours,
    calculate_24H_avg,
    calculate_daylight_avg,
    calculate_24H_total,
    calculate_daylight_total,
    convert_to_iso_format,
    reorder_columns,
    process_weather_data
)


# ---------- Fixtures ----------
@pytest.fixture
def sample_daily_data():
    """Полный набор daily данных"""
    return pd.DataFrame({
        'time': [1625097600, 1625184000],
        'sunrise': [1625126400, 1625212800],
        'sunset': [1625176800, 1625263200],
        'daylight_duration': [43200, 43200]
       
    })

@pytest.fixture
def sample_hourly_data():
    """Полный набор hourly данных с точно указанными полями"""
    return pd.DataFrame({
        'time': [1625097600, 1625101200, 1625104800],  # Unix-время
        'temperature_2m': [68.0, 70.0, 72.0],  # Температура на 2м
        'relative_humidity_2m': [65, 67, 70],  # Влажность на 2м
        'dew_point_2m': [45.0, 47.0, 50.0],  # Точка росы на 2м
        'apparent_temperature': [66.0, 68.0, 71.0],  # Кажущаяся температура
        'temperature_80m': [70.0, 72.0, 74.0],  # Температура на 80м
        'temperature_120m': [72.0, 74.0, 76.0],  # Температура на 120м
        'wind_speed_10m': [5.0, 6.0, 7.0],  # Скорость ветра на 10м
        'wind_speed_80m': [8.0, 9.0, 10.0],  # Скорость ветра на 80м
        'wind_direction_10m': [90, 95, 100],  # Направление ветра на 10м
        'wind_direction_80m': [95, 100, 105],  # Направление ветра на 80м
        'visibility': [50000.0, 55000.0, 60000.0],  # Видимость
        'evapotranspiration': [0.001, 0.001, 0.002],  # Испаряемость
        'weather_code': [3, 3, 4],  # Код погоды
        'soil_temperature_0cm': [60.0, 62.0, 64.0],  # Температура почвы 0см
        'soil_temperature_6cm': [62.0, 64.0, 66.0],  # Температура почвы 6см
        'rain': [0.1, 0.0, 0.2],  # Дождь
        'showers': [0.0, 0.1, 0.0],  # Ливни
        'snowfall': [0.0, 0.0, 0.0]  # Снегопад
    })
@pytest.fixture
def sample_api_response():
    """Фейковый ответ API в формате Open-Meteo"""
    return {
        'daily': {
            'time': [1625097600, 1625184000],
            'sunrise': [1625126400, 1625212800],
            'sunset': [1625176800, 1625263200],
            'daylight_duration': [43200, 43200]
        },
        'hourly': {
            'time': [1625097600, 1625101200, 1625104800],  # Unix-время
            'temperature_2m': [68.0, 70.0, 72.0],  # Температура на 2м
            'relative_humidity_2m': [65, 67, 70],  # Влажность на 2м
            'dew_point_2m': [45.0, 47.0, 50.0],  # Точка росы на 2м
            'apparent_temperature': [66.0, 68.0, 71.0],  # Кажущаяся температура
            'temperature_80m': [70.0, 72.0, 74.0],  # Температура на 80м
            'temperature_120m': [72.0, 74.0, 76.0],  # Температура на 120м
            'wind_speed_10m': [5.0, 6.0, 7.0],  # Скорость ветра на 10м
            'wind_speed_80m': [8.0, 9.0, 10.0],  # Скорость ветра на 80м
            'wind_direction_10m': [90, 95, 100],  # Направление ветра на 10м
            'wind_direction_80m': [95, 100, 105],  # Направление ветра на 80м
            'visibility': [50000.0, 55000.0, 60000.0],  # Видимость
            'evapotranspiration': [0.001, 0.001, 0.002],  # Испаряемость
            'weather_code': [3, 3, 4],  # Код погоды
            'soil_temperature_0cm': [60.0, 62.0, 64.0],  # Температура почвы 0см
            'soil_temperature_6cm': [62.0, 64.0, 66.0],  # Температура почвы 6см
            'rain': [0.1, 0.0, 0.2],  # Дождь
            'showers': [0.0, 0.1, 0.0],  # Ливни
            'snowfall': [0.0, 0.0, 0.0]  # Снегопад
        },
        'utc_offset_seconds': 10800  # UTC+3
    }

# ---------- Tests for parse_unix_time ----------
def test_parse_unix_time_variations():
    """Только секунды, без миллисекунд"""
    test_cases = [
        (1625097600, 0, 0),    # UTC
        (1625097600, 3600, 1),  # UTC+1
        (1625097600, -14400, 20)  # UTC-4
    ]
    
    for timestamp, offset, expected_hour in test_cases:
        result = parse_unix_time(timestamp, offset)
        assert result.hour == expected_hour

def test_parse_unix_time_invalid():
    """Проверяет обработку невалидных входных данных"""
    with pytest.raises(TypeError):
        parse_unix_time("not_a_number")
    
    with pytest.raises(ValueError):
        parse_unix_time(-100)  # отрицательное время

# ---------- Tests for extract_weather_data ----------
def test_extract_weather_data_success(sample_api_response):
    """Проверяет корректное извлечение данных из API ответа"""
    result = extract_weather_data(sample_api_response)
    
    assert isinstance(result['daily'], pd.DataFrame)
    assert isinstance(result['hourly'], pd.DataFrame)
    assert result['utc_offset_seconds'] == 10800
    assert len(result['daily']) == 2
    assert len(result['hourly']) == 3

def test_extract_weather_data_missing_keys():
    """Проверяет обработку ответа с отсутствующими обязательными полями"""
    with pytest.raises(KeyError):
        extract_weather_data({'invalid': 'data'})

# ---------- Tests for DataFrame processing ----------
def test_add_date_column_transformation(sample_hourly_data, sample_api_response):
    """Проверяет что:
    - Колонка 'date' добавляется корректно
    - Дата соответствует временной метке
    - Исходные данные не изменяются
    """
    original_columns = set(sample_hourly_data.columns)
    utc_offset = sample_api_response['utc_offset_seconds']
  
    # Функция ДОЛЖНА прибавлять offset (time в UTC)
    result_with_offset = add_date_column(df=sample_hourly_data.copy(), utc_offset=utc_offset)
    expected_date = pd.Timestamp(1625097600 + utc_offset, unit='s').date()
    assert result_with_offset['date'].iloc[0] == expected_date

def test_merge_daily_hourly_integration(sample_daily_data, sample_hourly_data):
    """Проверяет базовую логику объединения daily и hourly данных"""
    # Подготовка
    daily = add_date_column(sample_daily_data.copy())
    hourly = add_date_column(sample_hourly_data.copy())
    
    # Вызов
    result = merge_daily_hourly(hourly, daily)
    
    # 1. Проверка сохранения структуры
    assert len(result) == len(hourly), "Количество строк должно совпадать с hourly данными"
    
    # 2. Проверка ключевых колонок
    assert 'date' in result.columns, "Колонка 'date' должна присутствовать"
    assert 'time_hourly' in result.columns, "Исходные временные метки hourly должны сохраниться"
    
    # 3. Проверка что нужные суффиксы прибавились
    assert any(col.endswith('_daily') for col in result.columns), "Должны быть колонки из daily данных"
    assert any(col.endswith('_hourly') for col in result.columns), "Должны быть колонки из hourly данных"

# ---------- Tests for unit conversions ----------
def test_convert_weather_units_transformations(sample_hourly_data):
    """Проверяет преобразование всех типов единиц"""
    result = convert_weather_units(sample_hourly_data.copy())
    
    # Проверка преобразования температуры
    assert 'temperature_2m_celsius' in result.columns
    assert np.isclose(result['temperature_2m_celsius'].iloc[0], 20.0)  # 68F → 20C
    
    # Проверка преобразования скорости ветра
    if 'wind_speed_10m' in sample_hourly_data.columns:
        assert 'wind_speed_10m_m_per_s' in result.columns
        assert np.isclose(result['wind_speed_10m_m_per_s'].iloc[0], 5 * 0.514444)

def test_convert_weather_units_edge_cases():
    """Проверяет обработку специальных случаев:
    - Пустой DataFrame
    - Пропущенные значения
    """
  # Проверка что пустой DataFrame вызывает исключение
    empty_df = pd.DataFrame(columns=['temperature_2m'])
    with pytest.raises(ValueError):  # или другое исключение, которое выбрасывает ваша функция
        convert_weather_units(empty_df)

# ---------- Tests for daylight calculations ----------
def test_daylight_flag_calculation(sample_daily_data, sample_hourly_data):
    """Проверяет корректность расчета флага daylight"""
    merged = merge_daily_hourly(
        add_date_column(sample_daily_data),
        add_date_column(sample_hourly_data)
    )
    
    result = add_daylight_flag(merged)
    assert 'is_daylight' in result.columns
    assert result['is_daylight'].dtype == bool
    # Проверяем что ночное время правильно помечено
    assert not result.loc[result['time_hourly'] < result['sunrise'], 'is_daylight'].any()

# ---------- Tests for aggregation functions ----------
def test_24h_avg_calculation():
    """Проверяет расчет среднесуточных показателей"""
    # Создаем тестовые данные с почасовыми значениями за 3 дня
    np.random.seed(42)
    date_range = pd.date_range(start="2023-01-01", end="2023-01-03 23:00:00", freq="h")
    
    # Генерируем тестовые данные с суточными колебаниями
    test_data = {
        "datetime": date_range,
        "temperature_2m_celsius": 10 + 5*np.sin(np.arange(len(date_range))/12) + np.random.normal(0, 1, len(date_range)),
        "relative_humidity_2m": np.clip(50 + 20*np.sin(np.arange(len(date_range))/12) + np.random.normal(0, 5, len(date_range)), 0, 100),
        "dew_point_2m_celsius": 5 + 3*np.sin(np.arange(len(date_range))/12) + np.random.normal(0, 0.5, len(date_range)),
        "apparent_temperature_celsius": 9 + 6*np.sin(np.arange(len(date_range))/12) + np.random.normal(0, 1, len(date_range)),
        "temperature_80m_celsius": 8 + 4*np.sin(np.arange(len(date_range))/12) + np.random.normal(0, 1, len(date_range)),
        "temperature_120m_celsius": 7 + 3*np.sin(np.arange(len(date_range))/12) + np.random.normal(0, 1, len(date_range)),
        "wind_speed_10m_m_per_s": np.clip(3 + 2*np.sin(np.arange(len(date_range))/12) + np.random.normal(0, 1, len(date_range)), 0, None),
        "wind_speed_80m_m_per_s": np.clip(6 + 3*np.sin(np.arange(len(date_range))/12) + np.random.normal(0, 1.5, len(date_range)), 0, None),
        "visibility_m": np.clip(10000 + 2000*np.sin(np.arange(len(date_range))/12) + np.random.normal(0, 500, len(date_range)), 0, None)
    }
    
    df = pd.DataFrame(test_data)
    
    # Вызываем тестируемую функцию
    result = calculate_24H_avg(df)
    
    # Проверяем что добавились все ожидаемые колонки
    expected_avg_columns = [
        'avg_temperature_2m_24h',
        'avg_relative_humidity_2m_24h',
        'avg_dew_point_2m_24h',
        'avg_apparent_temperature_24h',
        'avg_temperature_80m_24h',
        'avg_temperature_120m_24h',
        'avg_wind_speed_10m_24h',
        'avg_wind_speed_80m_24h',
        'avg_visibility_24h'
    ]
    
    for col in expected_avg_columns:
        assert col in result.columns, f"Отсутствует колонка {col}"
    
    # Проверяем что средние значения рассчитаны правильно
    # Для этого сгруппируем данные по дням и сравним с результатом функции
    daily_avg = df.groupby(df['datetime'].dt.date).mean()
    
    for orig_col, avg_col in zip(COLS_TO_AVG, expected_avg_columns):
        # Берем первый день для проверки
        expected_value = daily_avg[orig_col].iloc[0]
        # Берем первое значение из результатов (должно совпадать для всех записей этого дня)
        actual_value = result[avg_col].iloc[0]
        
        assert np.isclose(expected_value, actual_value, rtol=1e-3), (
            f"Неверное среднее для {avg_col}. Ожидалось {expected_value}, получено {actual_value}"
        )
    
    # Проверяем что все значения для одного дня одинаковы
    for day in df['datetime'].dt.date.unique():
        day_mask = result['datetime'].dt.date == day
        for avg_col in expected_avg_columns:
            day_values = result.loc[day_mask, avg_col]
            assert day_values.nunique() == 1, (
                f"Значения {avg_col} должны быть одинаковы в пределах одного дня"
            )

# Дополнительные тесты для проверки обработки ошибок
def test_24h_avg_missing_columns():
    """Проверяет обработку случая с отсутствующими колонками"""
    df = pd.DataFrame({
        'datetime': pd.date_range("2023-01-01", periods=24, freq="h"),
        'temperature_2m_celsius': range(24)
    })
    
    with pytest.raises(ValueError, match="Отсутствуют колонки"):
        calculate_24H_avg(df)


# ---------- Test full pipeline ----------
def test_full_processing_pipeline(sample_hourly_data, sample_daily_data):
    """Комплексный тест всего процесса обработки"""
    result = process_weather_data(sample_hourly_data, sample_daily_data)
    
    # Проверяем что нежелательные колонки удалены
    assert 'temperature_2m' not in result.columns
    assert 'time_hourly' not in result.columns
    
    # Проверяем что данные не потеряны
    #...