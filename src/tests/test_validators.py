import pytest
import pandas as pd
from datetime import datetime
from validators import is_valid_time_interval, validate_dataframe_columns

def test_is_valid_time_interval_success():
    """Проверяет корректные временные интервалы"""
    # Корректный интервал
    result, msg = is_valid_time_interval(['2023-01-01', '2023-01-10'])
    assert result is True
    assert msg == ""

    # Граничный случай (максимальное количество дней)
    result, msg = is_valid_time_interval(['2023-01-01', '2023-01-31'], max_days=31)
    assert result is True
    assert msg == ""

def test_is_valid_time_interval_failures():
    """Проверяет обработку некорректных интервалов"""
    # Не список (передаем строку вместо списка)
    result, msg = is_valid_time_interval("2023-01-01,2023-01-10")
    assert result is False
    assert "Интервал должен быть списком" in msg

    # Не 2 элемента
    result, msg = is_valid_time_interval(['2023-01-01'])
    assert result is False
    assert "Требуется ровно 2 даты" in msg

    # Пустые даты
    result, msg = is_valid_time_interval(['', '2023-01-10'])
    assert result is False
    assert "Даты не могут быть пустыми" in msg

    # Неправильный порядок дат
    result, msg = is_valid_time_interval(['2023-01-10', '2023-01-01'])
    assert result is False
    assert "Начальная дата не может быть позже конечной" in msg

    # Слишком большой интервал
    result, msg = is_valid_time_interval(['2023-01-01', '2023-02-01'], max_days=30)
    assert result is False
    assert "Интервал превышает максимально допустимый срок (30 дней)" in msg

    # Неверный формат даты
    result, msg = is_valid_time_interval(['01-01-2023', '2023-01-10'])
    assert result is False
    assert "Неверный формат даты (требуется YYYY-MM-DD)" in msg

def test_validate_dataframe_columns_success():
    """Проверяет успешную валидацию колонок"""
    test_df = pd.DataFrame(columns=['col1', 'col2', 'col3'])
    # Должен завершиться без ошибок
    validate_dataframe_columns(test_df, required_cols=['col1', 'col2', 'col3'])

def test_validate_dataframe_columns_failures():
    """Проверяет обработку ошибок валидации колонок"""
    test_df = pd.DataFrame(columns=['col1', 'col2'])
    
    # Проверяем что функция завершает программу при ошибках
    with pytest.raises(SystemExit):
        validate_dataframe_columns(test_df, required_cols=['col1', 'col2', 'col3'])
    
    with pytest.raises(SystemExit):
        validate_dataframe_columns(test_df, required_cols=['col1'])

def test_validate_dataframe_columns_error_messages():
    """Проверяет корректность сообщений об ошибках"""
    test_df = pd.DataFrame(columns=['col1', 'col3', 'extra_col'])
    
    with pytest.raises(SystemExit) as excinfo:
        validate_dataframe_columns(test_df, required_cols=['col1', 'col2'])
    
    error_msg = str(excinfo.value)
    assert "Отсутствуют обязательные колонки" in error_msg
    assert "col2" in error_msg
    assert "Обнаружены лишние колонки" in error_msg
    assert "extra_col" in error_msg
    assert "col3" in error_msg