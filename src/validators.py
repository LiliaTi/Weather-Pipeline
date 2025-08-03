from datetime import datetime
from typing import List, Tuple
import pandas as pd
from constants import MAX_DAYS
from constants import RESULT_COLS

def is_valid_time_interval(
    interval: List[str],
    max_days: int = MAX_DAYS
) -> Tuple[bool, str]:
    """
    Проверяет и валидирует временной интервал.
    
    Возвращает:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if not isinstance(interval, list):
        return (False, "Интервал должен быть списком")
    
    if len(interval) != 2:
        return (False, "Требуется ровно 2 даты [start_date, end_date]")
    
    if not all(interval):
        return (False, "Даты не могут быть пустыми")
    
    try:
        start = datetime.strptime(interval[0], '%Y-%m-%d').date()
        end = datetime.strptime(interval[1], '%Y-%m-%d').date()
        
        if start > end:
            return (False, "Начальная дата не может быть позже конечной")
            
        if (end - start).days + 1 > max_days: # обе даты включаются в интервал
            return (False, f"Интервал превышает максимально допустимый срок ({max_days} дней)")
            
        return (True, "")
        
    except ValueError:
        return (False, "Неверный формат даты (требуется YYYY-MM-DD)")



def validate_dataframe_columns(
    df: pd.DataFrame,
    required_cols: List[str] = RESULT_COLS,
) -> None:
    """
    Проверяет точное соответствие колонок датафрейма требованиям.
    
    Args:
        df: Проверяемый DataFrame
        
    Raises:
        SystemExit: Если обнаружены расхождения с ожидаемыми колонками
    """
    current_columns = set(df.columns)
    required_columns = set(required_cols)
    
    
    # Находим расхождения
    missing_columns = required_columns - current_columns
    extra_columns = current_columns - required_columns
    
    errors = []
    
    if missing_columns:
        errors.append(f"Отсутствуют обязательные колонки: {sorted(missing_columns)}")
    
    if extra_columns:
        errors.append(f"Обнаружены лишние колонки: {sorted(extra_columns)}")
    
    if errors:
        error_msg = "\n".join(errors)
        raise SystemExit(error_msg)
    
    print("Проверка колонок успешно пройдена: все колонки соответствуют требованиям")


