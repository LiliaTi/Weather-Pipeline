import requests
from constants import URL, DEFAULT_PARAMS
from typing import List, Dict, Any, Optional
import validators

def fetch_weather_data(
    base_url: str = URL,
    params: Optional[Dict[str, Any]] = None,
    interval: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Запрашивает данные о погоде с API Open-Meteo.
    
    Аргументы:
        base_url: URL API (по умолчанию Open-Meteo)
        params: Дополнительные параметры запроса (опционально)
        interval: Список из двух дат [start_date, end_date] в формате 'YYYY-MM-DD' (опционально)
    
    Возвращает:
        Словарь с данными о погоде в формате JSON.
    
    Исключения:
        ValueError: При некорректном интервале дат
        requests.exceptions.RequestException: При ошибке HTTP-запроса.
    """
    request_params = DEFAULT_PARAMS.copy()
    
    if params:
        request_params.update(params)
    
    if interval is not None:
        is_valid, error_msg = validators.is_valid_time_interval(interval)
        if not is_valid:
            raise ValueError(f"Некорректный интервал: {error_msg}")
            
        request_params.update({
            "start_date": interval[0],
            "end_date": interval[1]
        })
    
    try:
        response = requests.get(base_url, params=request_params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(f"Ошибка HTTP-запроса: {e}")