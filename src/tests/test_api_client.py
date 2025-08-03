import pytest
import requests
import requests_mock
from api_client import fetch_weather_data
from validators import is_valid_time_interval


@pytest.fixture
def mock_weather_response():
    """Фикстура с моком успешного ответа API"""
    return {
        "latitude": 55.75,
        "longitude": 37.61,
        "hourly": {
            "temperature_2m": [20.5, 21.0, 20.7]
        }
    }

def test_fetch_weather_data_success(requests_mock, mock_weather_response):
    """Проверяет успешный запрос данных о погоде"""
    test_url = "https://api.test-meteo.com/v1/forecast"
    requests_mock.get(test_url, json=mock_weather_response, status_code=200)
    
    result = fetch_weather_data(base_url=test_url)
    
    assert isinstance(result, dict)
    assert "latitude" in result
    assert "hourly" in result
    assert requests_mock.call_count == 1

def test_fetch_weather_data_with_params(requests_mock, mock_weather_response):
    """Проверяет запрос с дополнительными параметрами"""
    test_url = "https://api.test-meteo.com/v1/forecast"
    requests_mock.get(test_url, json=mock_weather_response)
    
    params = {"latitude": 55.75, "longitude": 37.61}
    result = fetch_weather_data(base_url=test_url, params=params)
    
    assert result == mock_weather_response
    assert requests_mock.last_request.qs["latitude"][0] == "55.75"

def test_fetch_weather_data_with_interval(requests_mock, mock_weather_response):
    """Проверяет запрос с временным интервалом"""
    test_url = "https://api.test-meteo.com/v1/forecast"
    requests_mock.get(test_url, json=mock_weather_response)
    
    interval = ["2023-01-01", "2023-01-10"]
    result = fetch_weather_data(base_url=test_url, interval=interval)
    
    assert result == mock_weather_response
    assert requests_mock.last_request.qs["start_date"][0] == "2023-01-01"
    assert requests_mock.last_request.qs["end_date"][0] == "2023-01-10"

def test_fetch_weather_data_invalid_interval():
    """Проверяет обработку некорректного интервала"""
    with pytest.raises(ValueError):
        fetch_weather_data(interval=["2023-01-10", "2023-01-01"])

def test_fetch_weather_data_http_error(requests_mock):
    """Проверяет обработку ошибки HTTP-запроса"""
    test_url = "https://api.test-meteo.com/v1/forecast"
    requests_mock.get(test_url, status_code=500)
    
    with pytest.raises(requests.exceptions.RequestException):
        fetch_weather_data(base_url=test_url)