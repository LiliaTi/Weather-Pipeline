#UTC_OFFSET_SECONDS = 25_200 # часовая разница между временем по Новосибирску и UTC+0 (в секундах)
MAX_DAYS = 10 # максимальный размер временного интервала для запроса к API (в днях)
URL = "https://api.open-meteo.com/v1/forecast" # API open-meteo
DEFAULT_PARAMS = { # Дефолтные параметры запроса к API
    "latitude": 55.0344,
    "longitude": 82.9434,
    "daily": "sunrise,sunset,daylight_duration",
    "hourly": ("temperature_2m,relative_humidity_2m,dew_point_2m,"
              "apparent_temperature,temperature_80m,temperature_120m,"
              "wind_speed_10m,wind_speed_80m,wind_direction_10m,"
              "wind_direction_80m,visibility,evapotranspiration,"
              "weather_code,soil_temperature_0cm,soil_temperature_6cm,"
              "rain,showers,snowfall"),
    "timezone": "auto",
    "timeformat": "unixtime",
    "wind_speed_unit": "kn",
    "temperature_unit": "fahrenheit",
    "precipitation_unit": "inch",
    "start_date": '2025-05-16',  # дефолтные даты
    "end_date": '2025-05-30'
}

COLS_TO_CELSIUS = [
 "temperature_2m",
 "dew_point_2m", # только для расчетов
 "apparent_temperature",
 "temperature_80m",
 "temperature_120m",
 "soil_temperature_0cm",
 "soil_temperature_6cm" 
]

COLS_TO_M_PER_S = [
"wind_speed_10m",
"wind_speed_80m",    
]

COLS_TO_M = [
"visibility"    
]

COLS_TO_MM = [
"rain", 
"showers",
"snowfall"    
]

COLS_TO_AVG = [
"temperature_2m_celsius", # temperature_2m_celsius, avg_temperature_2m_24h, avg_temperature_2m_daylight
"relative_humidity_2m", # %, avg_relative_humidity_2m_24h, avg_relative_humidity_2m_daylight
"dew_point_2m_celsius", # C, avg_dew_point_2m_24h, avg_dew_point_2m_daylight
"apparent_temperature_celsius", # apparent_temperature_celsius, avg_apparent_temperature_24h, avg_apparent_temperature_daylight
"temperature_80m_celsius", # temperature_80m_celsius, avg_temperature_80m_24h, avg_temperature_80m_daylight
"temperature_120m_celsius", # temperature_120m_celsius, avg_temperature_120m_24h, avg_temperature_120m_daylight
"wind_speed_10m_m_per_s", # wind_speed_10m_m_per_s, avg_wind_speed_10m_24h, avg_wind_speed_10m_daylight
"wind_speed_80m_m_per_s", # wind_speed_80m_m_per_s, avg_wind_speed_80m_24h, avg_wind_speed_80m_daylight
"visibility_m" # avg_visibility_24h, avg_visibility_daylight    
]

COLS_TO_TOTAL = [
"rain_mm", # rain_mm, total_rain_24h, total_rain_daylight
"showers_mm", # showers_mm, total_showers_24h, total_showers_daylight
"snowfall_mm", # snowfall_mm, total_snowfall_24h, total_snowfall_daylight    
]

COLS_TO_ISO = [
"sunrise",
"sunset"
]

COLS_TO_DROP = [
    'wind_direction_10m',
    'wind_direction_80m',
    'evapotranspiration',
    'weather_code',
    'time_daily',
    'time_hourly',
    #'datetime',
    'date', 
    'daylight_duration', 
    'is_daylight',
    'dew_point_2m_celsius', 
    'relative_humidity_2m', 
    'visibility_m'
]

RESULT_COLS = [
    'datetime', # добавила к Итоговоц таблице колонку чтобы использовать как индекс в гипертаблице timescaledb
    'avg_temperature_2m_24h',
    'avg_relative_humidity_2m_24h',
    'avg_dew_point_2m_24h',
    'avg_apparent_temperature_24h',
    'avg_temperature_80m_24h',
    'avg_temperature_120m_24h',
    'avg_wind_speed_10m_24h',
    'avg_wind_speed_80m_24h',
    'avg_visibility_24h',
    'total_rain_24h',
    'total_showers_24h',
    'total_snowfall_24h',
    'avg_temperature_2m_daylight',
    'avg_relative_humidity_2m_daylight',
    'avg_dew_point_2m_daylight',
    'avg_apparent_temperature_daylight',
    'avg_temperature_80m_daylight',
    'avg_temperature_120m_daylight',
    'avg_wind_speed_10m_daylight',
    'avg_wind_speed_80m_daylight',
    'avg_visibility_daylight',
    'total_rain_daylight',
    'total_showers_daylight',
    'total_snowfall_daylight',
    'wind_speed_10m_m_per_s',
    'wind_speed_80m_m_per_s',
    'temperature_2m_celsius',
    'apparent_temperature_celsius',
    'temperature_80m_celsius',
    'temperature_120m_celsius',
    'soil_temperature_0cm_celsius',
    'soil_temperature_6cm_celsius',
    'rain_mm',
    'showers_mm',
    'snowfall_mm',
    'daylight_hours',
    'sunset_iso',
    'sunrise_iso'
]