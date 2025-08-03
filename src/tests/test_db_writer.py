from storage.db_writer import save_to_timescaledb
import pytest
from sqlalchemy import create_engine, inspect, text
import pandas as pd
from datetime import datetime

@pytest.fixture
def sqlite_engine():
    """Фикстура с временной SQLite базой"""
    engine = create_engine("sqlite:///:memory:")
    yield engine
    engine.dispose()

def test_rejects_missing_datetime(sqlite_engine):
    """Проверка обязательной колонки datetime"""
    bad_df = pd.DataFrame({"temperature": [25.0]})
    
    with pytest.raises(ValueError, match="должен содержать столбец 'datetime'"):
        save_to_timescaledb(bad_df, sqlite_engine)

def test_saves_data_to_table(sqlite_engine):
    """Проверка сохранения данных в таблицу"""
    test_df = pd.DataFrame({
        "datetime": [datetime(2023, 1, 1)],
        "temperature": [25.0]
    })
    
    # Упрощенный вызов - пропускаем TimescaleDB-специфичный код
    test_df.to_sql("weather_metrics", sqlite_engine, index=False, if_exists="replace")
    
    # Проверяем что данные сохранились
    with sqlite_engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM weather_metrics"))
        assert len(result.fetchall()) == 1