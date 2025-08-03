from sqlalchemy import inspect, text
import pandas as pd

def save_to_timescaledb(df, engine, table_name="weather_metrics"):
    """
    Сохраняет данные в гипертаблицу с индексом datetime с проверкой существования таблицы и пропуском дубликатов
    
    Параметры:
        df: DataFrame с данными (должен содержать столбец 'datetime')
        engine: SQLAlchemy engine
        table_name: имя таблицы в БД
    """
    if 'datetime' not in df.columns:
        raise ValueError("DataFrame должен содержать столбец 'datetime'")

    with engine.begin() as conn:
        inspector = inspect(engine)
        table_exists = inspector.has_table(table_name)

        if not table_exists:
            # Создаем таблицу если не существует
            df.head(0).to_sql(
                name=table_name,
                con=conn,
                if_exists='fail',
                index=False
            )
            
            # Создаем гипертаблицу и уникальный индекс
            conn.execute(text(f"""
                SELECT create_hypertable(
                    '{table_name}'::regclass, 
                    'datetime',
                    chunk_time_interval => INTERVAL '7 days',
                    if_not_exists => TRUE
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_{table_name}_datetime 
                ON {table_name} (datetime);
            """))
            print(f"Создана гипертаблица {table_name} с уникальным индексом")
        else:
            # Проверяем что это гипертаблица (исправленный запрос)
            is_hypertable = conn.execute(
                text("""
                    SELECT 1 
                    FROM timescaledb_information.hypertables 
                    WHERE hypertable_name = :table_name
                """),
                {'table_name': table_name}
            ).scalar()
            
            if not is_hypertable:
                raise ValueError(f"Таблица {table_name} не является гипертаблицей")

            # Проверяем наличие уникального индекса (исправленный запрос)
            has_unique_index = conn.execute(
                text("""
                    SELECT 1 
                    FROM pg_indexes 
                    WHERE tablename = :table_name 
                    AND indexdef LIKE '%%UNIQUE%%' 
                    AND indexdef LIKE '%%datetime%%'
                """),
                {'table_name': table_name}
            ).scalar()
            
            if not has_unique_index:
                conn.execute(text(f"""
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_{table_name}_datetime 
                    ON {table_name} (datetime);
                """))

        # Оптимизированная вставка с обработкой дубликатов
        column_names = ', '.join([f'"{col}"' for col in df.columns])
        placeholders = ', '.join([f':{col}' for col in df.columns])
        
        conn.execute(
            text(f"""
                INSERT INTO {table_name} ({column_names}) 
                VALUES ({placeholders})
                ON CONFLICT (datetime) DO NOTHING
            """),
            [dict(row) for _, row in df.iterrows()]
        )
    
    print(f"Записи добавлены в {table_name} (дубликаты пропущены)")