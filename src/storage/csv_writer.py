import pandas as pd
from datetime import datetime
import os
from pathlib import Path

def save_to_csv(df: pd.DataFrame, 
                filename: str = "weather_data", 
                output_dir: str = "data/output") -> str:
    """
    Сохраняет DataFrame в CSV файл с автоматическим именем
    
    Параметры:
        df: DataFrame с данными для сохранения
        filename: Базовое имя файла (без расширения)
        output_dir: Директория для сохранения
        
    Возвращает:
        Полный путь к сохраненному файлу
        
    Пример:
        save_to_csv(weather_df, "weather_2023", "data/export")
    """
    # Создаем директорию, если ее нет
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Генерируем имя файла с timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{filename}_{timestamp}.csv"
    full_path = os.path.join(output_dir, csv_filename)
    
    # Сохраняем в CSV
    df.to_csv(full_path, index=False, encoding='utf-8')
    
    print(f"Данные сохранены в: {full_path}")
    return full_path