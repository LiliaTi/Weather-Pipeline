import argparse
from pathlib import Path
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
import json

from api_client import fetch_weather_data
from transformations import extract_weather_data, process_weather_data
from storage.csv_writer import save_to_csv
from storage.db_writer import save_to_timescaledb


# Инициализация БД
load_dotenv()
engine = create_engine(
    f"postgresql+psycopg2://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@"
    f"{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
)

def process_json_to_db():
    """Загружает JSON из data/input в БД"""
    input_path = Path('/app/data/input/weather.json')
    data = json.loads(input_path.read_text())
    extracted = extract_weather_data(data)
    transformed = process_weather_data(
        daily_df=extracted['daily'],
        hourly_df=extracted['hourly'],
        utc_offset=extracted['utc_offset_seconds']
    )
    save_to_timescaledb(transformed, engine)
    print("Данные из data/input/weather.json загружены в БД")

def process_json_to_csv():
    """Сохраняет JSON из data/input в CSV"""
    input_path = Path('data/input/weather.json')
    data = json.loads(input_path.read_text())
    extracted = extract_weather_data(data)
    transformed = process_weather_data(
        daily_df=extracted['daily'],
        hourly_df=extracted['hourly'],
        utc_offset=extracted['utc_offset_seconds']
    )
    save_to_csv(transformed)
    print("Данные сохранены в data/output/result.csv")

def process_api_to_db(start_date: str, end_date: str):
    """Загружает данные API в БД"""
    api_data = fetch_weather_data(interval=[start_date, end_date])
    extracted = extract_weather_data(api_data)
    transformed = process_weather_data(
        daily_df=extracted['daily'],
        hourly_df=extracted['hourly'],
        utc_offset=extracted['utc_offset_seconds']
    )
    save_to_timescaledb(transformed, engine)
    print(f"Данные за период с {start_date} по {end_date} загружены в БД")

def process_api_to_csv(start_date: str, end_date: str):
    """Сохраняет данные API в CSV"""
    api_data = fetch_weather_data(interval=[start_date, end_date])
    extracted = extract_weather_data(api_data)
    transformed = process_weather_data(
        daily_df=extracted['daily'],
        hourly_df=extracted['hourly'],
        utc_offset=extracted['utc_offset_seconds']
    )
    save_to_csv(transformed)
    print(f"Данные за период с {start_date} по {end_date} сохранены в CSV")

def main():
    parser = argparse.ArgumentParser(description="ETL для погодных данных")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Режимы работы
    subparsers.add_parser("json_to_db", help="JSON из data/input → БД")
    subparsers.add_parser("json_to_csv", help="JSON из data/input → CSV")
    
    api_db_parser = subparsers.add_parser("api_to_db", help="API → БД")
    api_db_parser.add_argument("--start-date", required=True, help="Начальная дата (YYYY-MM-DD)")
    api_db_parser.add_argument("--end-date", required=True, help="Конечная дата (YYYY-MM-DD)")

    api_csv_parser = subparsers.add_parser("api_to_csv", help="API → CSV")
    api_csv_parser.add_argument("--start-date", required=True, help="Начальная дата (YYYY-MM-DD)")
    api_csv_parser.add_argument("--end-date", required=True, help="Конечная дата (YYYY-MM-DD)")

    args = parser.parse_args()

    if args.command == "json_to_db":
        process_json_to_db()
    elif args.command == "json_to_csv":
        process_json_to_csv()
    elif args.command == "api_to_db":
        process_api_to_db(args.start_date, args.end_date)
    elif args.command == "api_to_csv":
        process_api_to_csv(args.start_date, args.end_date)

if __name__ == "__main__":
    main()