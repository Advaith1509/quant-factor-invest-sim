import pandas as pd
from utils.db import get_engine

def load_table(table_name):
    engine = get_engine()
    query = f"SELECT * FROM {table_name};"
    return pd.read_sql(query, engine)

if __name__ == "__main__":
    prices = load_table("daily_prices")
    print(prices.head())