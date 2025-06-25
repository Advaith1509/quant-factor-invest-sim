import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from urllib.parse import quote_plus

load_dotenv()
def get_engine():
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    db = os.getenv("DB_NAME")
    if not all([user, password, host, port, db]):
        raise ValueError("database connection parameters are not set correctly in the env vars.")
    
    encoded_password = quote_plus(password)
    db_url = f"postgresql+psycopg2://{user}:{encoded_password}@{host}:{port}/{db}"
    engine = create_engine(db_url)
    return engine

