import pandas as pd
import os
import psycopg2
from psycopg2.extensions import connection as PGConnection

def get_db_connection() -> PGConnection | None:
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT", "5432"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
        )
        return conn

    except psycopg2.Error as e:
        print(f"Error connecting to PostgreSQL database: {e}")
        return None



def load_table(name: str) -> pd.DataFrame:
    conn = get_db_connection()
    if conn is None:
        raise RuntimeError("Could not connect to database.")

    df = pd.read_sql(f"SELECT * FROM {name}", conn)
    conn.close()
    return df
