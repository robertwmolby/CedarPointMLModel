import psycopg2
import pandas as pd

def get_db_connection():
    try:
        return psycopg2.connect(
            host="cp-ai.cbsscwgeqp5j.us-east-2.rds.amazonaws.com",
            port=5432,
            database="postgres",
            user="postgres",
            password="CedarP0int"
        )
    except psycopg2.Error as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return None


def load_table(name: str) -> pd.DataFrame:
    conn = get_db_connection()
    if conn is None:
        raise RuntimeError("Could not connect to database.")

    df = pd.read_sql(f"SELECT * FROM {name}", conn)
    conn.close()
    return df
