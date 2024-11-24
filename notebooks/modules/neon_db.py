from sqlalchemy import create_engine, text
import pandas as pd
import os
from pathlib import Path

# engine = create_engine(NEON_DATABASE_URL)
engine = create_engine(os.environ["NEON_DATABASE_URL"])

def run_neon_query(query):
    with engine.connect() as conn:
        result = conn.execute(text(query))
    df = pd.DataFrame(result.fetchall(), columns=result.keys())
    return df

def load_sql_query(filename):
    query_path = Path("../queries") / filename
    with open(query_path, "r") as f:
        return f.read()