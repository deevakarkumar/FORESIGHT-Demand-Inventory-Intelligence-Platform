import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Load DATABASE_URL from .env in project root
BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=BASE_DIR / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    try:
        import streamlit as st
        DATABASE_URL = st.secrets.get("DATABASE_URL")
    except Exception:
        DATABASE_URL = None

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set. Add it to your .env file or Streamlit Secrets.")

# Normalize postgres URL to use psycopg v3 driver if needed
if DATABASE_URL:
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
    elif DATABASE_URL.startswith("postgresql://"):
        DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

def save_data_to_db(df, table_name):

    df = df.copy()

    logger.info("Saving %s rows to %s...", len(df), table_name)

    df.to_sql(
        table_name,
        engine,
        schema="public",
        if_exists="replace",
        index=False
    )

    logger.info("Data saved to %s successfully!", table_name)
