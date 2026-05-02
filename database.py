from sqlalchemy import create_engine
import os

DB_PATH = 'data/trialmitra.db'
os.makedirs('data', exist_ok=True)

# Create a generic SQLAlchemy engine
engine = create_engine(f'sqlite:///{DB_PATH}', echo=False)

def get_engine():
    return engine
