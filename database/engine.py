from sqlmodel import create_engine, SQLModel
from core.config import Config

engine = create_engine(Config.DATABASE_URL, echo=True)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine) 