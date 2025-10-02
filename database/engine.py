from sqlmodel import create_engine, SQLModel
from core.config import Config


def get_engine():
    config = Config()
    return create_engine(config.DATABASE_URL, echo=True)


def create_db_and_tables(engine):
    SQLModel.metadata.create_all(engine)
