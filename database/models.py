from sqlmodel import SQLModel, Field
from datetime import datetime

class Article(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    title: str
    link: str
    published: str
    source: str
    fetched_at: str

class Event(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    timestamp: str
    description: str
    keywords: str  # Comma-separated keywords
    cluster_id: int

class Summary(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    timestamp: str
    summary_text: str 