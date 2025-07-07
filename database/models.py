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
    source: str | None = Field(default="unknown")
    category: str | None = Field(default="misc")
    content_hash: str | None = Field(default=None)  # For deduplication

class ActionLog(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    timestamp: str
    action_name: str
    params: str  # JSON string of parameters
    status: str # 'success' or 'failure'
    result: str # JSON string of the result

class MoodLog(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    timestamp: str
    mood_vector: str  # JSON string of the mood vector

class SituationLog(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    timestamp: str
    situation_type: str
    prompt: str
    context: str # JSON string of the context

class DecisionLog(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    timestamp: str
    situation_id: int | None = Field(default=None, foreign_key="situationlog.id")
    raw_response: str

class ExperimentLog(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    timestamp: str
    hypothesis: str
    results: str # JSON string of the results 