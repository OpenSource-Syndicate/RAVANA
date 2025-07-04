from modules.knowledge_compression.main import compress_knowledge
from sqlmodel import Session
from database.models import Summary

class KnowledgeService:
    def __init__(self, engine):
        self.engine = engine

    def compress_and_save_knowledge(self):
        logs = {"recent_interactions": ["Sample log entry"] * 10}  # Placeholder
        summary = compress_knowledge(logs)
        with Session(self.engine) as session:
            summary_obj = Summary(
                timestamp=summary["timestamp"],
                summary_text=summary["summary"]
            )
            session.add(summary_obj)
            session.commit()
        return summary 