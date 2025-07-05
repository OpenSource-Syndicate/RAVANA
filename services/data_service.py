import asyncio
from typing import List
from sqlmodel import Session, select
from modules.information_processing.trend_analysis.trend_engine import fetch_feeds
from modules.event_detection.event_detector import process_data_for_events
from database.models import Article, Event, ActionLog, MoodLog, SituationLog, DecisionLog, ExperimentLog
from datetime import datetime
import json

class DataService:
    def __init__(self, engine, feed_urls: List[str]):
        self.engine = engine
        self.feed_urls = feed_urls

    def fetch_and_save_articles(self):
        new_articles = fetch_feeds(self.feed_urls)
        if new_articles:
            with Session(self.engine) as session:
                for article_data in new_articles:
                    statement = select(Article).where(Article.link == article_data['link'])
                    if not session.exec(statement).first():
                        article = Article(
                            title=article_data['title'],
                            link=article_data['link'],
                            published=article_data.get('published', datetime.utcnow().isoformat()),
                            source=article_data['source'],
                            fetched_at=datetime.utcnow().isoformat()
                        )
                        session.add(article)
                session.commit()
            return len(new_articles)
        return 0

    def detect_and_save_events(self):
        with Session(self.engine) as session:
            stmt = select(Article).order_by(Article.fetched_at.desc()).limit(10)
            articles = session.exec(stmt).all()
            if articles:
                texts = [article.title + " " + article.link for article in articles]
                result = process_data_for_events(texts)
                events = result.get("events", [])
                for event in events:
                    event_obj = Event(
                        timestamp=event.get("timestamp", datetime.utcnow().isoformat()),
                        description=event.get("description", "No description"),
                        keywords=",".join(event.get("keywords", [])),
                        cluster_id=event.get("cluster_id", -1)
                    )
                    session.add(event_obj)
                session.commit()
                return len(events)
        return 0

    def save_action_log(self, action_name: str, params: dict, status: str, result: any):
        """Saves a record of an executed action to the database."""
        with Session(self.engine) as session:
            action_log = ActionLog(
                timestamp=datetime.utcnow().isoformat(),
                action_name=action_name,
                params=json.dumps(params),
                status=status,
                result=json.dumps(result)
            )
            session.add(action_log)
            session.commit()

    def save_mood_log(self, mood_vector: dict):
        """Saves a record of the current mood vector."""
        with Session(self.engine) as session:
            mood_log = MoodLog(
                timestamp=datetime.utcnow().isoformat(),
                mood_vector=json.dumps(mood_vector)
            )
            session.add(mood_log)
            session.commit()

    def save_situation_log(self, situation: dict) -> int:
        """Saves a record of a generated situation and returns its ID."""
        with Session(self.engine) as session:
            situation_log = SituationLog(
                timestamp=datetime.utcnow().isoformat(),
                situation_type=situation.get('type', 'unknown'),
                prompt=situation.get('prompt', ''),
                context=json.dumps(situation.get('context', {}))
            )
            session.add(situation_log)
            session.commit()
            session.refresh(situation_log)
            return situation_log.id

    def save_decision_log(self, situation_id: int, raw_response: str):
        """Saves a record of a decision made by the AGI."""
        with Session(self.engine) as session:
            decision_log = DecisionLog(
                timestamp=datetime.utcnow().isoformat(),
                situation_id=situation_id,
                raw_response=raw_response
            )
            session.add(decision_log)
            session.commit()

    def save_experiment_log(self, hypothesis: str, results: dict):
        """Saves a record of a self-reflection experiment."""
        with Session(self.engine) as session:
            experiment_log = ExperimentLog(
                timestamp=datetime.utcnow().isoformat(),
                hypothesis=hypothesis,
                results=json.dumps(results)
            )
            session.add(experiment_log)
            session.commit() 