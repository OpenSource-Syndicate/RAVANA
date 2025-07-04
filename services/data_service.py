import asyncio
from typing import List
from sqlmodel import Session, select
from modules.information_processing.trend_analysis.trend_engine import fetch_feeds
from modules.event_detection.event_detector import process_data_for_events
from database.models import Article, Event
from datetime import datetime

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