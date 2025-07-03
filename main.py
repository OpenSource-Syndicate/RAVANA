import asyncio
import logging
import sys
import os
import random
from datetime import datetime
from typing import List, Dict, Any
import json

# Add modules directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.decision_engine.llm import decision_maker_loop, generate_hypothetical_scenarios
from modules.knowledge_compression.main import compress_knowledge
from modules.information_processing.youtube_transcription.youtube_transcription import transcribe_youtube_video
from modules.information_processing.trend_analysis.trend_engine import fetch_feeds, analyze_trends
from modules.event_detection.event_detector import process_data_for_events, load_models as load_event_models
from modules.episodic_memory.memory import (
    extract_memories_api,
    save_memories,
    get_relevant_memories_api,
    consolidate_memories_api,
    startup_event as init_memory_db,
    app as memory_app
)
from modules.agent_self_reflection.self_modification import (
    generate_hypothesis,
    design_and_run_experiment,
    run_experiment_from_prompt
)
from modules.situation_generator.situation_generator import SituationGenerator
from modules.emotional_intellegence.emotional_intellegence import EmotionalIntelligence
from modules.curiosity_trigger.curiosity_trigger import CuriosityTrigger
from modules.agi_experimentation.llm import agi_experimentation_engine
from sentence_transformers import SentenceTransformer
from sqlmodel import SQLModel, Session, create_engine, select, Field

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# SQLite database setup
DATABASE_URL = "sqlite:///ravana_agi.db"
engine = create_engine(DATABASE_URL, echo=True)

# Define database models
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

# Create tables if they don't exist
SQLModel.metadata.create_all(engine)

class EnhancedSituationGenerator(SituationGenerator):
    def generate_situation(self) -> Dict[str, str]:
        """Enhanced situation generator that occasionally uses recent events."""
        if random.random() < 0.2:  # 20% chance to use a recent event
            with Session(engine) as session:
                # Fetch the most recent event
                stmt = select(Event).order_by(Event.timestamp.desc()).limit(1)
                event = session.exec(stmt).first()
                if event:
                    scenarios = generate_hypothetical_scenarios(trends=[event.description])
                    if scenarios:
                        return {'type': 'event_based', 'prompt': scenarios[0]}
        # Default situation generation (fallback)
        return {'type': 'default', 'prompt': 'Think about a random topic.'}

class AGISystem:
    def __init__(self):
        logger.info("Initializing Ravana AGI System...")
        
        # Load shared models
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        load_event_models(embedding_model_instance=self.embedding_model)
        memory_app.embedding_model = self.embedding_model

        # Initialize modules
        self.situation_generator = EnhancedSituationGenerator()
        self.emotional_intelligence = EmotionalIntelligence()
        self.episodic_memory = init_memory_db()
        self.reflection_module = type("ReflectionModule", (), {
            "generate_hypothesis": generate_hypothesis
        })
        self.experimentation_module = type("ExperimentationModule", (), {
            "design_and_run_experiment": design_and_run_experiment,
            "run_experiment_from_prompt": run_experiment_from_prompt
        })

        # Shared state
        self.shared_state = {
            "mood": self.emotional_intelligence.get_mood_vector(),
            "current_situation": None,
            "recent_memories": [],
            "long_term_goals": [],
            "mood_history": [],
        }

        # Feed URLs for data collection
        self.feed_urls = [
            "http://rss.cnn.com/rss/cnn_latest.rss",
            "https://feeds.bbci.co.uk/news/rss.xml",
            # Add more feed URLs as needed
        ]

    async def data_collection_task(self):
        """Background task to fetch articles from RSS feeds every hour."""
        while True:
            try:
                logger.info("Fetching feeds...")
                new_articles = fetch_feeds(self.feed_urls)
                
                if new_articles:
                    with Session(engine) as session:
                        for article_data in new_articles:
                            # Check if article already exists
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
                    logger.info(f"Feeds fetched and {len(new_articles)} new articles saved.")
                else:
                    logger.info("No new articles found.")
            except Exception as e:
                logger.error(f"Error in data collection: {e}")
            await asyncio.sleep(3600)  # Every hour

    async def event_detection_task(self):
        """Background task to detect events from articles every 10 minutes."""
        while True:
            try:
                with Session(engine) as session:
                    # Fetch recent articles
                    stmt = select(Article).order_by(Article.fetched_at.desc()).limit(10)
                    articles = session.exec(stmt).all()
                    if articles:
                        texts = [article.title + " " + article.link for article in articles]  # Simple text
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
                        logger.info(f"Detected and saved {len(events)} events.")
            except Exception as e:
                logger.error(f"Error in event detection: {e}")
            await asyncio.sleep(600)  # Every 10 minutes

    async def knowledge_compression_task(self):
        """Background task to compress knowledge every 24 hours."""
        while True:
            try:
                # Fetch recent logs (simplified; adjust to actual log source)
                logs = {"recent_interactions": ["Sample log entry"] * 10}  # Placeholder
                summary = compress_knowledge(logs)
                with Session(engine) as session:
                    summary_obj = Summary(
                        timestamp=summary["timestamp"],
                        summary_text=summary["summary"]
                    )
                    session.add(summary_obj)
                    session.commit()
                logger.info("Knowledge compressed and summary saved.")
            except Exception as e:
                logger.error(f"Error in knowledge compression: {e}")
            await asyncio.sleep(86400)  # Every 24 hours

    async def run_autonomous_loop(self):
        logger.info("Starting autonomous loop...")
        # Start background tasks
        asyncio.create_task(self.data_collection_task())
        asyncio.create_task(self.event_detection_task())
        asyncio.create_task(self.knowledge_compression_task())

        while True:
            try:
                # 1. Sense: Generate a new situation
                situation = self.situation_generator.generate_situation()
                self.shared_state['current_situation'] = situation
                logger.info(f"New Situation: {situation['type']} - {situation['prompt']}")

                # 2. Recall: Get relevant memories
                try:
                    relevant_memories = await get_relevant_memories_api({"query_text": situation['prompt']})
                    if relevant_memories:
                        self.shared_state['recent_memories'] = relevant_memories.relevant_memories
                    else:
                        self.shared_state['recent_memories'] = []
                except Exception as e:
                    logger.error(f"Could not retrieve memories: {e}")
                    self.shared_state['recent_memories'] = []

                # 3. Think: Decide on a plan
                decision_input = {
                    "situation": situation['prompt'],
                    "memory": [mem.text for mem in self.shared_state['recent_memories']],
                    "rag_context": situation.get('context'),
                }
                decision = await asyncio.to_thread(decision_maker_loop, **decision_input)
                logger.info(f"Decision: {decision}")
                
                # 4. Act: Execute the plan
                action_output = await self.execute_action(decision)
                logger.info(f"Action Output: {action_output}")

                # 5. Feel: Update emotional state
                self.emotional_intelligence.process_action_natural(action_output)
                self.shared_state['mood'] = self.emotional_intelligence.get_mood_vector()
                self.shared_state['mood_history'].append(self.emotional_intelligence.get_dominant_mood())
                if len(self.shared_state['mood_history']) > 50:
                    self.shared_state['mood_history'].pop(0)
                logger.info(f"Updated Mood: {self.shared_state['mood']}")

                # 6. Memorize: Store the interaction
                interaction_summary = f"Situation: {situation['prompt']}\nDecision: {decision}\nAction Output: {action_output}"
                memories_to_save = await extract_memories_api({"user_input": interaction_summary, "ai_output": ""})
                await asyncio.to_thread(save_memories, memories_to_save.memories)
                
                # 7. Reflect: Run self-reflection and experimentation
                dominant_mood = self.emotional_intelligence.get_dominant_mood()
                if dominant_mood == "Reflective":
                    logger.info("Agent is in a Reflective mood, starting self-modification and experimentation.")
                    # Generate a hypothesis
                    hypothesis = self.reflection_module.generate_hypothesis(self.shared_state)
                    if hypothesis:
                        logger.info(f"Generated Hypothesis: {hypothesis}")
                        # Design and run an experiment to test the hypothesis
                        experiment_results = await asyncio.to_thread(
                            self.experimentation_module.design_and_run_experiment,
                            hypothesis,
                            self.shared_state
                        )
                        logger.info(f"Experiment Results: {experiment_results}")
                        # The results could be used to inform future reflections or modifications
                    else:
                        logger.info("No new hypothesis generated.")

                # 8. Trigger Curiosity
                elif dominant_mood in ["Bored", "Confused"]:
                    logger.info(f"Agent is {dominant_mood}, triggering curiosity.")
                    curiosity_output = await asyncio.to_thread(CuriosityTrigger.trigger, self.shared_state.get('recent_memories', []))
                    logger.info(f"Curiosity Output: {curiosity_output}")

                await asyncio.sleep(10)  # Pause between loops
            except Exception as e:
                logger.error(f"Error in autonomous loop: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait a minute before retrying
                
    async def execute_action(self, decision):
        raw_response = decision.get('raw_response', "")
        if "experiment" in raw_response.lower():
            # This can be triggered by a decision, or a self-generated hypothesis
            return await asyncio.to_thread(self.experimentation_module.run_experiment_from_prompt, raw_response)
        elif "youtube.com" in raw_response.lower():
            url = raw_response.split(" ")[-1]
            return await asyncio.to_thread(transcribe_youtube_video, url)
        elif "trends" in raw_response.lower():
            return await asyncio.to_thread(analyze_trends)
        elif "curiosity" in raw_response.lower():
            # This can be triggered by a decision
            return await asyncio.to_thread(CuriosityTrigger.trigger, self.shared_state.get('recent_memories', []))
        else:
            return "No specific action taken from decision."

async def main():
    agi = AGISystem()
    await agi.run_autonomous_loop()

if __name__ == "__main__":
    asyncio.run(main())