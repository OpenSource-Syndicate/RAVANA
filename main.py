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
    async def generate_situation(self) -> Dict[str, str]:
        """Enhanced situation generator that can use recent events, summaries, or memories."""
        roll = random.random()

        if roll < 0.25:  # 25% chance to use a recent event
            with Session(engine) as session:
                # Fetch a few recent events and pick one
                stmt = select(Event).order_by(Event.timestamp.desc()).limit(5)
                events = session.exec(stmt).all()
                if events:
                    event = random.choice(events)
                    # Generate a more engaging prompt
                    prompt = f"A recent event was reported: '{event.description}'. What are the potential short-term and long-term consequences of this?"
                    return {'type': 'event_based', 'prompt': prompt, 'context': event.description}
        
        elif roll < 0.5: # 25% chance to reflect on a recent memory
            try:
                # Query for a "general" topic to get some recent memories.
                recent_memories_response = await get_relevant_memories_api({"query_text": "recent interesting topics"})
                if recent_memories_response and recent_memories_response.relevant_memories:
                    memory = random.choice(recent_memories_response.relevant_memories)
                    prompt = f"Let's reflect on this memory: '{memory.text}'. How does this connect to my long-term goals or understanding of the world?"
                    return {'type': 'memory_reflection', 'prompt': prompt, 'context': memory.text}
            except Exception as e:
                logger.warning(f"Could not fetch memories for situation generation: {e}")
                # Fallback to default if memory retrieval fails
        
        # Fallback to default situation generation (50% chance)
        return {'type': 'default', 'prompt': 'What is a surprising fact I learned recently, and what makes it surprising?'}

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

        # For graceful shutdown
        self._shutdown = asyncio.Event()
        self.background_tasks = []

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

    async def stop(self):
        """Gracefully stops the AGI system and its background tasks."""
        if self._shutdown.is_set():
            return
            
        logger.info("Initiating graceful shutdown...")
        self._shutdown.set()

        logger.info(f"Cancelling {len(self.background_tasks)} background tasks...")
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for all background tasks to be cancelled
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        logger.info("All background tasks stopped.")

    async def _memorize_interaction(self, situation_prompt: str, decision: dict, action_output: Any):
        """Extracts and saves memories from an interaction."""
        interaction_summary = f"Situation: {situation_prompt}\nDecision: {decision}\nAction Output: {action_output}"
        try:
            memories_to_save = await extract_memories_api({"user_input": interaction_summary, "ai_output": ""})
            if memories_to_save and memories_to_save.memories:
                await asyncio.to_thread(save_memories, memories_to_save.memories)
                logger.info(f"Saved {len(memories_to_save.memories)} new memories.")
        except Exception as e:
            logger.error(f"Failed during memorization: {e}", exc_info=True)

    async def data_collection_task(self):
        """Background task to fetch articles from RSS feeds every hour."""
        while not self._shutdown.is_set():
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
            
            try:
                # Wait for an hour, but be responsive to shutdown
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                # Task was cancelled, exit the loop
                break
        logger.info("Data collection task shut down.")

    async def event_detection_task(self):
        """Background task to detect events from articles every 10 minutes."""
        while not self._shutdown.is_set():
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
            
            try:
                await asyncio.sleep(600)  # Every 10 minutes
            except asyncio.CancelledError:
                break
        logger.info("Event detection task shut down.")

    async def knowledge_compression_task(self):
        """Background task to compress knowledge every 24 hours."""
        while not self._shutdown.is_set():
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
            
            try:
                await asyncio.sleep(86400)  # Every 24 hours
            except asyncio.CancelledError:
                break
        logger.info("Knowledge compression task shut down.")

    async def run_autonomous_loop(self):
        logger.info("Starting autonomous loop...")
        # Start background tasks
        self.background_tasks.append(asyncio.create_task(self.data_collection_task()))
        self.background_tasks.append(asyncio.create_task(self.event_detection_task()))
        self.background_tasks.append(asyncio.create_task(self.knowledge_compression_task()))

        while not self._shutdown.is_set():
            try:
                # 1. Sense: Generate a new situation
                situation = await self.situation_generator.generate_situation()
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

                # 5. & 6. Feel and Memorize (concurrently)
                mood_update_task = asyncio.to_thread(self.emotional_intelligence.process_action_natural, action_output)
                memorization_task = self._memorize_interaction(situation['prompt'], decision, action_output)
                
                await asyncio.gather(mood_update_task, memorization_task)

                # Update state after concurrent tasks
                self.shared_state['mood'] = self.emotional_intelligence.get_mood_vector()
                self.shared_state['mood_history'].append(self.emotional_intelligence.get_dominant_mood())
                if len(self.shared_state['mood_history']) > 50:
                    self.shared_state['mood_history'].pop(0)
                logger.info(f"Updated Mood: {self.shared_state['mood']}")
                
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

                try:
                    await asyncio.sleep(10)  # Pause between loops
                except asyncio.CancelledError:
                    break # Exit loop if main task is cancelled
            except Exception as e:
                logger.error(f"Error in autonomous loop: {e}", exc_info=True)
                if self._shutdown.is_set():
                    break
                await asyncio.sleep(60)  # Wait a minute before retrying
        logger.info("Autonomous loop finished.")
                
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
    main_task = None
    try:
        main_task = asyncio.create_task(agi.run_autonomous_loop())
        await main_task
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutdown signal received.")
    finally:
        if main_task and not main_task.done():
            main_task.cancel()
        await agi.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application shut down by user.")