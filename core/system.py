import asyncio
import logging
import random
from typing import Any, Dict, List
from sentence_transformers import SentenceTransformer
from sqlmodel import Session, select

from modules.decision_engine.llm import decision_maker_loop, generate_hypothetical_scenarios, agi_experimentation_engine
from modules.information_processing.youtube_transcription.youtube_transcription import transcribe_youtube_video
from modules.information_processing.trend_analysis.trend_engine import analyze_trends
from modules.agent_self_reflection.self_modification import (
    generate_hypothesis,
    design_and_run_experiment,
    run_experiment_from_prompt
)
from modules.situation_generator.situation_generator import SituationGenerator
from modules.emotional_intellegence.emotional_intellegence import EmotionalIntelligence
from modules.curiosity_trigger.curiosity_trigger import CuriosityTrigger
from datetime import datetime
from core.config import Config
from services.data_service import DataService
from services.knowledge_service import KnowledgeService
from services.memory_service import MemoryService

logger = logging.getLogger(__name__)


class AGISystem:
    def __init__(self, engine):
        logger.info("Initializing Ravana AGI System...")
        
        self.engine = engine
        # Load shared models
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize services
        self.data_service = DataService(engine, Config.FEED_URLS)
        self.knowledge_service = KnowledgeService(engine)
        self.memory_service = MemoryService()

        # Initialize modules
        self.situation_generator = SituationGenerator()
        self.emotional_intelligence = EmotionalIntelligence()
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

    async def stop(self):
        """Gracefully stops the AGI system and its background tasks."""
        if self._shutdown.is_set():
            return
            
        logger.info("Initiating graceful shutdown...")
        self._shutdown.set()

        logger.info(f"Cancelling {len(self.background_tasks)} background tasks...")
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        logger.info("All background tasks stopped.")

    async def _memorize_interaction(self, situation_prompt: str, decision: dict, action_output: Any):
        """Extracts and saves memories from an interaction."""
        interaction_summary = f"Situation: {situation_prompt}\nDecision: {decision}\nAction Output: {action_output}"
        try:
            memories_to_save = await self.memory_service.extract_memories(interaction_summary, "")
            if memories_to_save and memories_to_save.memories:
                await self.memory_service.save_memories(memories_to_save.memories)
                logger.info(f"Saved {len(memories_to_save.memories)} new memories.")
        except Exception as e:
            logger.error(f"Failed during memorization: {e}", exc_info=True)

    async def run_autonomous_loop(self):
        """The main autonomous loop of the AGI system."""
        logger.info("Starting autonomous loop...")

        self.background_tasks.append(asyncio.create_task(self.data_collection_task()))
        self.background_tasks.append(asyncio.create_task(self.event_detection_task()))
        self.background_tasks.append(asyncio.create_task(self.knowledge_compression_task()))
        self.background_tasks.append(asyncio.create_task(self.memory_consolidation_task()))

        while not self._shutdown.is_set():
            try:
                logger.info("New loop iteration.")
                situation = await self.situation_generator.generate_situation()
                self.shared_state["current_situation"] = situation
                logger.info(f"Generated situation: {situation}")

                try:
                    logger.info("Getting relevant memories.")
                    memory_response = await self.memory_service.get_relevant_memories(situation['prompt'])
                    self.shared_state["recent_memories"] = memory_response.relevant_memories
                    logger.info("Got relevant memories.")
                except Exception as e:
                    logger.warning(f"Could not retrieve memories: {e}")
                    self.shared_state["recent_memories"] = []

                logger.info("Making a decision.")
                decision = await asyncio.to_thread(
                    decision_maker_loop,
                    situation=situation,
                    memory=self.shared_state["recent_memories"]
                )
                logger.info(f"Made decision: {decision}")

                logger.info("Executing action.")
                action_output = await self.execute_action(decision)
                logger.info(f"Action output: {action_output}")

                logger.info("Memorizing interaction.")
                await self._memorize_interaction(situation['prompt'], decision, action_output)
                logger.info("Memorized interaction.")

                logger.info("Updating mood.")
                self.emotional_intelligence.process_action_natural(str(action_output))
                self.shared_state["mood"] = self.emotional_intelligence.get_mood_vector()
                self.shared_state["mood_history"].append(self.shared_state["mood"])
                logger.info("Updated mood.")

                if random.random() < 0.1:
                    logger.info("Starting self-reflection cycle...")
                    hypothesis = self.reflection_module.generate_hypothesis(self.shared_state)
                    experiment_results = await asyncio.to_thread(
                        self.experimentation_module.run_experiment_from_prompt,
                        hypothesis
                    )
                    logger.info(f"Reflection experiment results: {experiment_results}")

                logger.info("End of loop iteration. Sleeping for 10 seconds.")
                await asyncio.sleep(10)

            except asyncio.CancelledError:
                logger.info("Autonomous loop cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in autonomous loop: {e}", exc_info=True)
                await asyncio.sleep(60)

    async def execute_action(self, decision: Dict[str, Any]) -> Any:
        """Executes the action determined by the decision engine."""
        action_type = decision.get("action")
        params = decision.get("params", {})

        if action_type == "fetch_and_analyze_trends":
            return await asyncio.to_thread(analyze_trends, Config.FEED_URLS)
        elif action_type == "transcribe_youtube":
            return await asyncio.to_thread(transcribe_youtube_video, params.get("url"))
        elif action_type == "run_hypothetical_scenario":
            return await generate_hypothetical_scenarios(params.get("scenario_description"))
        elif action_type == "compress_knowledge":
            return await asyncio.to_thread(self.knowledge_service.compress_and_save_knowledge)
        else:
            logger.warning(f"Unknown action type: {action_type}")
            return "No action taken."

    async def data_collection_task(self):
        """Background task to fetch articles from RSS feeds every hour."""
        while not self._shutdown.is_set():
            try:
                logger.info("Fetching feeds...")
                num_saved = await asyncio.to_thread(self.data_service.fetch_and_save_articles)
                if num_saved > 0:
                    logger.info(f"Feeds fetched and {num_saved} new articles saved.")
                else:
                    logger.info("No new articles found.")
            except Exception as e:
                logger.error(f"Error in data collection: {e}")
            
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                break
        logger.info("Data collection task shut down.")

    async def event_detection_task(self):
        """Background task to detect events from articles every 10 minutes."""
        while not self._shutdown.is_set():
            try:
                num_events = await asyncio.to_thread(self.data_service.detect_and_save_events)
                if num_events > 0:
                    logger.info(f"Detected and saved {num_events} events.")
            except Exception as e:
                logger.error(f"Error in event detection: {e}")
            
            try:
                await asyncio.sleep(600)
            except asyncio.CancelledError:
                break
        logger.info("Event detection task shut down.")

    async def knowledge_compression_task(self):
        """Background task to compress knowledge every 24 hours."""
        while not self._shutdown.is_set():
            try:
                summary = await asyncio.to_thread(self.knowledge_service.compress_and_save_knowledge)
                logger.info(f"Compressed and saved knowledge summary.")
            except Exception as e:
                logger.error(f"Error in knowledge compression: {e}")

            try:
                await asyncio.sleep(86400)
            except asyncio.CancelledError:
                break
        logger.info("Knowledge compression task shut down.")

    async def memory_consolidation_task(self):
        """Periodically consolidates memories to optimize retrieval and relevance."""
        while not self._shutdown.is_set():
            try:
                logger.info("Starting memory consolidation...")
                consolidation_result = await self.memory_service.consolidate_memories()
                logger.info(f"Memory consolidation finished. Report: {consolidation_result}")
            except Exception as e:
                logger.error(f"Error during memory consolidation: {e}", exc_info=True)

            try:
                await asyncio.sleep(21600)
            except asyncio.CancelledError:
                break
        logger.info("Memory consolidation task shut down.") 