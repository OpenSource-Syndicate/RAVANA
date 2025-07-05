import asyncio
import logging
import random
from typing import Any, Dict, List
from sentence_transformers import SentenceTransformer
from sqlmodel import Session, select

from modules.decision_engine.llm import decision_maker_loop
from modules.reflection_module import ReflectionModule
from modules.experimentation_module import ExperimentationModule
from modules.situation_generator.situation_generator import SituationGenerator
from modules.emotional_intellegence.emotional_intellegence import EmotionalIntelligence
from modules.curiosity_trigger.curiosity_trigger import CuriosityTrigger
from datetime import datetime
from core.config import Config
from services.data_service import DataService
from services.knowledge_service import KnowledgeService
from services.memory_service import MemoryService
from core.state import SharedState
from core.action_manager import ActionManager

logger = logging.getLogger(__name__)


class AGISystem:
    def __init__(self, engine):
        logger.info("Initializing Ravana AGI System...")
        
        self.engine = engine
        # Load shared models
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        
        # Initialize services
        self.data_service = DataService(engine, Config.FEED_URLS)
        self.knowledge_service = KnowledgeService(engine)
        self.memory_service = MemoryService()

        # Initialize modules
        self.situation_generator = SituationGenerator()
        self.emotional_intelligence = EmotionalIntelligence()
        self.curiosity_trigger = CuriosityTrigger()
        self.reflection_module = ReflectionModule()
        self.experimentation_module = ExperimentationModule()

        # Initialize action manager
        self.action_manager = ActionManager(self, self.data_service)

        # For graceful shutdown
        self._shutdown = asyncio.Event()
        self.background_tasks = []

        # Shared state
        self.shared_state = SharedState(
            initial_mood=self.emotional_intelligence.get_mood_vector()
        )
        self.behavior_modifiers: Dict[str, Any] = {}

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

    async def _handle_behavior_modifiers(self):
        if self.behavior_modifiers.get('suggest_break'):
            logger.info("Mood suggests taking a break. Sleeping for a short while.")
            await asyncio.sleep(Config.LOOP_SLEEP_DURATION * 2)
            self.behavior_modifiers = {}  # Reset modifiers

    async def _handle_curiosity(self):
        if random.random() < Config.CURIOSITY_CHANCE:
            logger.info("Curiosity triggered. Generating new topics...")
            context_for_curiosity = ". ".join([m['content'] for m in self.shared_state.recent_memories])
            if not context_for_curiosity:
                context_for_curiosity = "Artificial intelligence, machine learning, and consciousness."
            
            curiosity_topics = await asyncio.to_thread(self.curiosity_trigger.get_curiosity_topics_llm, [context_for_curiosity])
            self.shared_state.curiosity_topics = curiosity_topics
            logger.info(f"Generated curiosity topics: {curiosity_topics}")
        else:
            self.shared_state.curiosity_topics = []

    async def _generate_situation(self):
        situation = await self.situation_generator.generate_situation(
            curiosity_topics=self.shared_state.curiosity_topics,
            behavior_modifiers=self.behavior_modifiers
        )
        self.shared_state.current_situation = situation
        
        # Log the generated situation
        situation_id = await asyncio.to_thread(self.data_service.save_situation_log, situation)
        self.shared_state.current_situation_id = situation_id
        
        logger.info(f"Generated situation: {situation}")
        return situation

    async def _retrieve_memories(self, situation_prompt: str):
        try:
            logger.info("Getting relevant memories.")
            memory_response = await self.memory_service.get_relevant_memories(situation_prompt)
            self.shared_state.recent_memories = memory_response.relevant_memories
            logger.info("Got relevant memories.")
        except Exception as e:
            logger.warning(f"Could not retrieve memories: {e}")
            self.shared_state.recent_memories = []

    async def _make_decision(self, situation: dict):
        logger.info("Making a decision.")
        
        # Get available actions from the action manager's registry
        available_actions = self.action_manager.action_registry.get_action_definitions()
        
        decision = await asyncio.to_thread(
            decision_maker_loop,
            situation=situation,
            memory=self.shared_state.recent_memories,
            mood=self.shared_state.mood,
            actions=available_actions
        )
        
        # Log the decision
        await asyncio.to_thread(
            self.data_service.save_decision_log,
            self.shared_state.current_situation_id,
            decision['raw_response']
        )

        logger.info(f"Made decision: {decision}")
        return decision

    async def _execute_and_memorize(self, situation_prompt: str, decision: dict):
        logger.info("Executing action.")
        action_output = await self.action_manager.execute_action(decision)
        logger.info(f"Action output: {action_output}")

        logger.info("Memorizing interaction.")
        await self._memorize_interaction(situation_prompt, decision, action_output)
        logger.info("Memorized interaction.")
        return action_output

    async def _update_mood_and_reflect(self, action_output: Any):
        logger.info("Updating mood.")
        old_mood = self.shared_state.mood.copy()
        self.emotional_intelligence.process_action_natural(str(action_output))
        self.shared_state.mood = self.emotional_intelligence.get_mood_vector()
        new_mood = self.shared_state.mood
        self.shared_state.mood_history.append(self.shared_state.mood)

        # Log the mood
        await asyncio.to_thread(self.data_service.save_mood_log, new_mood)
        
        logger.info("Updated mood.")

        self.behavior_modifiers = self.emotional_intelligence.influence_behavior()
        if self.behavior_modifiers:
            logger.info(f"Generated behavior modifiers for next loop: {self.behavior_modifiers}")

        mood_changed_for_better = self._did_mood_improve(old_mood, new_mood)
        
        if not mood_changed_for_better:
            logger.info("Mood did not improve. Starting self-reflection cycle...")
            hypothesis = self.reflection_module.generate_hypothesis(self.shared_state)
            if hypothesis:
                experiment_results = await asyncio.to_thread(
                    self.experimentation_module.run_experiment_from_prompt,
                    hypothesis
                )
                # Log the experiment
                await asyncio.to_thread(
                    self.data_service.save_experiment_log,
                    hypothesis,
                    experiment_results
                )
                logger.info(f"Reflection experiment results: {experiment_results}")
        else:
            logger.info("Mood improved or stayed the same, skipping reflection.")

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

                await self._handle_behavior_modifiers()
                await self._handle_curiosity()
                
                situation = await self._generate_situation()
                await self._retrieve_memories(situation['prompt'])
                decision = await self._make_decision(situation)
                action_output = await self._execute_and_memorize(situation['prompt'], decision)
                await self._update_mood_and_reflect(action_output)

                logger.info(f"End of loop iteration. Sleeping for {Config.LOOP_SLEEP_DURATION} seconds.")
                await asyncio.sleep(Config.LOOP_SLEEP_DURATION)

            except asyncio.CancelledError:
                logger.info("Autonomous loop cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in autonomous loop: {e}", exc_info=True)
                await asyncio.sleep(Config.ERROR_SLEEP_DURATION)

    def _did_mood_improve(self, old_mood: Dict[str, float], new_mood: Dict[str, float]) -> bool:
        """Checks if the mood has improved based on a simple score."""
        old_score = sum(old_mood.get(m, 0) for m in Config.POSITIVE_MOODS) - sum(old_mood.get(m, 0) for m in Config.NEGATIVE_MOODS)
        new_score = sum(new_mood.get(m, 0) for m in Config.POSITIVE_MOODS) - sum(new_mood.get(m, 0) for m in Config.NEGATIVE_MOODS)
        
        logger.info(f"Mood score changed from {old_score:.2f} to {new_score:.2f}")
        return new_score > old_score

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
                await asyncio.sleep(Config.DATA_COLLECTION_INTERVAL) # Use config value
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
                await asyncio.sleep(Config.EVENT_DETECTION_INTERVAL) # Use config value
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