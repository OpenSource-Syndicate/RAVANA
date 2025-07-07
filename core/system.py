import asyncio
import logging
import random
import hashlib
import json
from typing import Any, Dict, List
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sqlmodel import Session, select
from datetime import datetime, timedelta

from modules.decision_engine.llm import decision_maker_loop
from modules.reflection_module import ReflectionModule
from modules.experimentation_module import ExperimentationModule
from modules.situation_generator.situation_generator import SituationGenerator
from modules.emotional_intellegence.emotional_intellegence import EmotionalIntelligence
from modules.curiosity_trigger.curiosity_trigger import CuriosityTrigger
from modules.agi_experimentation_engine import AGIExperimentationEngine
from core.config import Config
from services.data_service import DataService
from services.knowledge_service import KnowledgeService
from services.memory_service import MemoryService
from core.state import SharedState
from core.action_manager import ActionManager
from database.models import Event
from modules.decision_engine.search_result_manager import search_result_manager

logger = logging.getLogger(__name__)


class AGISystem:
    def __init__(self, engine):
        logger.info("Initializing Ravana AGI System...")
        
        self.engine = engine
        self.session = Session(engine)
        self.config = Config()
        # Load shared models
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.sentiment_classifier = pipeline('sentiment-analysis')

        # Initialize services
        self.data_service = DataService(
            engine,
            Config.FEED_URLS,
            self.embedding_model,
            self.sentiment_classifier
        )
        self.knowledge_service = KnowledgeService(engine)
        self.memory_service = MemoryService()

        # Initialize modules
        self.situation_generator = SituationGenerator(
            embedding_model=self.embedding_model,
            sentiment_classifier=self.sentiment_classifier
        )
        self.emotional_intelligence = EmotionalIntelligence()
        self.curiosity_trigger = CuriosityTrigger()
        self.reflection_module = ReflectionModule(self)
        self.experimentation_module = ExperimentationModule(self)
        self.experimentation_engine = AGIExperimentationEngine(self)

        # Initialize action manager
        self.action_manager = ActionManager(self, self.data_service)

        # New state for multi-step plans
        self.current_plan: List[Dict] = []
        self.current_task_prompt: str = None

        # For graceful shutdown
        self._shutdown = asyncio.Event()
        self.background_tasks = []

        # Shared state
        self.shared_state = SharedState(
            initial_mood=self.emotional_intelligence.get_mood_vector()
        )
        self.behavior_modifiers: Dict[str, Any] = {}
        self.last_interaction_time: datetime = None
        self.experiment_tracker: Dict[str, int] = {}
        self.research_in_progress: Dict[str, asyncio.Task] = {}
        self.research_results: Dict[str, Any] = {}

    async def _check_for_search_results(self):
        """Checks for and processes any completed search results."""
        search_result = search_result_manager.get_result()
        if search_result:
            logger.info(f"Retrieved search result: {search_result}")
            
            # Add to shared state for immediate use
            if 'search_results' not in self.shared_state:
                self.shared_state.search_results = []
            self.shared_state.search_results.append(search_result)
            
            # Add to memory for long-term retention
            try:
                memory_summary = f"Search result retrieved: {search_result[:200]}..."  # Truncate for memory
                memories_to_save = await self.memory_service.extract_memories(memory_summary, "")
                if memories_to_save and memories_to_save.memories:
                    await self.memory_service.save_memories(memories_to_save.memories)
                    logger.info(f"Saved {len(memories_to_save.memories)} memories from search result")
            except Exception as e:
                logger.error(f"Failed to save search result to memory: {e}", exc_info=True)
            
            # Add to knowledge base for future reference
            try:
                await self.knowledge_service.add_knowledge(
                    content=search_result,
                    source="web_search",
                    category="search_result"
                )
                logger.info("Added search result to knowledge base")
            except Exception as e:
                logger.error(f"Failed to add search result to knowledge base: {e}", exc_info=True)

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
        self.session.close()

    async def _memorize_interaction(self, situation_prompt: str, decision: dict, action_output: Any):
        """Extracts and saves memories from an interaction."""
        interaction_summary = f"Situation: {situation_prompt}\nDecision: {decision}\nAction Output: {action_output}"
        try:
            memories_to_save = await self.memory_service.extract_memories(interaction_summary, "")
            if memories_to_save and memories_to_save.memories:
                await self.memory_service.save_memories(memories_to_save.memories)
                logger.info(f"Saved {len(memories_to_save.memories)} new memories.")
                self.last_interaction_time = datetime.utcnow()
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
            if self.shared_state.recent_memories:
                # Use a more descriptive name for the context
                context_for_curiosity = ". ".join([m['content'] for m in self.shared_state.recent_memories])
            else:
                context_for_curiosity = "Artificial intelligence, machine learning, and consciousness."
            
            curiosity_topics = await asyncio.to_thread(
                self.curiosity_trigger.get_curiosity_topics_llm, 
                [context_for_curiosity]
            )
            self.shared_state.curiosity_topics = curiosity_topics
            logger.info(f"Generated curiosity topics: {curiosity_topics}")
        else:
            self.shared_state.curiosity_topics = []

    async def _generate_situation(self):
        situation = await self.situation_generator.generate_situation(
            shared_state=self.shared_state,
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
            actions=available_actions,
            rag_context=self.shared_state.search_results
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

        # If the action output contains a directive (like initiating an experiment),
        # merge it into the behavior modifiers for the next loop.
        if isinstance(action_output, dict) and 'action' in action_output:
            if action_output['action'] == 'initiate_experiment':
                logger.info(f"Action output contains a directive to '{action_output['action']}'. Starting experiment.")
                self.experimentation_engine.start_experiment(action_output)

        mood_changed_for_better = self._did_mood_improve(old_mood, new_mood)
        
        if not mood_changed_for_better and random.random() < Config.REFLECTION_CHANCE:
            logger.info("Mood has not improved. Initiating reflection.")
            # This is where you can trigger a reflection process
            # For now, we'll just log it.
            self.reflection_module.reflect(self.shared_state)
        else:
            logger.info("Mood improved or stayed the same, skipping reflection.")

    async def get_recent_events(self, time_limit_seconds: int = 3600) -> List[Event]:
        """
        Retrieves recent events from the database.
        """
        time_limit = datetime.utcnow() - timedelta(seconds=time_limit_seconds)
        stmt = select(Event).where(Event.timestamp >= time_limit).order_by(Event.timestamp.desc())
        
        loop = asyncio.get_running_loop()
        try:
            # Use a thread pool executor for the synchronous DB call
            result = await loop.run_in_executor(
                None,  # Uses the default executor
                lambda: self.session.exec(stmt).all()
            )
            return result
        except Exception as e:
            logger.error(f"Database query for recent events failed: {e}", exc_info=True)
            return []

    async def run_iteration(self):
        """Runs a single iteration of the AGI's thought process."""
        # 1. Check for external data and completed tasks
        await self._check_for_search_results()

        # 2. Handle any mood-based behavior modifiers from the previous loop
        await self._handle_behavior_modifiers()
        
        # 3. Handle Curiosity
        await self._handle_curiosity()

        # 4. Decide on the next action
        if self.current_plan:
            # Continue with the existing plan
            decision = self.current_plan.pop(0)
            logger.info(f"Continuing with task: '{self.current_task_prompt}'. {len(self.current_plan)} steps remaining.")
            situation_prompt = self.current_task_prompt
        elif self.shared_state.current_task:
            situation_prompt = self.shared_state.current_task
            await self._retrieve_memories(situation_prompt)
            situation = {
                'prompt': situation_prompt,
                'context': self.shared_state.recent_memories
            }
            decision = await self._make_decision(situation)
            self.current_task_prompt = situation_prompt
        else:
            # Autonomous mode: no specific task, generate a situation
            situation = await self._generate_situation()
            situation_prompt = situation['prompt']
            await self._retrieve_memories(situation_prompt)
            decision = await self._make_decision(situation)

        # Execute action and update state
        action_output = await self._execute_and_memorize(situation_prompt, decision)

        # Check if the decision included a plan
        raw_response = decision.get("raw_response", "{}")
        try:
            # Find the JSON block in the raw response
            json_start = raw_response.find("```json")
            json_end = raw_response.rfind("```")
            if json_start != -1 and json_end != -1 and json_start < json_end:
                json_str = raw_response[json_start + 7:json_end].strip()
                decision_data = json.loads(json_str)
            else:
                # Try parsing the whole string if no block is found
                decision_data = json.loads(raw_response)
                
            plan = decision_data.get("plan")
            if plan and isinstance(plan, list) and len(plan) > 1:
                # The first step was already chosen as the main action, so store the rest
                self.current_plan = plan[1:]
                self.current_task_prompt = situation_prompt
                logger.info(f"Found and stored a multi-step plan with {len(self.current_plan)} steps remaining.")
            else:
                # If the plan is done or was a single step, clear it.
                self.current_plan = []
                self.current_task_prompt = None
                if plan:
                    logger.info("Plan found, but only had one step which was already executed.")

        except json.JSONDecodeError:
            logger.warning("Could not parse plan from decision response.")
            self.current_plan = []
            self.current_task_prompt = None


        # Update mood and reflect
        await self._update_mood_and_reflect(action_output)

    async def run_autonomous_loop(self):
        """The main autonomous loop of the AGI."""
        logger.info("Starting autonomous loop...")
        
        # Start background tasks
        self.background_tasks.append(asyncio.create_task(self.data_collection_task()))
        self.background_tasks.append(asyncio.create_task(self.event_detection_task()))
        self.background_tasks.append(asyncio.create_task(self.knowledge_compression_task()))
        self.background_tasks.append(asyncio.create_task(self.memory_consolidation_task()))

        while not self._shutdown.is_set():
            try:
                if self.experimentation_engine.active_experiment:
                    await self.experimentation_engine.run_experiment_step()
                else:
                    await self.run_iteration()
                
                logger.info(f"End of loop iteration. Sleeping for {Config.LOOP_SLEEP_DURATION} seconds.")
                await asyncio.sleep(Config.LOOP_SLEEP_DURATION)
            except Exception as e:
                logger.critical(f"Critical error in autonomous loop: {e}", exc_info=True)
                await asyncio.sleep(Config.LOOP_SLEEP_DURATION * 5) # Longer sleep after critical error
        
        logger.info("Autonomous loop has been stopped.")

    async def run_single_task(self, prompt: str):
        """Runs the AGI for a single task specified by the prompt."""
        logger.info(f"--- Running Single Task: {prompt} ---")
        self.shared_state.current_task = prompt
        
        max_iterations = Config.MAX_ITERATIONS
        for i in range(max_iterations):
            if self._shutdown.is_set():
                logger.info("Task appears to be complete. Ending run.")
                break
            await self.run_iteration()

            if not self.current_plan and not self.current_task_prompt:
                logger.info("Task appears to be complete. Ending run.")
                break
            
            await asyncio.sleep(1) # Give a moment for async operations
        else:
            logger.warning(f"Task exceeded {max_iterations} iterations. Ending run.")

        logger.info("--- Single Task Finished ---")

    def _did_mood_improve(self, old_mood: Dict[str, float], new_mood: Dict[str, float]) -> bool:
        """
        Checks if the overall mood has improved based on positive and negative mood components.
        """
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