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

from core.llm import decision_maker_loop
from core.shutdown_coordinator import ShutdownCoordinator, load_previous_state, cleanup_state_file
from modules.reflection_module import ReflectionModule
from modules.experimentation_module import ExperimentationModule
from modules.situation_generator.situation_generator import SituationGenerator
from modules.emotional_intellegence.emotional_intellegence import EmotionalIntelligence
from modules.curiosity_trigger.curiosity_trigger import CuriosityTrigger
from modules.agi_experimentation_engine import AGIExperimentationEngine
from core.config import Config
from modules.personality.personality import Personality
from services.data_service import DataService
from services.knowledge_service import KnowledgeService
from services.memory_service import MemoryService
from core.state import SharedState
from core.action_manager import ActionManager
from core.enhanced_action_manager import EnhancedActionManager
from modules.adaptive_learning.learning_engine import AdaptiveLearningEngine
from services.multi_modal_service import MultiModalService
from database.models import Event
from modules.decision_engine.search_result_manager import search_result_manager

# Import Snake Agent state for restoration (conditionally)
try:
    from core.snake_agent import SnakeAgentState
except ImportError:
    SnakeAgentState = None

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
        self.curiosity_trigger = CuriosityTrigger()  # Now enhanced with async capabilities
        self.reflection_module = ReflectionModule(self)
        self.experimentation_module = ExperimentationModule(self)
        self.experimentation_engine = AGIExperimentationEngine(self)

        # Initialize enhanced action manager
        self.action_manager = EnhancedActionManager(self, self.data_service)
        
        # Initialize adaptive learning engine
        self.learning_engine = AdaptiveLearningEngine(self)
        
        # Initialize multi-modal service
        self.multi_modal_service = MultiModalService()

        # Initialize personality
        self.personality = Personality(
            name=Config.PERSONA_NAME,
            origin=Config.PERSONA_ORIGIN,
            creativity=Config.PERSONA_CREATIVITY
        )
        
        # Initialize Snake Agent if enabled (Enhanced Version)
        self.snake_agent = None
        if Config.SNAKE_AGENT_ENABLED:
            try:
                # Try enhanced version first, fall back to original if needed
                enhanced_mode = getattr(Config, 'SNAKE_ENHANCED_MODE', True)
                if enhanced_mode:
                    from core.snake_agent_enhanced import EnhancedSnakeAgent
                    self.snake_agent = EnhancedSnakeAgent(self)
                    logger.info("Enhanced Snake Agent initialized and ready")
                else:
                    from core.snake_agent import SnakeAgent
                    self.snake_agent = SnakeAgent(self)
                    logger.info("Standard Snake Agent initialized and ready")
            except Exception as e:
                logger.error(f"Failed to initialize Snake Agent: {e}")
                # Fallback to standard version if enhanced fails
                try:
                    from core.snake_agent import SnakeAgent
                    self.snake_agent = SnakeAgent(self)
                    logger.info("Fallback to standard Snake Agent successful")
                except Exception as fallback_error:
                    logger.error(f"Fallback Snake Agent also failed: {fallback_error}")
                    self.snake_agent = None

        # New state for multi-step plans
        self.current_plan: List[Dict] = []
        self.current_task_prompt: str = None

        # For graceful shutdown
        self._shutdown = asyncio.Event()
        self.background_tasks = []
        self.shutdown_coordinator = ShutdownCoordinator(self)
        
        # Register cleanup handlers
        self.shutdown_coordinator.register_cleanup_handler(self._cleanup_database_session)
        self.shutdown_coordinator.register_cleanup_handler(self._cleanup_models)
        self.shutdown_coordinator.register_cleanup_handler(self._save_final_state, is_async=True)
        
        # Register Snake Agent cleanup if enabled
        if self.snake_agent:
            self.shutdown_coordinator.register_cleanup_handler(self._cleanup_snake_agent, is_async=True)

        # Shared state
        self.shared_state = SharedState(
            initial_mood=self.emotional_intelligence.get_mood_vector()
        )
        self.behavior_modifiers: Dict[str, Any] = {}
        self.last_interaction_time: datetime = None
        self.experiment_tracker: Dict[str, int] = {}
        self.research_in_progress: Dict[str, asyncio.Task] = {}
        self.research_results: Dict[str, Any] = {}
        # Invention tracking
        self.invention_history = []
        
        # Load previous state if available
        asyncio.create_task(self._load_previous_state())

    async def _check_for_search_results(self):
        """Enhanced search result processing with better error handling."""
        try:
            search_result = search_result_manager.get_result()
            if search_result:
                logger.info(f"Retrieved search result: {search_result[:100]}...")
                
                # Add to shared state for immediate use
                if not hasattr(self.shared_state, 'search_results'):
                    self.shared_state.search_results = []
                self.shared_state.search_results.append(search_result)
                
                # Limit search results to prevent memory bloat
                if len(self.shared_state.search_results) > 10:
                    self.shared_state.search_results = self.shared_state.search_results[-10:]
                
                # Add to memory for long-term retention
                try:
                    memory_summary = f"Search result retrieved: {search_result[:300]}..."
                    memories_to_save = await self.memory_service.extract_memories(memory_summary, "")
                    if memories_to_save and hasattr(memories_to_save, 'memories') and memories_to_save.memories:
                        await self.memory_service.save_memories(memories_to_save.memories)
                        logger.info(f"Saved {len(memories_to_save.memories)} memories from search result")
                except Exception as e:
                    logger.warning(f"Failed to save search result to memory: {e}")
                
                # Add to knowledge base for future reference (with enhanced error handling)
                try:
                    knowledge_result = await asyncio.to_thread(
                        self.knowledge_service.add_knowledge,
                        content=search_result,
                        source="web_search",
                        category="search_result"
                    )
                    if not knowledge_result.get('duplicate', False):
                        logger.info("Added new search result to knowledge base")
                    else:
                        logger.info("Search result already exists in knowledge base")
                except Exception as e:
                    logger.warning(f"Failed to add search result to knowledge base: {e}")
                    
        except Exception as e:
            logger.error(f"Error in search result processing: {e}", exc_info=True)

    async def stop(self, reason: str = "manual"):
        """Gracefully stops the AGI system using the shutdown coordinator."""
        if self._shutdown.is_set():
            logger.warning("Stop already called, ignoring duplicate request")
            return
            
        logger.info(f"AGI System stop requested - Reason: {reason}")
        await self.shutdown_coordinator.initiate_shutdown(reason)
    
    def _cleanup_database_session(self):
        """Clean up database session."""
        try:
            if hasattr(self, 'session') and self.session:
                self.session.close()
                logger.info("Database session closed")
        except Exception as e:
            logger.error(f"Error closing database session: {e}")
    
    def _cleanup_models(self):
        """Clean up loaded models and free memory."""
        try:
            # Clear model references to help with memory cleanup
            if hasattr(self, 'embedding_model'):
                del self.embedding_model
            if hasattr(self, 'sentiment_classifier'):
                del self.sentiment_classifier
            logger.info("Model references cleared")
        except Exception as e:
            logger.error(f"Error cleaning up models: {e}")
    
    async def _save_final_state(self):
        """Save final system state before shutdown."""
        try:
            if Config.STATE_PERSISTENCE_ENABLED:
                # This is handled by the shutdown coordinator
                # but we can add any AGI-specific state saving here
                logger.info("Final state saving handled by shutdown coordinator")
        except Exception as e:
            logger.error(f"Error saving final state: {e}")
    
    async def _cleanup_snake_agent(self):
        """Clean up Snake Agent resources."""
        try:
            if self.snake_agent:
                await self.snake_agent.stop()
                logger.info("Snake Agent stopped and cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up Snake Agent: {e}")
    
    async def start_snake_agent(self):
        """Start Snake Agent background operation."""
        if self.snake_agent and Config.SNAKE_AGENT_ENABLED:
            try:
                logger.info("Starting Snake Agent background operation...")
                snake_task = asyncio.create_task(self.snake_agent.start_autonomous_operation())
                self.background_tasks.append(snake_task)
                logger.info("Snake Agent started successfully")
            except Exception as e:
                logger.error(f"Failed to start Snake Agent: {e}")
    
    def get_snake_agent_status(self) -> Dict[str, Any]:
        """Get Snake Agent status information."""
        if not self.snake_agent:
            return {"enabled": False, "status": "not_initialized"}
        
        return {
            "enabled": Config.SNAKE_AGENT_ENABLED,
            "status": "active" if self.snake_agent.running else "inactive",
            **self.snake_agent.get_status()
        }
    
    async def _load_previous_state(self):
        """Load previous system state if available."""
        try:
            previous_state = await load_previous_state()
            if not previous_state:
                logger.info("No previous state found, starting fresh")
                return
            
            logger.info("Attempting to restore previous system state...")
            
            # Extract AGI system state
            agi_state = previous_state.get("agi_system", {})
            
            # Restore mood if available
            if "mood" in agi_state and hasattr(self, 'emotional_intelligence'):
                try:
                    self.emotional_intelligence.set_mood_vector(agi_state["mood"])
                    logger.info("Restored previous mood state")
                except Exception as e:
                    logger.warning(f"Could not restore mood state: {e}")
            
            # Restore current plans
            if "current_plan" in agi_state:
                self.current_plan = agi_state["current_plan"]
                self.current_task_prompt = agi_state.get("current_task_prompt")
                if self.current_plan:
                    logger.info(f"Restored plan with {len(self.current_plan)} remaining steps")
            
            # Restore shared state
            if "shared_state" in agi_state and hasattr(self, 'shared_state'):
                shared_data = agi_state["shared_state"]
                if "current_task" in shared_data:
                    self.shared_state.current_task = shared_data["current_task"]
                if "current_situation_id" in shared_data:
                    self.shared_state.current_situation_id = shared_data["current_situation_id"]
                logger.info("Restored shared state")
            
            # Restore invention history
            if "invention_history" in agi_state:
                self.invention_history = agi_state["invention_history"]
                logger.info(f"Restored {len(self.invention_history)} invention history entries")
            
            # Restore Snake Agent state
            if "snake_agent" in agi_state and self.snake_agent and SnakeAgentState:
                try:
                    snake_data = agi_state["snake_agent"]
                    if "state" in snake_data:
                        # Restore Snake Agent state
                        restored_state = SnakeAgentState.from_dict(snake_data["state"])
                        self.snake_agent.state = restored_state
                        
                        # Restore counters
                        self.snake_agent.analysis_count = snake_data.get("analysis_count", 0)
                        self.snake_agent.experiment_count = snake_data.get("experiment_count", 0)
                        self.snake_agent.communication_count = snake_data.get("communication_count", 0)
                        
                        logger.info(f"Restored Snake Agent state: {len(restored_state.pending_experiments)} pending experiments, {len(restored_state.communication_queue)} queued communications")
                except Exception as e:
                    logger.warning(f"Could not restore Snake Agent state: {e}")
            
            logger.info("✅ Previous system state restored successfully")
            
            # Clean up the state file after successful recovery
            cleanup_state_file()
            
        except Exception as e:
            logger.error(f"Error loading previous state: {e}")
            logger.info("Continuing with fresh initialization")

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
        """Enhanced curiosity handling with async operations and better context."""
        if random.random() < Config.CURIOSITY_CHANCE:
            logger.info("Curiosity triggered. Generating new topics...")
            
            try:
                # Extract recent topics from memories
                recent_topics = []
                if self.shared_state.recent_memories:
                    for memory in self.shared_state.recent_memories[:10]:  # Last 10 memories
                        if isinstance(memory, dict):
                            content = memory.get('content', '')
                        else:
                            content = str(memory)
                        
                        # Extract key topics from memory content
                        if content:
                            recent_topics.append(content[:100])  # Limit length
                
                if not recent_topics:
                    recent_topics = ["artificial intelligence", "machine learning", "consciousness", "creativity"]
                
                # Use enhanced async curiosity trigger
                curiosity_topics = await self.curiosity_trigger.get_curiosity_topics_llm(
                    recent_topics, 
                    n=5, 
                    lateralness=0.8  # High lateralness for creative exploration
                )
                
                self.shared_state.curiosity_topics = curiosity_topics
                logger.info(f"Generated {len(curiosity_topics)} curiosity topics: {curiosity_topics}")

                # Personality-influenced invention generation: sometimes create invention ideas
                try:
                    if random.random() < 0.4:  # 40% of curiosity events produce invention ideas
                        ideas = self.personality.invent_ideas(curiosity_topics, n=3)
                        self.shared_state.invention_ideas = ideas
                        # Persist a short log
                        for idea in ideas:
                            self.invention_history.append({"idea": idea, "ts": datetime.utcnow().isoformat()})
                        logger.info(f"Personality generated {len(ideas)} invention ideas.")
                except Exception as e:
                    logger.warning(f"Personality invention generation failed: {e}")
                
                # Occasionally trigger a full curiosity exploration
                if random.random() < 0.3:  # 30% chance
                    try:
                        content, prompt = await self.curiosity_trigger.trigger(recent_topics, lateralness=0.9)
                        if content and len(content) > 100:
                            # Add the curiosity content to knowledge base
                            await self.knowledge_service.add_knowledge(
                                content=content[:2000],  # Limit size
                                source="curiosity_trigger",
                                category="exploration"
                            )
                            logger.info("Added curiosity exploration to knowledge base")
                    except Exception as e:
                        logger.warning(f"Failed to process curiosity exploration: {e}")
                
            except Exception as e:
                logger.error(f"Curiosity handling failed: {e}", exc_info=True)
                self.shared_state.curiosity_topics = ["explore new possibilities", "question assumptions"]
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
        logger.info("Making enhanced decision with adaptive learning.")
        
        # Get available actions from the action manager's registry
        available_actions = self.action_manager.action_registry.get_action_definitions()
        
        # Apply adaptive learning to decision context
        decision_context = {
            'situation': situation,
            'mood': self.shared_state.mood,
            'memory': self.shared_state.recent_memories,
            'rag_context': getattr(self.shared_state, 'search_results', [])
        }
        
        learning_adaptations = await self.learning_engine.apply_learning_to_decision(decision_context)

        # Let personality influence decision context
        try:
            persona_mods = self.personality.influence_decision(decision_context)
            if persona_mods:
                decision_context['persona_mods'] = persona_mods
                logger.info(f"Applied persona modifiers to decision: {persona_mods}")
        except Exception as e:
            logger.debug(f"Personality influence failed: {e}")
        
        # Enhanced decision making with learning adaptations
        decision = await asyncio.to_thread(
            decision_maker_loop,
            situation=situation,
            memory=self.shared_state.recent_memories,
            mood=self.shared_state.mood,
            actions=available_actions,
            rag_context=getattr(self.shared_state, 'search_results', []),
            persona={
                'name': self.personality.name,
                'traits': self.personality.traits,
                'creativity': self.personality.creativity,
                'communication_style': self.personality.get_communication_style()
            }
        )
        
        # Apply learning adaptations to decision
        if learning_adaptations:
            # Adjust confidence based on learning
            confidence_modifier = learning_adaptations.get('confidence_modifier', 1.0)
            original_confidence = decision.get('confidence', 0.5)
            decision['confidence'] = min(1.0, max(0.0, original_confidence * confidence_modifier))
            
            # Add learning context
            decision['learning_adaptations'] = learning_adaptations
            decision['mood_context'] = self.shared_state.mood
            decision['memory_context'] = self.shared_state.recent_memories[:5]  # Last 5 memories
        
        # Log the enhanced decision
        await asyncio.to_thread(
            self.data_service.save_decision_log,
            self.shared_state.current_situation_id,
            decision['raw_response']
        )

        logger.info(f"Made enhanced decision with confidence {decision.get('confidence', 0.5):.2f}: {decision.get('action', 'unknown')}")
        return decision

    async def invention_task(self):
        """Background task where the personality occasionally picks an idea to pursue and records a lightweight outcome."""
        while not self._shutdown.is_set():
            try:
                # Only attempt occasionally
                await asyncio.sleep(Config.INVENTION_INTERVAL)
                ideas = getattr(self.shared_state, 'invention_ideas', None) or []
                if not ideas:
                    # generate seed ideas from recent memories or curiosity topics
                    topics = getattr(self.shared_state, 'curiosity_topics', []) or []
                    ideas = self.personality.invent_ideas(topics, n=2)

                if ideas:
                    chosen = self.personality.pick_idea_to_pursue(ideas)
                    # Simulate a lightweight experiment/outcome
                    outcome = {"success": random.random() < chosen.get('confidence', 0.5)}
                    self.personality.record_invention_outcome(chosen.get('id'), outcome)
                    self.invention_history.append({"idea": chosen, "outcome": outcome, "ts": datetime.utcnow().isoformat()})
                    logger.info(f"Personality pursued invention '{chosen.get('title')}' with outcome: {outcome}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in invention task: {e}", exc_info=True)
        logger.info("Invention task shut down.")

    async def _execute_and_memorize(self, situation_prompt: str, decision: dict):
        logger.info("Executing enhanced action.")
        
        # Use enhanced action execution
        action_output = await self.action_manager.execute_action_enhanced(decision)
        logger.info(f"Enhanced action output: {action_output}")
        
        # Record decision outcome for learning
        success = not (isinstance(action_output, dict) and action_output.get('error'))
        await self.learning_engine.record_decision_outcome(decision, action_output, success)

        logger.info("Memorizing interaction.")
        await self._memorize_interaction(situation_prompt, decision, action_output)
        logger.info("Memorized interaction.")
        
        # Clear action cache periodically
        self.action_manager.clear_cache()
        
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
        
        # Start Snake Agent if enabled
        if Config.SNAKE_AGENT_ENABLED and self.snake_agent:
            await self.start_snake_agent()

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