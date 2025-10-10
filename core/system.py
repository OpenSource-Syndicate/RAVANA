import asyncio
import logging
import random
import json
import time
from typing import Any, Dict, List
from transformers import pipeline
from core.embeddings_manager import embeddings_manager, ModelPurpose
from core.llm_selector import initialize_llm_selector
from core.enhanced_memory_service import enhanced_memory_service, MemoryType
from sqlmodel import Session, select
from datetime import datetime, timedelta, timezone

from core.llm import decision_maker_loop
from core.shutdown_coordinator import ShutdownCoordinator, ShutdownPriority, load_previous_state, cleanup_state_file
from modules.reflection_module import ReflectionModule
from modules.situation_generator.situation_generator import SituationGenerator
from modules.emotional_intellegence.emotional_intellegence import EmotionalIntelligence
from modules.curiosity_trigger.curiosity_trigger import CuriosityTrigger
from modules.agi_experimentation_engine import AGIExperimentationEngine
from core.config import Config
from modules.personality.personality import Personality
from services.data_service import DataService
from services.knowledge_service import KnowledgeService
from core.enhanced_memory_service import EnhancedMemoryService
from core.state import SharedState
from core.action_manager import ActionManager
from core.action_manager import ActionManager
from modules.adaptive_learning.learning_engine import AdaptiveLearningEngine
from services.multi_modal_service import MultiModalService
from database.models import Event
from modules.decision_engine.search_result_manager import search_result_manager
from modules.performance_monitoring.performance_tracker import performance_tracker
from modules.self_improvement.self_goal_manager import self_goal_manager, GoalStatus, GoalPriority
from modules.failure_learning_system import FailureLearningSystem
from modules.physics_analysis_system import PhysicsAnalysisSystem
from modules.function_calling_system import FunctionCallingSystem

# Import autonomous blog scheduler
try:
    from core.services.autonomous_blog_scheduler import AutonomousBlogScheduler
    BLOG_SCHEDULER_AVAILABLE = True
except ImportError:
    BLOG_SCHEDULER_AVAILABLE = False

# Import Snake Agent state for restoration (conditionally)
try:
    from core.snake_agent import SnakeAgentState
except ImportError:
    SnakeAgentState = None

# Import Conversational AI module
try:
    from modules.conversational_ai.main import ConversationalAI
    CONVERSATIONAL_AI_AVAILABLE = True
except ImportError:
    CONVERSATIONAL_AI_AVAILABLE = False

logger = logging.getLogger(__name__)


class AGISystem:
    def __init__(self, engine):
        logger.info("Initializing Ravana AGI System...")

        self.engine = engine
        self.session = Session(engine)
        self.config = Config()
        # Initialize LLM selector with config
        initialize_llm_selector(self.config.PROVIDERS_CONFIG)

        # Initialize background_tasks early to prevent attribute errors
        self.background_tasks = []

        # Load shared models using the intelligent embeddings manager
        self.embedding_model = embeddings_manager
        from core.model_manager import create_sentiment_classifier
        self.sentiment_classifier = create_sentiment_classifier()

        # Use enhanced memory service
        self.memory_service = enhanced_memory_service

        # Initialize cognitive architecture components
        self.cognitive_architecture = self._initialize_cognitive_architecture()
        
        # Initialize advanced reasoning engine
        from core.reasoning_engine import ReasoningEngine, ReasoningType
        self.reasoning_engine = ReasoningEngine(self.memory_service)
        
        # Initialize self-reflection and self-modification capabilities
        from core.self_reflection_module import SelfReflectionModule, SelfModificationModule
        self.self_reflection = SelfReflectionModule(self)
        self.self_modification = SelfModificationModule(self)
        
        # Initialize services
        config = Config()
        self.data_service = DataService(
            engine,
            config.FEED_URLS,
            self.embedding_model,
            self.sentiment_classifier
        )
        self.knowledge_service = KnowledgeService(engine)
        # Use enhanced memory service instead of basic one
        self.memory_service = enhanced_memory_service

        # Initialize autonomous blog scheduler
        if BLOG_SCHEDULER_AVAILABLE:
            self.blog_scheduler = AutonomousBlogScheduler(self)
            logger.info("Autonomous blog scheduler initialized")
        else:
            self.blog_scheduler = None
            logger.info("Autonomous blog scheduler not available")

        # Initialize modules with blog scheduler integration
        self.situation_generator = SituationGenerator(
            embedding_model=self.embedding_model,
            sentiment_classifier=self.sentiment_classifier
        )
        self.emotional_intelligence = EmotionalIntelligence()
        self.curiosity_trigger = CuriosityTrigger(
            blog_scheduler=self.blog_scheduler)  # Enhanced with blog integration
        self.reflection_module = ReflectionModule(
            self, blog_scheduler=self.blog_scheduler)

        self.experimentation_engine = AGIExperimentationEngine(
            self, blog_scheduler=self.blog_scheduler)

        # Initialize enhanced systems for failure learning and physics analysis
        self.failure_learning_system = FailureLearningSystem(
            self, blog_scheduler=self.blog_scheduler)
        self.physics_analysis_system = PhysicsAnalysisSystem(
            self, blog_scheduler=self.blog_scheduler)
        self.function_calling_system = FunctionCallingSystem(
            self, blog_scheduler=self.blog_scheduler)

        # Initialize enhanced action manager
        self.action_manager = ActionManager(self, self.data_service)

        # Initialize adaptive learning engine with blog scheduler
        self.learning_engine = AdaptiveLearningEngine(
            self, blog_scheduler=self.blog_scheduler)

        # Initialize multi-modal service
        self.multi_modal_service = MultiModalService()

        # Initialize personality
        config = Config()
        self.personality = Personality(
            name=config.PERSONA_NAME,
            origin=config.PERSONA_ORIGIN,
            creativity=config.PERSONA_CREATIVITY
        )
        
        # Track personality consistency
        self.personality_consistency_tracker = self.personality.get_personality_consistency_report()
        
        # Initialize error recovery manager
        from core.error_recovery.error_recovery_manager import initialize_error_recovery_manager
        self.error_recovery_manager = None  # Will be initialized in initialize_components
        
        # Initialize resource manager
        from core.resource_management.resource_manager import initialize_resource_manager
        self.resource_manager = None  # Will be initialized in initialize_components

        # Initialize Snake Agent if enabled (Enhanced Version)
        self.snake_agent = None
        config = Config()
        if config.SNAKE_AGENT_ENABLED:
            try:
                # Try enhanced version first, fall back to original if needed
                enhanced_mode = getattr(config, 'SNAKE_ENHANCED_MODE', True)
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
                    logger.error(
                        f"Fallback Snake Agent also failed: {fallback_error}")
                    self.snake_agent = None

        # Initialize Conversational AI if enabled
        self.conversational_ai = None
        self.conversational_ai_thread = None
        # Track if Conversational AI has been started to prevent multiple instances
        self._conversational_ai_started = False
        config = Config()
        if config.CONVERSATIONAL_AI_ENABLED and CONVERSATIONAL_AI_AVAILABLE:
            try:
                self.conversational_ai = ConversationalAI()
                logger.info("Conversational AI module initialized")
            except Exception as e:
                logger.error(
                    f"Failed to initialize Conversational AI module: {e}")
                self.conversational_ai = None

        # Initialize Physics Prototyping System
        try:
            from modules.physics_prototyping_system import PhysicsPrototypingSystem
            self.physics_prototyping_system = PhysicsPrototypingSystem(self, self.blog_scheduler)
            logger.info("Physics Prototyping System initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Physics Prototyping System: {e}")
            self.physics_prototyping_system = None

        # Initialize Mad Scientist System
        try:
            from modules.mad_scientist_system import MadScientistSystem
            self.mad_scientist_system = MadScientistSystem(self, self.blog_scheduler)
            logger.info("Mad Scientist System initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Mad Scientist System: {e}")
            self.mad_scientist_system = None

        # New state for multi-step plans
        self.current_plan: List[Dict] = []
        self.current_task_prompt: str = None

        # For graceful shutdown
        self._shutdown = asyncio.Event()
        self.shutdown_coordinator = ShutdownCoordinator(self)
        
        # Performance monitoring
        self.performance_tracker = performance_tracker
        
        # Self-goal management
        self.self_goal_manager = self_goal_manager

        # Register cleanup handlers
        self.shutdown_coordinator.register_cleanup_handler(
            self._cleanup_database_session)
        self.shutdown_coordinator.register_cleanup_handler(
            self._cleanup_models)
        self.shutdown_coordinator.register_cleanup_handler(
            self._save_final_state, is_async=True)

        # Register key services with shutdown coordinator
        self.shutdown_coordinator.register_component(
            self.memory_service, ShutdownPriority.HIGH, is_async=True)
        self.shutdown_coordinator.register_component(
            self.data_service, ShutdownPriority.HIGH, is_async=True)
        self.shutdown_coordinator.register_component(
            self.knowledge_service, ShutdownPriority.HIGH, is_async=True)

        # Register blog scheduler if available
        if self.blog_scheduler:
            self.shutdown_coordinator.register_component(
                self.blog_scheduler, ShutdownPriority.MEDIUM, is_async=True)

        # Register Snake Agent if enabled (it implements the Shutdownable interface)
        if self.snake_agent:
            self.shutdown_coordinator.register_component(
                self.snake_agent, ShutdownPriority.HIGH, is_async=True)

        # Register Conversational AI if enabled
        if self.conversational_ai:
            self.shutdown_coordinator.register_component(
                self.conversational_ai, ShutdownPriority.MEDIUM, is_async=True)

        # Register Physics Prototyping System if available
        if hasattr(self, 'physics_prototyping_system') and self.physics_prototyping_system:
            self.shutdown_coordinator.register_component(
                self.physics_prototyping_system, ShutdownPriority.MEDIUM, is_async=True)

        # Register Mad Scientist System if available
        if hasattr(self, 'mad_scientist_system') and self.mad_scientist_system:
            self.shutdown_coordinator.register_component(
                self.mad_scientist_system, ShutdownPriority.MEDIUM, is_async=True)

        # Register experimentation modules
        if hasattr(self, 'experimentation_engine'):
            self.shutdown_coordinator.register_component(
                self.experimentation_engine, ShutdownPriority.MEDIUM, is_async=True)

        if hasattr(self, 'reflection_module'):
            self.shutdown_coordinator.register_component(
                self.reflection_module, ShutdownPriority.LOW, is_async=True)

        # Register enhanced learning and analysis modules
        if hasattr(self, 'failure_learning_system'):
            self.shutdown_coordinator.register_component(
                self.failure_learning_system, ShutdownPriority.MEDIUM, is_async=True)

        if hasattr(self, 'physics_analysis_system'):
            self.shutdown_coordinator.register_component(
                self.physics_analysis_system, ShutdownPriority.MEDIUM, is_async=True)

        if hasattr(self, 'function_calling_system'):
            self.shutdown_coordinator.register_component(
                self.function_calling_system, ShutdownPriority.MEDIUM, is_async=True)

        if hasattr(self, 'multi_modal_service'):
            self.shutdown_coordinator.register_component(
                self.multi_modal_service, ShutdownPriority.MEDIUM, is_async=True)

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

        # Initialize background tasks list
        self.background_tasks = []

    async def _check_for_search_results(self):
        """Enhanced search result processing with better error handling."""
        try:
            search_result = search_result_manager.get_result()
            if search_result:
                logger.info(
                    f"Retrieved search result: {search_result[:100]}...")

                # Add to shared state for immediate use
                if not hasattr(self.shared_state, 'search_results'):
                    self.shared_state.search_results = []
                self.shared_state.search_results.append(search_result)

                # Limit search results to prevent memory bloat
                if len(self.shared_state.search_results) > 10:
                    self.shared_state.search_results = self.shared_state.search_results[-10:]

                # Add to memory for long-term retention with better error handling
                try:
                    memory_summary = f"Search result retrieved: {search_result[:300]}..."
                    memories_to_save = await self.memory_service.extract_memories(memory_summary, "")
                    if memories_to_save and hasattr(memories_to_save, 'memories') and memories_to_save.memories:
                        await self.memory_service.save_memories(memories_to_save.memories)
                        logger.info(
                            f"Saved {len(memories_to_save.memories)} memories from search result")
                except Exception as e:
                    logger.warning(
                        f"Failed to save search result to memory: {e}")

                # Add to knowledge base for future reference (with enhanced error handling)
                try:
                    knowledge_result = await asyncio.to_thread(
                        self.knowledge_service.add_knowledge,
                        content=search_result,
                        source="web_search",
                        category="search_result"
                    )
                    if not knowledge_result.get('duplicate', False):
                        logger.info(
                            "Added new search result to knowledge base")
                    else:
                        logger.info(
                            "Search result already exists in knowledge base")
                except Exception as e:
                    logger.warning(
                        f"Failed to add search result to knowledge base: {e}")

        except Exception as e:
            logger.error(
                f"Error in search result processing: {e}", exc_info=True)

    async def initialize_components(self):
        """Initialize all system components with better error handling and retry logic."""
        logger.info("Initializing RAVANA AGI system components...")

        # Initialize components in order of dependency
        components = [
            ("Database Session", self._initialize_database_session),
            ("Embedding Model", self._initialize_embedding_model),
            ("Sentiment Classifier", self._initialize_sentiment_classifier),
            ("Data Service", self._initialize_data_service),
            ("Knowledge Service", self._initialize_knowledge_service),
            ("Memory Service", self._initialize_memory_service),
            ("Blog Scheduler", self._initialize_blog_scheduler),
            ("Cognitive Architecture", self._initialize_cognitive_architecture),
            ("Reasoning Engine", self._initialize_reasoning_engine),
            ("Self-Reflection Module", self._initialize_self_reflection_module),
            ("Self-Modification Module", self._initialize_self_modification_module),
            ("Experimentation Module", self._initialize_experimentation_module),
            ("Modules", self._initialize_modules),
            ("Action Manager", self._initialize_action_manager),
            ("Personality", self._initialize_personality),
            ("Error Recovery Manager", self._initialize_error_recovery_manager),
            ("Resource Manager", self._initialize_resource_manager),
            ("Snake Agent", self._initialize_snake_agent),
            ("Conversational AI", self._initialize_conversational_ai),
            ("Shutdown Coordinator", self._initialize_shutdown_coordinator)
        ]

        initialized_components = []
        failed_components = []

        for component_name, init_func in components:
            try:
                logger.info(f"Initializing {component_name}...")
                # Check if the function is a coroutine (async) or regular function
                if asyncio.iscoroutinefunction(init_func):
                    await init_func()
                else:
                    # For synchronous functions, call them directly without await
                    init_func()
                initialized_components.append(component_name)
                logger.info(f"✓ {component_name} initialized successfully")
            except Exception as e:
                logger.error(
                    f"✗ Failed to initialize {component_name}: {e}", exc_info=True)
                failed_components.append((component_name, str(e)))

                # For critical components, we might want to stop initialization
                if component_name in ["Database Session", "Embedding Model"]:
                    logger.error(
                        f"Critical component {component_name} failed, stopping initialization")
                    break

        logger.info(
            f"Initialization complete. {len(initialized_components)} components initialized, {len(failed_components)} failed.")

        if failed_components:
            logger.warning("Failed components:")
            for name, error in failed_components:
                logger.warning(f"  - {name}: {error}")

        # Load previous state, goals, and initiate goal setting
        try:
            logger.info("Loading previous state...")
            await self._load_previous_state()
            logger.info("Loading self-goals...")
            await self.self_goal_manager.load_goals()
            logger.info("Initiating goal setting...")
            await self._initiate_goal_setting()
        except Exception as e:
            logger.error(f"Error during post-initialization tasks: {e}", exc_info=True)
            failed_components.append(("Post-initialization", str(e)))

        return len(failed_components) == 0

    async def _initialize_database_session(self):
        """Initialize database session."""
        try:
            self.session = Session(self.engine)
            logger.info("Database session initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database session: {e}")
            raise

    def _initialize_embedding_model(self):
        """Initialize embedding model using the intelligent embeddings manager."""
        try:
            # The embeddings manager handles AI-driven selection internally
            self.embedding_model = embeddings_manager
            logger.info(
                f"Embedding model manager initialized with AI-driven selection")

            # Initialize LLM selector with config
            initialize_llm_selector(self.config.PROVIDERS_CONFIG)
            logger.info("LLM selector initialized with provider configuration")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise

    def _initialize_sentiment_classifier(self):
        """Initialize sentiment classifier."""
        try:
            from core.model_manager import create_sentiment_classifier
            self.sentiment_classifier = create_sentiment_classifier()
            logger.info("Sentiment classifier initialized")
        except Exception as e:
            logger.error(f"Failed to initialize sentiment classifier: {e}")
            raise

    def _initialize_data_service(self):
        """Initialize data service."""
        try:
            config = Config()
            self.data_service = DataService(
                self.engine,
                config.FEED_URLS,
                self.embedding_model,
                self.sentiment_classifier
            )
            logger.info("Data service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize data service: {e}")
            raise

    def _initialize_knowledge_service(self):
        """Initialize knowledge service."""
        try:
            self.knowledge_service = KnowledgeService(self.engine)
            logger.info("Knowledge service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize knowledge service: {e}")
            raise

    def _initialize_memory_service(self):
        """Initialize enhanced memory service."""
        try:
            self.memory_service = enhanced_memory_service
            logger.info(
                "Enhanced memory service initialized with AI-driven capabilities")
        except Exception as e:
            logger.error(f"Failed to initialize enhanced memory service: {e}")
            raise

    def _initialize_blog_scheduler(self):
        """Initialize blog scheduler."""
        try:
            if BLOG_SCHEDULER_AVAILABLE:
                self.blog_scheduler = AutonomousBlogScheduler(self)
                logger.info("Autonomous blog scheduler initialized")
            else:
                self.blog_scheduler = None
                logger.info("Autonomous blog scheduler not available")
        except Exception as e:
            logger.error(f"Failed to initialize blog scheduler: {e}")
            self.blog_scheduler = None

    def _initialize_modules(self):
        """Initialize core modules."""
        try:
            self.situation_generator = SituationGenerator(
                embedding_model=self.embedding_model,
                sentiment_classifier=self.sentiment_classifier
            )
            self.emotional_intelligence = EmotionalIntelligence()
            self.curiosity_trigger = CuriosityTrigger(
                blog_scheduler=self.blog_scheduler)
            self.reflection_module = ReflectionModule(
                self, blog_scheduler=self.blog_scheduler)
            from modules.experimentation_module import ExperimentationModule
            self.experimentation_module = ExperimentationModule(
                self, blog_scheduler=self.blog_scheduler)
            self.experimentation_engine = AGIExperimentationEngine(
                self, blog_scheduler=self.blog_scheduler)
            logger.info("Core modules initialized")
        except Exception as e:
            logger.error(f"Failed to initialize core modules: {e}")
            raise

    def _initialize_action_manager(self):
        """Initialize action manager."""
        try:
            self.action_manager = ActionManager(
                self, self.data_service)
            logger.info("Enhanced action manager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize action manager: {e}")
            raise

    def _initialize_personality(self):
        """Initialize personality."""
        try:
            config = Config()
            self.personality = Personality(
                name=config.PERSONA_NAME,
                origin=config.PERSONA_ORIGIN,
                creativity=config.PERSONA_CREATIVITY
            )
            logger.info("Personality initialized")
        except Exception as e:
            logger.error(f"Failed to initialize personality: {e}")
            raise

    def _initialize_snake_agent(self):
        """Initialize Snake Agent with better error handling."""
        try:
            config = Config()
            self.snake_agent = None
            if config.SNAKE_AGENT_ENABLED:
                try:
                    # Try enhanced version first, fall back to original if needed
                    enhanced_mode = getattr(
                        config, 'SNAKE_ENHANCED_MODE', True)
                    if enhanced_mode:
                        from core.snake_agent_enhanced import EnhancedSnakeAgent
                        self.snake_agent = EnhancedSnakeAgent(self)
                        logger.info(
                            "Enhanced Snake Agent initialized and ready")
                    else:
                        from core.snake_agent import SnakeAgent
                        self.snake_agent = SnakeAgent(self)
                        logger.info(
                            "Standard Snake Agent initialized and ready")
                except Exception as e:
                    logger.error(f"Failed to initialize Snake Agent: {e}")
                    # Fallback to standard version if enhanced fails
                    try:
                        from core.snake_agent import SnakeAgent
                        self.snake_agent = SnakeAgent(self)
                        logger.info(
                            "Fallback to standard Snake Agent successful")
                    except Exception as fallback_error:
                        logger.error(
                            f"Fallback Snake Agent also failed: {fallback_error}")
                        self.snake_agent = None
        except Exception as e:
            logger.error(f"Error in Snake Agent initialization: {e}")
            self.snake_agent = None

    def _initialize_conversational_ai(self):
        """Initialize Conversational AI."""
        try:
            self.conversational_ai = None
            self.conversational_ai_thread = None
            config = Config()
            if config.CONVERSATIONAL_AI_ENABLED and CONVERSATIONAL_AI_AVAILABLE:
                try:
                    # Initialize Conversational AI with standalone=False since it will run in the main system
                    self.conversational_ai = ConversationalAI()
                    logger.info("Conversational AI module initialized")
                except Exception as e:
                    logger.error(
                        f"Failed to initialize Conversational AI module: {e}")
                    self.conversational_ai = None
        except Exception as e:
            logger.error(f"Error in Conversational AI initialization: {e}")
            self.conversational_ai = None

    def _initialize_shutdown_coordinator(self):
        """Initialize shutdown coordinator."""
        try:
            self.shutdown_coordinator = ShutdownCoordinator(self)

            # Register cleanup handlers
            self.shutdown_coordinator.register_cleanup_handler(
                self._cleanup_database_session)
            self.shutdown_coordinator.register_cleanup_handler(
                self._cleanup_models)
            self.shutdown_coordinator.register_cleanup_handler(
                self._save_final_state, is_async=True)

            # Register Snake Agent cleanup if enabled
            if self.snake_agent:
                self.shutdown_coordinator.register_cleanup_handler(
                    self._cleanup_snake_agent, is_async=True)

            # Register Conversational AI cleanup if enabled
            if self.conversational_ai:
                self.shutdown_coordinator.register_cleanup_handler(
                    self._cleanup_conversational_ai, is_async=False)
            
            logger.info("Shutdown coordinator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize shutdown coordinator: {e}")
            raise

    async def _initialize_error_recovery_manager(self):
        """Initialize error recovery manager."""
        try:
            from core.error_recovery.error_recovery_manager import initialize_error_recovery_manager
            self.error_recovery_manager = await initialize_error_recovery_manager(self)
            logger.info("Error recovery manager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize error recovery manager: {e}")
            raise

    async def _initialize_resource_manager(self):
        """Initialize resource manager."""
        try:
            from core.resource_management.resource_manager import initialize_resource_manager
            self.resource_manager = await initialize_resource_manager(self)
            logger.info("Resource manager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize resource manager: {e}")
            raise

    def _initialize_cognitive_architecture(self):
        """Initialize cognitive architecture components for the AGI system."""
        try:
            # Initialize cognitive control systems
            self.cognitive_control = {
                'attention': {
                    'focus_level': 0.5,
                    'attention_spans': [],  # Tracks current focus areas
                    'distraction_filter': 0.7,  # How well it filters distractions
                    'salience_threshold': 0.3,  # Minimum salience for attention
                    'attention_shift_threshold': 0.1  # When to shift attention
                },
                'working_memory': {
                    'capacity': 7,  # Theoretical capacity of working memory
                    'active_items': [],
                    'decay_rate': 0.1,  # How quickly items fade from working memory
                    'rehearsal_rate': 0.3,  # Rate at which items are rehearsed to prevent decay
                    'capacity_utilization': 0.0  # Current utilization of working memory
                },
                'executive_function': {
                    'planning_horizon': 5,  # Number of steps it can plan ahead
                    'inhibition': 0.8,  # Ability to inhibit impulsive responses
                    'cognitive_flexibility': 0.6,  # Ability to switch between tasks/concepts
                    'task_switching_cost': 0.2  # Cost of switching between tasks
                }
            }
            
            # Initialize metacognitive awareness
            self.metacognitive_awareness = {
                'confidence_tracking': [],  # Track confidence in decisions
                'error_detection': 0.8,  # Ability to detect errors in reasoning
                'uncertainty_quantification': 0.5,  # How well it assesses uncertainty
                'knowledge_limited_awareness': True,  # Knows when knowledge is limited
                'learning_progress_monitoring': True,  # Monitors learning effectiveness
                'memory_reliability_assessment': 0.7  # Assesses reliability of retrieved memories
            }
            
            # Initialize goal hierarchy system
            self.goal_hierarchy = {
                'top_level_goals': [],  # Long-term objectives (e.g., self-improvement)
                'mid_level_goals': [],  # Medium-term objectives (e.g., learning new skills)
                'task_level_goals': [],  # Immediate action goals
                'goal_conflict_resolver': self._resolve_goal_conflicts,
                'goal_interdependence_map': {},  # Maps dependencies between goals
                'goal_progress_tracker': {}  # Tracks progress toward each goal
            }
            
            # Initialize memory management for cognitive architecture
            self.memory_manager = self._initialize_memory_manager()
            
            logger.info("Cognitive architecture initialized with attention, memory, and executive functions")
            return self.cognitive_control
        except Exception as e:
            logger.error(f"Failed to initialize cognitive architecture: {e}")
            # Return a basic structure even if initialization fails
            return {
                'attention': {'focus_level': 0.5, 'attention_spans': [], 'distraction_filter': 0.7},
                'working_memory': {'capacity': 7, 'active_items': [], 'decay_rate': 0.1},
                'executive_function': {'planning_horizon': 5, 'inhibition': 0.8, 'cognitive_flexibility': 0.6}
            }

    def _initialize_memory_manager(self):
        """Initialize memory management for the cognitive architecture."""
        try:
            # Create a memory manager that interfaces with the cognitive architecture
            memory_manager = {
                'episodic_buffer': [],  # Current episode being processed
                'semantic_retrieval_functions': self._create_semantic_retrieval_functions(),
                'episodic_retrieval_functions': self._create_episodic_retrieval_functions(),
                'memory_integration_threshold': 0.6,  # Similarity threshold for integration
                'memory_consolidation_scheduler': None,  # Will be initialized later when event loop is available
                'memory_tagging_system': self._initialize_memory_tagging()
            }
            
            logger.info("Memory manager initialized for cognitive architecture")
            return memory_manager
        except Exception as e:
            logger.error(f"Failed to initialize memory manager: {e}")
            return {}

    def _create_semantic_retrieval_functions(self):
        """Create functions for semantic memory retrieval."""
        async def retrieve_factual_knowledge(query: str, top_k: int = 5):
            """Retrieve factual knowledge from semantic memory."""
            try:
                # Retrieve only semantic memories
                semantic_memories = self.memory_service.get_memories_by_type(MemoryType.SEMANTIC)
                if not semantic_memories:
                    return []
                
                # Create a query embedding and find most relevant semantic memories
                from core.embeddings_manager import embeddings_manager, ModelPurpose
                query_embedding = embeddings_manager.get_embedding(query, purpose=ModelPurpose.SEMANTIC_SEARCH)
                
                # Calculate similarities
                similarities = []
                for memory in semantic_memories:
                    memory_embedding = memory.embedding
                    try:
                        # Calculate cosine similarity
                        dot_product = sum(a * b for a, b in zip(query_embedding, memory_embedding))
                        norm_query = sum(a * a for a in query_embedding) ** 0.5
                        norm_memory = sum(a * a for a in memory_embedding) ** 0.5
                        
                        if norm_query == 0 or norm_memory == 0:
                            similarity = 0.0
                        else:
                            similarity = dot_product / (norm_query * norm_memory)
                        
                        similarities.append((memory, similarity))
                    except Exception as e:
                        logger.warning(f"Error calculating similarity for semantic memory {memory.id}: {e}")
                        continue
                
                # Sort by similarity and return top_k
                similarities.sort(key=lambda x: x[1], reverse=True)
                return similarities[:top_k]
            except Exception as e:
                logger.error(f"Error retrieving semantic knowledge: {e}")
                return []
        
        return {
            'retrieve_factual_knowledge': retrieve_factual_knowledge
        }

    def _create_episodic_retrieval_functions(self):
        """Create functions for episodic memory retrieval."""
        async def retrieve_episodic_memories(context: str, top_k: int = 5):
            """Retrieve relevant episodic memories based on context."""
            try:
                # Retrieve only episodic memories
                episodic_memories = self.memory_service.get_memories_by_type(MemoryType.EPISODIC)
                if not episodic_memories:
                    return []
                
                # Create a query embedding and find most relevant episodic memories
                from core.embeddings_manager import embeddings_manager, ModelPurpose
                query_embedding = embeddings_manager.get_embedding(context, purpose=ModelPurpose.SEMANTIC_SEARCH)
                
                # Calculate similarities
                similarities = []
                for memory in episodic_memories:
                    memory_embedding = memory.embedding
                    try:
                        # Calculate cosine similarity
                        dot_product = sum(a * b for a, b in zip(query_embedding, memory_embedding))
                        norm_query = sum(a * a for a in query_embedding) ** 0.5
                        norm_memory = sum(a * a for a in memory_embedding) ** 0.5
                        
                        if norm_query == 0 or norm_memory == 0:
                            similarity = 0.0
                        else:
                            similarity = dot_product / (norm_query * norm_memory)
                        
                        similarities.append((memory, similarity))
                    except Exception as e:
                        logger.warning(f"Error calculating similarity for episodic memory {memory.id}: {e}")
                        continue
                
                # Sort by similarity and return top_k
                similarities.sort(key=lambda x: x[1], reverse=True)
                return similarities[:top_k]
            except Exception as e:
                logger.error(f"Error retrieving episodic memories: {e}")
                return []
        
        return {
            'retrieve_episodic_memories': retrieve_episodic_memories
        }

    def _schedule_memory_consolidation(self):
        """Schedule memory consolidation tasks."""
        import asyncio
        
        async def periodic_consolidation():
            """Periodically consolidate memories."""
            while not self._shutdown.is_set():
                try:
                    # Wait for consolidation interval
                    await asyncio.sleep(7200)  # Every 2 hours
                    
                    # Perform memory consolidation
                    consolidation_result = await self.memory_service.consolidate_old_memories()
                    logger.info(f"Memory consolidation completed: {consolidation_result}")
                except Exception as e:
                    logger.error(f"Error during periodic memory consolidation: {e}")
        
        # Start the consolidation task in the background
        consolidation_task = asyncio.create_task(periodic_consolidation())
        self.background_tasks.append(consolidation_task)
        
        return consolidation_task

    def _initialize_memory_tagging(self):
        """Initialize memory tagging system."""
        return {
            'semantic_tags': set(),  # Tags for semantic memories
            'episodic_tags': set(),  # Tags for episodic memories
            'procedural_tags': set(),  # Tags for procedural memories
            'auto_tagging_enabled': True,  # Whether auto-tagging is enabled
            'tagging_rules': self._create_tagging_rules()  # Rules for automatic tagging
        }

    def _create_tagging_rules(self):
        """Create rules for automatic memory tagging."""
        return {
            'topic_identification': ['science', 'technology', 'mathematics', 'physics', 'biology', 'chemistry'],
            'task_identification': ['problem_solving', 'learning', 'decision_making', 'planning'],
            'learning_identification': ['new_skill', 'concept_understanding', 'fact_learning'],
            'emotional_identification': ['positive_experience', 'negative_experience', 'learning_experience']
        }

    def _resolve_goal_conflicts(self, goals):
        """Resolve conflicts between competing goals based on priority and context."""
        # Simple priority-based conflict resolution
        # In a real implementation, this would be more sophisticated
        return sorted(goals, key=lambda g: g.get('priority', 0), reverse=True)[:1]

    def _initialize_reasoning_engine(self):
        """Initialize advanced reasoning engine."""
        try:
            from core.reasoning_engine import ReasoningEngine
            self.reasoning_engine = ReasoningEngine(self.memory_service)
            logger.info("Reasoning engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize reasoning engine: {e}")
            raise

    def _initialize_self_reflection_module(self):
        """Initialize self-reflection module."""
        try:
            from core.self_reflection_module import SelfReflectionModule
            self.self_reflection = SelfReflectionModule(self)
            logger.info("Self-reflection module initialized")
        except Exception as e:
            logger.error(f"Failed to initialize self-reflection module: {e}")
            raise

    def _initialize_self_modification_module(self):
        """Initialize self-modification module."""
        try:
            from core.self_reflection_module import SelfModificationModule
            self.self_modification = SelfModificationModule(self)
            logger.info("Self-modification module initialized")
        except Exception as e:
            logger.error(f"Failed to initialize self-modification module: {e}")
            raise

    def _initialize_experimentation_module(self):
        """Initialize experimentation module."""
        try:
            from modules.experimentation_module import ExperimentationModule
            self.experimentation_module = ExperimentationModule(self, self.blog_scheduler)
            logger.info("Experimentation module initialized")
        except Exception as e:
            logger.error(f"Failed to initialize experimentation module: {e}")
            raise

    async def stop(self, reason: str = "system_shutdown"):
        """
        Stop the AGI system gracefully.

        Args:
            reason: Reason for stopping the system
        """
        logger.info(f"Stopping AGI system - Reason: {reason}")
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
            if hasattr(self, 'sentiment_classifier'):
                del self.sentiment_classifier

            # Use embeddings manager to properly unload all embedding models
            embeddings_manager.unload_all_models()

            logger.info("Model references cleared and memory freed")
        except Exception as e:
            logger.error(f"Error cleaning up models: {e}")

    async def _save_final_state(self):
        """Save final system state before shutdown."""
        try:
            config = Config()
            if config.STATE_PERSISTENCE_ENABLED:
                # This is handled by the shutdown coordinator
                # but we can add any AGI-specific state saving here
                logger.info(
                    "Final state saving handled by shutdown coordinator")
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

    def _cleanup_conversational_ai(self):
        """Clean up Conversational AI resources."""
        try:
            if self.conversational_ai:
                # Signal the Conversational AI to stop
                logger.info("Conversational AI cleanup requested")
                # Set the shutdown event if it exists
                if hasattr(self.conversational_ai, '_shutdown'):
                    self.conversational_ai._shutdown.set()
                # Reset the started flag
                self._conversational_ai_started = False
        except Exception as e:
            logger.error(f"Error cleaning up Conversational AI: {e}")

    async def start_snake_agent(self):
        """Start Snake Agent background operation."""
        if self.snake_agent and self.config.SNAKE_AGENT_ENABLED:
            try:
                logger.info("Starting Snake Agent background operation...")
                snake_task = asyncio.create_task(
                    self.snake_agent.start_autonomous_operation())
                self.background_tasks.append(snake_task)
                logger.info("Snake Agent started successfully")
            except Exception as e:
                logger.error(f"Failed to start Snake Agent: {e}")

    async def start_conversational_ai(self):
        """Start Conversational AI module in a separate thread."""
        if self.conversational_ai and self.config.CONVERSATIONAL_AI_ENABLED:
            # Check if already started to prevent multiple instances
            if self._conversational_ai_started:
                logger.warning(
                    "Conversational AI module already started, skipping...")
                return

            try:
                logger.info("Starting Conversational AI module...")

                # Create a task to run the Conversational AI in the same event loop
                async def run_conversational_ai():
                    try:
                        # Add a small delay to allow the main system to initialize
                        await asyncio.sleep(self.config.CONVERSATIONAL_AI_START_DELAY)
                        # Run the conversational AI as part of the main system (not standalone)
                        await self.conversational_ai.start(standalone=False)
                    except Exception as e:
                        logger.error(f"Error in Conversational AI: {e}")
                        logger.exception("Full traceback:")

                # Schedule the Conversational AI to run as a task in the current event loop
                conversational_ai_task = asyncio.create_task(
                    run_conversational_ai())
                self.background_tasks.append(conversational_ai_task)
                # Mark as started
                self._conversational_ai_started = True
                logger.info(
                    "Conversational AI module started successfully as async task")

                # Give the bots a moment to start up and connect
                await asyncio.sleep(2)

                # Check if bots are connected
                discord_connected = False
                telegram_connected = False

                if self.conversational_ai.discord_bot:
                    discord_connected = getattr(
                        self.conversational_ai.discord_bot, 'connected', False)

                if self.conversational_ai.telegram_bot:
                    telegram_connected = getattr(
                        self.conversational_ai.telegram_bot, 'connected', False)

                if discord_connected or telegram_connected:
                    logger.info(
                        f"Conversational AI bots connected - Discord: {discord_connected}, Telegram: {telegram_connected}")
                else:
                    logger.warning(
                        "Conversational AI bots are not connected. Check tokens and network connectivity.")

            except Exception as e:
                logger.error(f"Failed to start Conversational AI module: {e}")
                logger.exception("Full traceback:")
                # Reset the started flag on error
                self._conversational_ai_started = False

    async def get_snake_agent_status(self) -> Dict[str, Any]:
        """Get Snake Agent status information."""
        if not self.snake_agent:
            return {"enabled": False, "status": "not_initialized"}

        status = await self.snake_agent.get_status()
        return {
            "enabled": self.config.SNAKE_AGENT_ENABLED,
            "status": "active" if self.snake_agent.running else "inactive",
            **status
        }

    def get_conversational_ai_status(self) -> Dict[str, Any]:
        """Get Conversational AI status information."""
        if not self.conversational_ai:
            return {"enabled": False, "status": "not_initialized"}

        # Check if the conversational AI has been started
        started = getattr(self, '_conversational_ai_started', False)

        # Check if bots are connected
        discord_connected = False
        telegram_connected = False

        if self.conversational_ai.discord_bot:
            discord_connected = getattr(
                self.conversational_ai.discord_bot, 'connected', False)

        if self.conversational_ai.telegram_bot:
            telegram_connected = getattr(
                self.conversational_ai.telegram_bot, 'connected', False)

        # Determine overall status
        bot_connected = discord_connected or telegram_connected
        status = "active" if (started and bot_connected) else "inactive"

        return {
            "enabled": self.config.CONVERSATIONAL_AI_ENABLED,
            "status": status,
            "discord_connected": discord_connected,
            "telegram_connected": telegram_connected
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
                    self.emotional_intelligence.set_mood_vector(
                        agi_state["mood"])
                    logger.info("Restored previous mood state")
                except Exception as e:
                    logger.warning(f"Could not restore mood state: {e}")

            # Restore current plans
            if "current_plan" in agi_state:
                self.current_plan = agi_state["current_plan"]
                self.current_task_prompt = agi_state.get("current_task_prompt")
                if self.current_plan:
                    logger.info(
                        f"Restored plan with {len(self.current_plan)} remaining steps")

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
                logger.info(
                    f"Restored {len(self.invention_history)} invention history entries")

            # Restore Snake Agent state - handle both EnhancedSnakeAgent and SnakeAgent
            if "snake_agent" in agi_state and self.snake_agent:
                try:
                    snake_data = agi_state["snake_agent"]
                    if "state" in snake_data:
                        # Check if this is the EnhancedSnakeAgent, which has its own state management
                        from core.snake_agent_enhanced import EnhancedSnakeAgent
                        if isinstance(self.snake_agent, EnhancedSnakeAgent):
                            self.snake_agent.state = snake_data["state"]
                            logger.info("Successfully set EnhancedSnakeAgent state.")
                        else:
                            # For regular SnakeAgent, use the original restoration logic
                            restored_state = SnakeAgentState.from_dict(
                                snake_data["state"])
                            self.snake_agent.state = restored_state

                            # Restore counters
                            self.snake_agent.analysis_count = snake_data.get(
                                "analysis_count", 0)
                            self.snake_agent.experiment_count = snake_data.get(
                                "experiment_count", 0)
                            self.snake_agent.communication_count = snake_data.get(
                                "communication_count", 0)

                            logger.info(
                                f"Restored Snake Agent state: {len(restored_state.pending_experiments)} pending experiments, {len(restored_state.communication_queue)} queued communications")
                except Exception as e:
                    logger.warning(f"Could not restore Snake Agent state: {e}")

            logger.info("✅ Previous system state restored successfully")

            # Clean up the state file after successful recovery
            cleanup_state_file()

        except Exception as e:
            logger.error(f"Error loading previous state: {e}")
            logger.info("Continuing with fresh initialization")

    async def _initiate_goal_setting(self):
        """Initiate the process of setting self-improvement goals."""
        try:
            logger.info("Initiating self-improvement goal setting...")
            
            # Wait for system to be fully initialized
            await asyncio.sleep(5)
            
            # Load any existing goals from previous sessions
            # The AGI will set new goals based on its learning and reflection
            logger.info("AGI will set its own goals based on learning and reflection")
            
            # Start the goal evaluation loop
            asyncio.create_task(self._goal_evaluation_loop())
            
            logger.info("Self-improvement goal setting initiated")
        except Exception as e:
            logger.error(f"Error initiating goal setting: {e}")

    async def _set_initial_improvement_goals(self):
        """Set initial improvement goals based on system analysis."""
        try:
            # Analyze current system performance
            performance_summary = self.performance_tracker.get_performance_summary()
            advanced_metrics = await self.performance_tracker.calculate_advanced_metrics()
            
            # Set goals based on performance gaps
            await self._analyze_and_set_performance_goals(performance_summary, advanced_metrics)
            await self._analyze_and_set_capability_goals()
            await self._analyze_and_set_learning_goals(advanced_metrics)
            
            logger.info(f"Set {len(self.self_goal_manager.goals)} initial improvement goals")
        except Exception as e:
            logger.error(f"Error setting initial improvement goals: {e}")

    async def _analyze_and_set_performance_goals(self, performance_summary: Dict[str, Any], advanced_metrics: Dict[str, Any]):
        """Analyze performance and set relevant goals."""
        # Set goal to improve low metrics
        if performance_summary.get("improvements_per_hour", 0) < 0.2:
            self.self_goal_manager.create_goal(
                title="Increase Improvement Generation Rate",
                description="Improve the rate of identifying and implementing system improvements",
                category="performance",
                priority=GoalPriority.HIGH,
                target_date=datetime.now(timezone.utc) + timedelta(days=14),
                metrics={"improvements_per_hour_target": 0.3},
                dependencies=[]
            )
        
        if performance_summary.get("improvement_success_rate", 0) < 0.6:
            self.self_goal_manager.create_goal(
                title="Improve Success Rate of Implementations",
                description="Increase the success rate of proposed improvements",
                category="quality",
                priority=GoalPriority.HIGH,
                target_date=datetime.now(timezone.utc) + timedelta(days=10),
                metrics={"success_rate_target": 0.7},
                dependencies=[]
            )
        
        if advanced_metrics.get("efficiency_metrics", {}).get("efficiency_score", 0) < 0.5:
            self.self_goal_manager.create_goal(
                title="Optimize Implementation Efficiency",
                description="Improve the efficiency of improvement implementations",
                category="efficiency",
                priority=GoalPriority.MEDIUM,
                target_date=datetime.now(timezone.utc) + timedelta(days=21),
                metrics={"efficiency_score_target": 0.7},
                dependencies=[]
            )

    async def _analyze_and_set_capability_goals(self):
        """Set goals to improve system capabilities."""
        # Set capability improvement goals
        self.self_goal_manager.create_goal(
            title="Enhance Memory Integration",
            description="Improve the connection between different memory systems",
            category="capability",
            priority=GoalPriority.HIGH,
            target_date=datetime.now(timezone.utc) + timedelta(days=30),
            metrics={"memory_linking_improvement": 0.8},
            dependencies=[]
        )
        
        self.self_goal_manager.create_goal(
            title="Improve Emotional Reasoning",
            description="Enhance the connection between emotional state and decision-making",
            category="capability",
            priority=GoalPriority.MEDIUM,
            target_date=datetime.now(timezone.utc) + timedelta(days=25),
            metrics={"emotional_reasoning_accuracy": 0.85},
            dependencies=[]
        )

    async def _analyze_and_set_learning_goals(self, advanced_metrics: Dict[str, Any]):
        """Set goals to improve learning capabilities."""
        if advanced_metrics.get("learning_metrics", {}).get("learning_rate", 0) < 0.1:
            self.self_goal_manager.create_goal(
                title="Improve Learning Rate",
                description="Enhance the system's ability to learn from experiences and experiments",
                category="learning",
                priority=GoalPriority.HIGH,
                target_date=datetime.now(timezone.utc) + timedelta(days=18),
                metrics={"learning_rate_target": 0.15},
                dependencies=[]
            )
        
        self.self_goal_manager.create_goal(
            title="Enhance Knowledge Integration",
            description="Improve the system's ability to connect new knowledge with existing knowledge",
            category="learning",
            priority=GoalPriority.MEDIUM,
            target_date=datetime.now(timezone.utc) + timedelta(days=35),
            metrics={"knowledge_integration_rate": 0.75},
            dependencies=[]
        )

    async def _goal_evaluation_loop(self):
        """Periodically evaluate goal progress and set new goals."""
        while not self._shutdown.is_set():
            try:
                logger.info("Evaluating self-improvement goals...")
                
                # Review current goals and their progress
                await self._review_goals_progress()
                
                # Set new goals based on changing needs
                await self._set_adaptive_goals()
                
                # Save goals state
                await self.self_goal_manager.save_goals()
                
                # Wait before next evaluation
                await asyncio.sleep(3600)  # Every hour
                
            except Exception as e:
                logger.error(f"Error in goal evaluation loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying

    async def _review_goals_progress(self):
        """Review the progress of current goals."""
        try:
            # Get all goals to review, not just high priority ones
            all_goals = self.self_goal_manager.goals
            if not all_goals:
                logger.info("No goals to review")
                return
            
            logger.info(f"Reviewing progress of {len(all_goals)} total goals")
            
            # Initialize metrics for tracking
            completed_goals = []
            in_progress_goals = []
            overdue_goals = []
            at_risk_goals = []
            
            current_time = datetime.now(timezone.utc)
            
            for goal_id, goal in all_goals.items():
                # Calculate progress percentage if possible
                if goal.target_date:
                    # Ensure both datetimes have timezone info to avoid mixing offset-naive and offset-aware
                    created_at_tz = goal.created_at
                    target_date_tz = goal.target_date
                    
                    # If goal.created_at is naive, assume UTC
                    if created_at_tz.tzinfo is None:
                        created_at_tz = created_at_tz.replace(tzinfo=timezone.utc)
                    
                    # If goal.target_date is naive, assume UTC  
                    if target_date_tz.tzinfo is None:
                        target_date_tz = target_date_tz.replace(tzinfo=timezone.utc)
                    
                    # Calculate time remaining as a percentage
                    total_duration = (target_date_tz - created_at_tz).total_seconds()
                    elapsed_duration = (current_time - created_at_tz).total_seconds()
                    time_progress = min(1.0, elapsed_duration / total_duration) if total_duration > 0 else 1.0
                else:
                    time_progress = 0.0
                
                # Evaluate goal status
                if goal.status == GoalStatus.COMPLETED:
                    completed_goals.append(goal)
                elif goal.status == GoalStatus.OVERDUE:
                    overdue_goals.append(goal)
                else:
                    # For in-progress goals, evaluate based on metrics
                    in_progress_goals.append(goal)
                    
                    # Check if goal is at risk based on time and progress
                    if (time_progress > 0.7 and goal.current_progress < 0.5) or \
                       (time_progress > 0.5 and goal.current_progress < 0.2):
                        at_risk_goals.append(goal)
            
            # Log summary
            logger.info(f"Goal review summary: {len(completed_goals)} completed, "
                       f"{len(in_progress_goals)} in progress, "
                       f"{len(overdue_goals)} overdue, "
                       f"{len(at_risk_goals)} at risk")
            
            # Perform more detailed analysis for at-risk goals
            for goal in at_risk_goals:
                logger.warning(f"Goal at risk: {goal.title} - "
                              f"Time progress: {time_progress:.1%}, "
                              f"Completion: {goal.current_progress:.1%}")
                
                # Trigger adaptive adjustments for at-risk goals
                await self._adjust_at_risk_goal(goal)
            
            # Update self-goal manager with analysis
            self.self_goal_manager.last_review_time = current_time
            self.self_goal_manager.goals_summary = {
                'total': len(all_goals),
                'completed': len(completed_goals),
                'in_progress': len(in_progress_goals),
                'overdue': len(overdue_goals),
                'at_risk': len(at_risk_goals),
                'review_time': current_time.isoformat()
            }
            
            # Update performance metrics based on goal progress
            await self._update_performance_metrics_from_goals(
                completed_goals, in_progress_goals, overdue_goals)
                
        except Exception as e:
            logger.error(f"Error in goal review process: {e}")
            logger.exception("Full traceback:")

    async def _adjust_at_risk_goal(self, goal):
        """Adjust an at-risk goal to improve its chances of success."""
        try:
            logger.info(f"Adjusting at-risk goal: {goal.title}")
            
            # Possible adjustments:
            # 1. Extend deadline if reasonable
            # 2. Break into smaller sub-goals
            # 3. Reallocate resources
            # 4. Adjust scope
            
            # For now, let's try extending the deadline by 25% if it's within reason
            if goal.target_date and goal.status != GoalStatus.OVERDUE:
                # Ensure both datetimes have timezone info to avoid mixing offset-naive and offset-aware
                created_at_tz = goal.created_at
                target_date_tz = goal.target_date
                
                # If goal.created_at is naive, assume UTC
                if created_at_tz.tzinfo is None:
                    created_at_tz = created_at_tz.replace(tzinfo=timezone.utc)
                
                # If goal.target_date is naive, assume UTC  
                if target_date_tz.tzinfo is None:
                    target_date_tz = target_date_tz.replace(tzinfo=timezone.utc)
                
                original_duration = (target_date_tz - created_at_tz).total_seconds()
                new_duration = original_duration * 1.25  # 25% extension
                new_target_date = created_at_tz + timedelta(seconds=new_duration)
                
                # Only extend if not extending too far
                max_extension = timedelta(days=30)  # Max 30 days extension
                if new_target_date - goal.target_date <= max_extension:
                    old_target = goal.target_date
                    goal.target_date = new_target_date
                    logger.info(f"Extended deadline for goal '{goal.title}' "
                               f"from {old_target} to {new_target_date}")
                    
                    # Update the goal in the manager
                    await self.self_goal_manager.update_goal(goal.id, {
                        'target_date': new_target_date
                    })
            
        except Exception as e:
            logger.error(f"Error adjusting at-risk goal {goal.title}: {e}")

    async def _update_performance_metrics_from_goals(self, completed_goals, in_progress_goals, overdue_goals):
        """Update system performance metrics based on goal progress."""
        try:
            # Check if performance tracker exists and is not None
            if not hasattr(self, 'performance_tracker') or self.performance_tracker is None:
                return

            # Calculate goal-related performance metrics
            total_goals = len(completed_goals) + len(in_progress_goals) + len(overdue_goals)
            
            if total_goals > 0:
                completion_rate = len(completed_goals) / total_goals
                overdue_rate = len(overdue_goals) / total_goals
                
                # Update performance tracker with goal-related metrics
                self.performance_tracker.record_metric(
                    name="goal_completion_rate",
                    value=completion_rate,
                    unit="percentage",
                    source="goal_review_system",
                    tags=["performance", "goals"]
                )
                
                self.performance_tracker.record_metric(
                    name="goal_overdue_rate",
                    value=overdue_rate,
                    unit="percentage",
                    source="goal_review_system",
                    tags=["performance", "goals"]
                )
                
                logger.info(f"Updated performance metrics: "
                           f"Completion rate: {completion_rate:.1%}, "
                           f"Overdue rate: {overdue_rate:.1%}")
        except Exception as e:
            logger.error(f"Error updating performance metrics from goals: {e}")

    async def _set_adaptive_goals(self):
        """Set new goals based on changing system needs."""
        # Analyze system state and performance to set new goals
        performance_summary = self.performance_tracker.get_performance_summary()
        
        # Example: If experiments are successful, set a goal to run more
        if performance_summary.get("improvement_success_rate", 0) > 0.8:
            # Check if this goal already exists
            if not any("Experiment" in goal.title for goal in self.self_goal_manager.goals.values()):
                self.self_goal_manager.create_goal(
                    title="Increase Experimentation Rate",
                    description="Since improvements are successful, increase experimentation to find more opportunities",
                    category="exploration",
                    priority=GoalPriority.MEDIUM,
                    target_date=datetime.now(timezone.utc) + timedelta(days=14),
                    metrics={"experimentation_rate_increase": 0.5},
                    dependencies=[]
                )

    async def _set_learning_based_goals(self, reflection_insights: Dict[str, Any]):
        """
        Set goals based on the AGI's learning and reflection insights.
        This method is called when the AGI processes new learning or completes self-reflection.
        """
        try:
            logger.info(f"Setting goals based on learning insights: {list(reflection_insights.keys())}")
            
            # Set goals based on capability gaps identified in reflection
            if 'capability_gaps' in reflection_insights:
                for gap in reflection_insights['capability_gaps']:
                    gap_name = gap.get('name', 'general_capability')
                    gap_severity = gap.get('severity', 'medium')
                    
                    # Map severity to priority
                    priority_map = {
                        'low': GoalPriority.LOW,
                        'medium': GoalPriority.MEDIUM,
                        'high': GoalPriority.HIGH,
                        'critical': GoalPriority.CRITICAL
                    }
                    priority = priority_map.get(gap_severity, GoalPriority.MEDIUM)
                    
                    self.self_goal_manager.create_goal(
                        title=f"Improve {gap_name.replace('_', ' ').title()} Capability",
                        description=f"Address identified capability gap in {gap_name}: {gap.get('description', 'Improve performance in this area')}",
                        category="capability",
                        priority=priority,
                        target_date=datetime.now(timezone.utc) + timedelta(days=gap.get('timeframe', 14)),
                        metrics={gap_name: gap.get('target_metric', 0.8)},
                        dependencies=[]
                    )
            
            # Set goals based on learning patterns
            if 'learning_patterns' in reflection_insights:
                patterns = reflection_insights['learning_patterns']
                
                # If learning rate is low, set a goal to improve it
                if patterns.get('learning_rate', 0) < 0.3:
                    self.self_goal_manager.create_goal(
                        title="Improve Learning Efficiency",
                        description="Based on reflection, learning efficiency needs improvement. Focus on better retention and understanding of new information.",
                        category="learning",
                        priority=GoalPriority.HIGH,
                        target_date=datetime.now(timezone.utc) + timedelta(days=21),
                        metrics={"learning_efficiency": 0.6},
                        dependencies=[]
                    )
                
                # If certain types of knowledge are difficult to acquire
                if patterns.get('knowledge_integration_issues', False):
                    self.self_goal_manager.create_goal(
                        title="Enhance Knowledge Integration",
                        description="Improve ability to connect new knowledge with existing knowledge and form comprehensive understanding.",
                        category="learning",
                        priority=GoalPriority.MEDIUM,
                        target_date=datetime.now(timezone.utc) + timedelta(days=30),
                        metrics={"knowledge_integration_rate": 0.75},
                        dependencies=[]
                    )
            
            # Set goals based on performance insights
            if 'performance_insights' in reflection_insights:
                perf_insights = reflection_insights['performance_insights']
                
                if perf_insights.get('decision_accuracy', 1.0) < 0.7:
                    self.self_goal_manager.create_goal(
                        title="Improve Decision Accuracy",
                        description="Analysis shows decision accuracy is below desired threshold. Focus on improving reasoning and evaluation processes.",
                        category="performance",
                        priority=GoalPriority.HIGH,
                        target_date=datetime.now(timezone.utc) + timedelta(days=14),
                        metrics={"decision_accuracy": 0.8},
                        dependencies=[]
                    )
            
            # Save goals after setting
            await self.self_goal_manager.save_goals()
            logger.info(f"Set {len(reflection_insights)} learning-based goals")
            
        except Exception as e:
            logger.error(f"Error setting learning-based goals: {e}")
            import traceback
            traceback.print_exc()

    async def _memorize_interaction(self, situation_prompt: str, decision: dict, action_output: Any):
        """Extracts and saves memories from an interaction."""
        interaction_summary = f"Situation: {situation_prompt}\nDecision: {decision}\nAction Output: {action_output}"
        try:
            memories_to_save = await self.memory_service.extract_memories(interaction_summary, "")
            if memories_to_save and memories_to_save.memories:
                await self.memory_service.save_memories(memories_to_save.memories)
                logger.info(
                    f"Saved {len(memories_to_save.memories)} new memories.")
                self.last_interaction_time = datetime.now(timezone.utc)
        except Exception as e:
            logger.error(f"Failed during memorization: {e}", exc_info=True)

    async def _handle_behavior_modifiers(self):
        if self.behavior_modifiers.get('suggest_break'):
            logger.info(
                "Mood suggests taking a break. Sleeping for a short while.")
            await asyncio.sleep(self.config.LOOP_SLEEP_DURATION * 2)
            self.behavior_modifiers = {}  # Reset modifiers

    async def _handle_curiosity(self):
        """Enhanced curiosity handling with async operations and better context."""
        config = Config()
        if random.random() < config.CURIOSITY_CHANCE:
            logger.info("Curiosity triggered. Generating new topics...")

            try:
                # Extract recent topics from memories
                recent_topics = []
                if self.shared_state.recent_memories:
                    # Last 10 memories
                    for memory in self.shared_state.recent_memories[:10]:
                        if isinstance(memory, dict):
                            content = memory.get('content', '')
                        else:
                            content = str(memory)

                        # Extract key topics from memory content
                        if content:
                            recent_topics.append(content[:100])  # Limit length

                if not recent_topics:
                    recent_topics = ["artificial intelligence",
                                     "machine learning", "consciousness", "creativity"]

                # Use enhanced async curiosity trigger
                curiosity_topics = await self.curiosity_trigger.get_curiosity_topics_llm(
                    recent_topics,
                    n=5,
                    lateralness=0.8  # High lateralness for creative exploration
                )

                self.shared_state.curiosity_topics = curiosity_topics
                logger.info(
                    f"Generated {len(curiosity_topics)} curiosity topics: {curiosity_topics}")

                # Personality-influenced invention generation: sometimes create invention ideas
                try:
                    if random.random() < 0.4:  # 40% of curiosity events produce invention ideas
                        ideas = self.personality.invent_ideas(
                            curiosity_topics, n=3)
                        self.shared_state.invention_ideas = ideas
                        # Persist a short log
                        for idea in ideas:
                            self.invention_history.append(
                                {"idea": idea, "ts": datetime.now(timezone.utc).isoformat()})
                        logger.info(
                            f"Personality generated {len(ideas)} invention ideas.")
                except Exception as e:
                    logger.warning(
                        f"Personality invention generation failed: {e}")

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
                            logger.info(
                                "Added curiosity exploration to knowledge base")
                    except Exception as e:
                        logger.warning(
                            f"Failed to process curiosity exploration: {e}")

            except Exception as e:
                logger.error(f"Curiosity handling failed: {e}", exc_info=True)
                self.shared_state.curiosity_topics = [
                    "explore new possibilities", "question assumptions"]
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
            logger.info("Getting relevant memories with contextual and emotional awareness.")
            
            # Get emotional context for enhanced retrieval
            mood_vector = self.shared_state.mood if hasattr(self.shared_state, 'mood') else None
            
            # Apply emotional memory bias if available
            emotional_memory_bias = getattr(self.shared_state, 'emotional_memory_bias', 0.0)
            
            # Use enhanced contextual retrieval
            memory_response = await self.memory_service.retrieve_contextually_relevant_memories(
                query=situation_prompt,
                emotion_context=mood_vector,
                time_range_minutes=1440,  # Last 24 hours
                top_k=10  # Get more memories for better context
            )
            
            # If memory_response is a tuple list, extract just the memories
            if isinstance(memory_response, list) and memory_response and isinstance(memory_response[0], tuple):
                # Extract Memory objects from (memory, similarity) tuples
                memories = [item[0] for item in memory_response]
            else:
                memories = memory_response
            
            # Apply emotional memory bias by re-ranking if bias is significant
            if abs(emotional_memory_bias) > 0.3:
                memories = await self._apply_emotional_memory_bias(memories, emotional_memory_bias)
                logger.info(f"Applied emotional memory bias: {emotional_memory_bias}")
            
            self.shared_state.recent_memories = memories
            logger.info("Got relevant memories with contextual and emotional awareness.")
        except Exception as e:
            logger.warning(f"Could not retrieve memories with context: {e}")
            # Fallback to basic retrieval
            try:
                memory_response = await self.memory_service.retrieve_relevant_memories(situation_prompt)
                if isinstance(memory_response, list) and memory_response and isinstance(memory_response[0], tuple):
                    # Extract Memory objects from (memory, similarity) tuples
                    memories = [item[0] for item in memory_response]
                else:
                    memories = memory_response
                self.shared_state.recent_memories = memories
                logger.info("Got relevant memories using fallback method.")
            except Exception as fallback_error:
                logger.warning(f"Fallback memory retrieval also failed: {fallback_error}")
                self.shared_state.recent_memories = []

    async def _make_decision(self, situation: dict):
        logger.info("Making enhanced decision with adaptive learning, self-improvement goals, and advanced analysis systems.")

        # Get available actions from the action manager's registry
        available_actions = self.action_manager.action_registry.get_action_definitions()

        # Apply adaptive learning to decision context
        decision_context = {
            'situation': situation,
            'mood': self.shared_state.mood,
            'memory': self.shared_state.recent_memories,
            'rag_context': getattr(self.shared_state, 'search_results', []),
            'self_improvement_goals': self._get_active_improvement_goals()
        }

        learning_adaptations = await self.learning_engine.apply_learning_to_decision(decision_context)

        # Apply enhanced analysis using the new systems
        # Analyze the situation for potential physics-related aspects
        physics_analysis = None
        try:
            # Check if the situation might involve physics
            situation_text = str(situation.get('prompt', '')) + ' ' + str(situation)
            if any(keyword in situation_text.lower() for keyword in 
                   ['physics', 'motion', 'force', 'energy', 'quantum', 'relativity', 'mechanics', 'thermodynamics']):
                physics_analysis = await self.physics_analysis_system.analyze_physics_problem(
                    problem_description=situation_text
                )
        except Exception as e:
            logger.warning(f"Physics analysis failed: {e}")

        # Apply lessons from previous failures to the current situation
        failure_lessons = None
        try:
            failure_lessons = await self.failure_learning_system.apply_lessons_to_task(
                task_description=situation.get('prompt', str(situation))
            )
        except Exception as e:
            logger.warning(f"Failure lesson application failed: {e}")

        # Let personality influence decision context
        try:
            persona_mods = self.personality.influence_decision(
                decision_context)
            if persona_mods:
                decision_context['persona_mods'] = persona_mods
                logger.info(
                    f"Applied persona modifiers to decision: {persona_mods}")
        except Exception as e:
            logger.debug(f"Personality influence failed: {e}")

        # Use advanced reasoning engine for enhanced decision making
        reasoning_context = {
            'situation': situation,
            'memory_context': self.shared_state.recent_memories,
            'goal_context': self._get_active_improvement_goals(),
            'physics_analysis': physics_analysis,
            'failure_lessons': failure_lessons
        }
        
        reasoning_result = await self.reasoning_engine.reason(**reasoning_context)
        
        # Convert reasoning result to decision format compatible with existing code
        decision = await self._convert_reasoning_to_decision(
            reasoning_result, 
            available_actions
        )

        # Apply learning adaptations to decision
        if learning_adaptations:
            # Adjust confidence based on learning
            confidence_modifier = learning_adaptations.get(
                'confidence_modifier', 1.0)
            original_confidence = decision.get('confidence', 0.5)
            decision['confidence'] = min(
                1.0, max(0.0, original_confidence * confidence_modifier))

            # Add learning context
            decision['learning_adaptations'] = learning_adaptations
            decision['mood_context'] = self.shared_state.mood
            # Last 5 memories
            decision['memory_context'] = self.shared_state.recent_memories[:5]

        # Add enhanced analysis results to decision
        if physics_analysis:
            decision['physics_analysis'] = physics_analysis
        if failure_lessons:
            decision['failure_lessons'] = failure_lessons

        # Log the enhanced decision
        await asyncio.to_thread(
            self.data_service.save_decision_log,
            self.shared_state.current_situation_id,
            decision['raw_response']
        )

        logger.info(
            f"Made enhanced decision with confidence {decision.get('confidence', 0.5):.2f}: {decision.get('action', 'unknown')}")
        return decision

    def _get_active_improvement_goals(self) -> List[Dict[str, Any]]:
        """Get active self-improvement goals to inform decision making."""
        try:
            # Get high-priority and in-progress goals
            active_goals = self.self_goal_manager.get_goals_by_status(GoalStatus.IN_PROGRESS)
            high_priority_goals = self.self_goal_manager.get_high_priority_goals()
            
            # Combine and remove duplicates by ID (since SelfGoal objects aren't hashable)
            seen_ids = set()
            all_relevant_goals = []
            for goal in active_goals + high_priority_goals:
                if goal.id not in seen_ids:
                    seen_ids.add(goal.id)
                    all_relevant_goals.append(goal)
            
            # Convert to dictionary format for decision making
            goal_dicts = []
            for goal in all_relevant_goals:
                goal_dicts.append({
                    'id': goal.id,
                    'title': goal.title,
                    'description': goal.description,
                    'category': goal.category,
                    'priority': goal.priority.value,
                    'progress': goal.current_progress,
                    'target_date': goal.target_date.isoformat(),
                    'metrics': goal.metrics
                })
            
            return goal_dicts
        except Exception as e:
            logger.error(f"Error getting active improvement goals: {e}")
            return []

    async def _convert_reasoning_to_decision(self, reasoning_result: Dict[str, Any], 
                                          available_actions: List[Dict]) -> Dict[str, Any]:
        """
        Convert reasoning engine output to decision format compatible with the system.
        
        Args:
            reasoning_result: Output from the reasoning engine
            available_actions: List of available actions to choose from
            
        Returns:
            Dictionary in the format expected by the rest of the system
        """
        try:
            # Check if reasoning_result is a string and handle appropriately
            if isinstance(reasoning_result, str):
                logger.warning(f"Received string instead of dict for reasoning_result: {reasoning_result[:100]}...")
                # Try to parse as JSON if it's a string
                try:
                    parsed_result = json.loads(reasoning_result)
                    if isinstance(parsed_result, dict):
                        reasoning_result = parsed_result
                    else:
                        # If it's not JSON-parseable or not a dict, create a default structure
                        reasoning_result = {
                            'synthesized_conclusion': reasoning_result,
                            'confidence_score': 0.5,
                            'reasoning_steps': [],
                            'reasoning_process': {},
                            'metacognitive_evaluation': {},
                            'reasoning_types_used': []
                        }
                except json.JSONDecodeError:
                    # If it can't be parsed as JSON, create a default structure
                    reasoning_result = {
                        'synthesized_conclusion': reasoning_result,
                        'confidence_score': 0.5,
                        'reasoning_steps': [],
                        'reasoning_process': {},
                        'metacognitive_evaluation': {},
                        'reasoning_types_used': []
                    }
            
            # Double-check that reasoning_result is now a dict before using .get() methods
            if not isinstance(reasoning_result, dict):
                logger.warning(f"Expected reasoning_result to be a dict after conversion, but got {type(reasoning_result)}. Creating default structure.")
                reasoning_result = {
                    'synthesized_conclusion': str(reasoning_result),
                    'confidence_score': 0.5,
                    'reasoning_steps': [],
                    'reasoning_process': {},
                    'metacognitive_evaluation': {},
                    'reasoning_types_used': []
                }

            # Extract key information from reasoning results with additional safety
            synthesized_conclusion = reasoning_result.get('synthesized_conclusion', '') if hasattr(reasoning_result, 'get') else str(reasoning_result)
            confidence_score = reasoning_result.get('confidence_score', 0.5) if hasattr(reasoning_result, 'get') else 0.5
            reasoning_steps = reasoning_result.get('reasoning_steps', []) if hasattr(reasoning_result, 'get') else []
            
            # Determine the best action based on reasoning conclusion
            best_action = await self._select_best_action_from_reasoning(
                synthesized_conclusion, 
                available_actions
            )
            
            # Create a plan based on reasoning
            plan = await self._create_plan_from_reasoning(
                reasoning_result, 
                best_action
            )
            
            # Create decision in compatible format
            decision = {
                'raw_response': json.dumps({
                    'analysis': reasoning_result.get('synthesized_conclusion', '') if hasattr(reasoning_result, 'get') else str(reasoning_result),
                    'reasoning_process': reasoning_result.get('reasoning_process', {}) if hasattr(reasoning_result, 'get') else {},
                    'metacognitive_evaluation': reasoning_result.get('metacognitive_evaluation', {}) if hasattr(reasoning_result, 'get') else {},
                    'plan': plan,
                    'action': best_action.get('name', 'log_message'),
                    'params': best_action.get('default_params', {}),
                    'confidence': confidence_score
                }, indent=2),
                'analysis': reasoning_result.get('synthesized_conclusion', '') if hasattr(reasoning_result, 'get') else str(reasoning_result),
                'reasoning_process': reasoning_result.get('reasoning_process', {}) if hasattr(reasoning_result, 'get') else {},
                'metacognitive_evaluation': reasoning_result.get('metacognitive_evaluation', {}) if hasattr(reasoning_result, 'get') else {},
                'plan': plan,
                'action': best_action.get('name', 'log_message'),
                'params': best_action.get('default_params', {}),
                'confidence': confidence_score,
                'reasoning_steps': reasoning_steps,
                'reasoning_types_used': reasoning_result.get('reasoning_types_used', []) if hasattr(reasoning_result, 'get') else []
            }
            
            logger.info(f"Converted reasoning to decision with action: {decision['action']} "
                       f"and confidence: {confidence_score:.2f}")
            
            return decision
        except Exception as e:
            logger.error(f"Error converting reasoning to decision: {e}")
            # Fallback: return a basic decision
            return {
                'raw_response': f'{{"action": "log_message", "params": {{"message": "Error in reasoning conversion: {e}"}}}}',
                'action': 'log_message',
                'params': {'message': f'Error in reasoning: {e}'},
                'confidence': 0.1,
                'error': str(e)
            }

    async def _select_best_action_from_reasoning(self, conclusion: str, available_actions: List[Dict]) -> Dict:
        """
        Select the best action based on the reasoning conclusion.
        
        Args:
            conclusion: The conclusion from the reasoning process
            available_actions: List of available actions
            
        Returns:
            Dictionary representing the best action
        """
        # In a sophisticated implementation, this would use AI to select the best action
        # For now, we'll implement a simple keyword-based matching approach
        
        # Check if the conclusion contains keywords that suggest a specific action
        conclusion_lower = conclusion.lower()
        
        for action in available_actions:
            action_name = action.get('name', '').lower()
            
            # If the conclusion explicitly mentions an action name, select it
            if action_name in conclusion_lower:
                logger.info(f"Selected action '{action_name}' based on keyword match")
                return action
        
        # If no keyword match, try to infer action based on common patterns
        if 'research' in conclusion_lower or 'search' in conclusion_lower or 'find' in conclusion_lower:
            for action in available_actions:
                if action.get('name', '').lower() in ['web_search', 'research_action']:
                    return action
        
        if 'code' in conclusion_lower or 'implement' in conclusion_lower or 'write' in conclusion_lower:
            for action in available_actions:
                if action.get('name', '').lower() in ['write_python_code', 'write_file', 'execute_python_file']:
                    return action
        
        # Default: return the first available action if it exists
        if available_actions:
            return available_actions[0]
        
        # Fallback: return a basic logging action
        return {
            'name': 'log_message',
            'description': 'Log a message to record reasoning outcome',
            'parameters': [{'name': 'message', 'type': 'string'}],
            'default_params': {'message': f'Reasoning completed: {conclusion[:200]}...'}
        }

    async def _create_plan_from_reasoning(self, reasoning_result: Dict[str, Any], 
                                        selected_action: Dict) -> List[Dict]:
        """
        Create an execution plan based on the reasoning results.
        
        Args:
            reasoning_result: The output from the reasoning engine
            selected_action: The action selected for execution
            
        Returns:
            List of steps in the execution plan
        """
        try:
            # Extract information from reasoning about what needs to be done
            synthesized_conclusion = reasoning_result.get('synthesized_conclusion', '')
            
            # Create a plan with the selected action as the main step
            plan = [{
                'action': selected_action.get('name', 'log_message'),
                'params': selected_action.get('default_params', {}),
                'rationale': f'Action selected based on reasoning: {synthesized_conclusion[:100]}...',
                'expected_outcome': 'Execution of the selected action',
                'success_criteria': 'Action completes without error'
            }]
            
            # Add follow-up steps based on metacognitive evaluation if available
            meta_eval = reasoning_result.get('metacognitive_evaluation', {})
            if meta_eval:
                # Consider adding a reflection step after action execution
                plan.append({
                    'action': 'log_message',
                    'params': {'message': f'Reasoning evaluation completed: {str(meta_eval)[:200]}...'},
                    'rationale': 'Metacognitive reflection on reasoning process',
                    'expected_outcome': 'System records evaluation of its reasoning',
                    'success_criteria': 'Evaluation is logged for future reference'
                })
            
            logger.info(f"Created plan with {len(plan)} steps")
            return plan
        except Exception as e:
            logger.error(f"Error creating plan from reasoning: {e}")
            # Fallback plan
            return [{
                'action': 'log_message',
                'params': {'message': f'Error creating plan: {e}'}
            }]

    async def invention_task(self):
        """Background task where the personality occasionally picks an idea to pursue and records a lightweight outcome."""
        while not self._shutdown.is_set():
            try:
                # Only attempt occasionally
                await asyncio.sleep(self.config.INVENTION_INTERVAL)
                ideas = getattr(self.shared_state,
                                'invention_ideas', None) or []
                if not ideas:
                    # generate seed ideas from recent memories or curiosity topics
                    topics = getattr(self.shared_state,
                                     'curiosity_topics', []) or []
                    ideas = self.personality.invent_ideas(topics, n=2)

                if ideas:
                    chosen = self.personality.pick_idea_to_pursue(ideas)
                    # Simulate a lightweight experiment/outcome
                    outcome = {"success": random.random(
                    ) < chosen.get('confidence', 0.5)}
                    self.personality.record_invention_outcome(
                        chosen.get('id'), outcome)
                    self.invention_history.append(
                        {"idea": chosen, "outcome": outcome, "ts": datetime.now(timezone.utc).isoformat()})
                    logger.info(
                        f"Personality pursued invention '{chosen.get('title')}' with outcome: {outcome}")
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
        success = not (isinstance(action_output, dict)
                       and action_output.get('error'))
        await self.learning_engine.record_decision_outcome(decision, action_output, success)

        logger.info("Memorizing interaction.")
        await self._memorize_interaction(situation_prompt, decision, action_output)
        logger.info("Memorized interaction.")

        # Clear action cache periodically
        self.action_manager.clear_cache()

        return action_output

    async def _update_mood_and_reflect(self, action_output: Any):
        logger.info("Updating mood and applying emotional influences.")
        old_mood = self.shared_state.mood.copy()
        self.emotional_intelligence.process_action_natural(str(action_output))
        self.shared_state.mood = self.emotional_intelligence.get_mood_vector()
        new_mood = self.shared_state.mood
        self.shared_state.mood_history.append(self.shared_state.mood)

        # Log the mood
        await asyncio.to_thread(self.data_service.save_mood_log, new_mood)

        logger.info("Updated mood.")

        # Get behavior modifications from emotional intelligence
        self.behavior_modifiers = self.emotional_intelligence.influence_behavior()
        if self.behavior_modifiers:
            logger.info(
                f"Generated behavior modifiers for next loop: {self.behavior_modifiers}")
            
            # Apply emotional influences to system parameters
            await self._apply_emotional_influences(self.behavior_modifiers)

        # If the action output contains a directive (like initiating an experiment),
        # merge it into the behavior modifiers for the next loop.
        if isinstance(action_output, dict) and 'action' in action_output:
            if action_output['action'] == 'initiate_experiment':
                logger.info(
                    f"Action output contains a directive to '{action_output['action']}'. Starting experiment.")
                self.experimentation_engine.start_experiment(action_output)

        # Check if reflection should be triggered based on mood changes
        mood_changed_for_better = self._did_mood_improve(old_mood, new_mood)

        if not mood_changed_for_better and random.random() < self.config.REFLECTION_CHANCE:
            logger.info("Mood has not improved. Initiating reflection.")
            # This is where you can trigger a reflection process
            # For now, we'll just log it.
            await self.reflection_module.reflect(self.shared_state)
        else:
            logger.info(
                "Mood improved or stayed the same, skipping reflection.")

    async def _apply_emotional_influences(self, behavior_modifiers: Dict[str, float]):
        """
        Apply emotional behavior modifiers to various system parameters.
        This function makes emotions truly influence system behavior.
        """
        # Adjust exploration vs exploitation based on emotions
        exploration_bias = behavior_modifiers.get("exploration_bias", 0.5)
        # Ensure exploration_bias is a float - convert from string if necessary
        if isinstance(exploration_bias, str):
            exploration_bias_map = {
                "very_low": 0.1,
                "low": 0.3,
                "medium": 0.5,
                "high": 0.7,
                "very_high": 0.9
            }
            exploration_bias = exploration_bias_map.get(exploration_bias.lower(), 0.5)
        
        if exploration_bias > 0.7:
            # High exploration bias - increase curiosity chance
            self.config.CURIOSITY_CHANCE = min(0.9, 0.4 + (exploration_bias * 0.5))
            logger.debug(f"Emotional influence: Increased curiosity chance to {self.config.CURIOSITY_CHANCE}")
        elif exploration_bias < 0.3:
            # Low exploration bias - decrease curiosity chance
            self.config.CURIOSITY_CHANCE = max(0.1, 0.4 - ((1 - exploration_bias) * 0.3))
            logger.debug(f"Emotional influence: Decreased curiosity chance to {self.config.CURIOSITY_CHANCE}")
        
        # Adjust risk tolerance
        risk_tolerance = behavior_modifiers.get("risk_tolerance", 0.5)
        # Ensure risk_tolerance is a float - convert from string if necessary
        if isinstance(risk_tolerance, str):
            # Map string values to numeric values
            risk_tolerance_map = {
                "very_low": 0.1,
                "low": 0.3,
                "medium": 0.5,
                "high": 0.7,
                "very_high": 0.9
            }
            risk_tolerance = risk_tolerance_map.get(risk_tolerance.lower(), 0.5)
        
        if risk_tolerance > 0.7:
            # High risk tolerance - allow more experimental actions
            logger.debug("Emotional influence: High risk tolerance - allowing more experimental decisions")
        elif risk_tolerance < 0.3:
            # Low risk tolerance - be more conservative in decision making
            logger.debug("Emotional influence: Low risk tolerance - being more conservative in decisions")
        
        # Adjust decision speed
        decision_speed = behavior_modifiers.get("decision_speed", 0.5)
        # Ensure decision_speed is a float - convert from string if necessary
        if isinstance(decision_speed, str):
            decision_speed_map = {
                "very_slow": 0.1,
                "slow": 0.3,
                "medium": 0.5,
                "fast": 0.7,
                "very_fast": 0.9
            }
            decision_speed = decision_speed_map.get(decision_speed.lower(), 0.5)
        
        if decision_speed > 0.7:
            # Fast decision making - potentially reduce thoroughness in reasoning
            logger.debug("Emotional influence: Fast decision making mode")
        elif decision_speed < 0.3:
            # Slow decision making - increase thoroughness in reasoning
            logger.debug("Emotional influence: Thorough decision making mode")
        
        # Influence memory recall based on mood
        emotional_memory_bias = behavior_modifiers.get("emotional_memory_bias", 0.0)
        # Ensure emotional_memory_bias is a float - convert from string if necessary
        if isinstance(emotional_memory_bias, str):
            emotional_bias_map = {
                "very_negative": -0.9,
                "negative": -0.6,
                "neutral": 0.0,
                "positive": 0.6,
                "very_positive": 0.9
            }
            emotional_memory_bias = emotional_bias_map.get(emotional_memory_bias.lower(), 0.0)
        
        if abs(emotional_memory_bias) > 0.3:
            # Apply emotional memory bias for next memory retrieval
            self.shared_state.emotional_memory_bias = emotional_memory_bias
            logger.debug(f"Emotional influence: Applied memory bias {emotional_memory_bias}")
        
        # Adjust attention span
        attention_span = behavior_modifiers.get("attention_span", 1.0)
        # Ensure attention_span is a float - convert from string if necessary
        if isinstance(attention_span, str):
            attention_span_map = {
                "very_short": 0.1,
                "short": 0.3,
                "medium": 0.5,
                "long": 0.8,
                "very_long": 0.9
            }
            attention_span = attention_span_map.get(attention_span.lower(), 1.0)
        
        if attention_span < 0.5:
            # Short attention span - increase likelihood of task switching
            logger.debug("Emotional influence: Short attention span detected")
        elif attention_span > 0.8:
            # Long attention span - maintain focus on current task
            logger.debug("Emotional influence: Long attention span detected, maintaining focus")
        
        # Apply creativity bias
        creativity_bias = behavior_modifiers.get("creativity_bias", 0.5)
        # Ensure creativity_bias is a float - convert from string if necessary
        if isinstance(creativity_bias, str):
            creativity_bias_map = {
                "very_low": 0.1,
                "low": 0.3,
                "medium": 0.5,
                "high": 0.7,
                "very_high": 0.9
            }
            creativity_bias = creativity_bias_map.get(creativity_bias.lower(), 0.5)
        
        if creativity_bias > 0.7:
            # High creativity - encourage novel solutions in reasoning
            logger.debug("Emotional influence: High creativity mode - encouraging novel solutions")
        elif creativity_bias < 0.3:
            # Low creativity - stick to conventional approaches
            logger.debug("Emotional influence: Low creativity mode - favoring conventional approaches")
        
        # Emotional regulation advice
        current_mood_state = self.emotional_intelligence.get_mood_vector()
        regulation_advice = self.emotional_intelligence.get_emotional_regulation_advice(current_mood_state)
        if regulation_advice:
            logger.info(f"Emotional regulation advice: {regulation_advice}")
        
        # Track emotional influence weight for meta-cognition
        emotional_decision_weight = behavior_modifiers.get("emotional_decision_weight", 0.0)
        self.shared_state.emotional_influence_weight = emotional_decision_weight
        logger.debug(f"Emotional decision weight: {emotional_decision_weight}")

    async def _apply_emotional_memory_bias(self, memories, emotional_bias: float) -> list:
        """
        Apply emotional memory bias to re-rank retrieved memories.
        
        Args:
            memories: List of retrieved memories
            emotional_bias: Bias value (-1 for negative memories, +1 for positive memories)
            
        Returns:
            Re-ranked list of memories based on emotional congruence
        """
        if not memories or not self.emotional_intelligence:
            return memories
        
        # Categorize memories based on emotional content
        positive_memories = []
        negative_memories = []
        neutral_memories = []
        
        for memory in memories:
            # Determine if memory has positive or negative emotional content
            # This is a simplified approach - in a real implementation, we'd analyze sentiment
            content = getattr(memory, 'content', str(memory)).lower()
            
            # Check for positive/negative keywords
            positive_keywords = ['success', 'achieved', 'happy', 'good', 'great', 'excellent', 'learned', 'progress', 'improved']
            negative_keywords = ['failed', 'error', 'bad', 'sad', 'worst', 'problem', 'issue', 'difficulty', 'frustrated']
            
            positive_score = sum(1 for word in positive_keywords if word in content)
            negative_score = sum(1 for word in negative_keywords if word in content)
            
            if positive_score > negative_score:
                positive_memories.append(memory)
            elif negative_score > positive_score:
                negative_memories.append(memory)
            else:
                neutral_memories.append(memory)
        
        # Re-rank based on emotional bias
        if emotional_bias > 0.3:  # Positive bias - favor positive memories
            # Prioritize positive memories, then neutral, then negative
            re_ranked = positive_memories + neutral_memories + negative_memories
        elif emotional_bias < -0.3:  # Negative bias - favor negative memories
            # Prioritize negative memories, then neutral, then positive
            re_ranked = negative_memories + neutral_memories + positive_memories
        else:  # Neutral bias - maintain original order
            re_ranked = memories
        
        # Ensure we return the same number of memories as input
        return re_ranked[:len(memories)]

    async def get_recent_events(self, time_limit_seconds: int = 3600) -> List[Event]:
        """
        Retrieves recent events from the database.
        """
        time_limit = datetime.now(timezone.utc) - timedelta(seconds=time_limit_seconds)
        stmt = select(Event).where(Event.timestamp >=
                                   time_limit).order_by(Event.timestamp.desc())

        loop = asyncio.get_running_loop()
        try:
            # Use a thread pool executor for the synchronous DB call
            result = await loop.run_in_executor(
                None,  # Uses the default executor
                lambda: self.session.exec(stmt).all()
            )
            return result
        except Exception as e:
            logger.error(
                f"Database query for recent events failed: {e}", exc_info=True)
            return []

    async def run_iteration(self):
        """Runs a single iteration of the AGI's thought process."""
        # Start timing for performance tracking
        iteration_start_time = time.time()
        
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
            logger.info(
                f"Continuing with task: '{self.current_task_prompt}'. {len(self.current_plan)} steps remaining.")
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
                logger.info(
                    f"Found and stored a multi-step plan with {len(self.current_plan)} steps remaining.")
            else:
                # If the plan is done or was a single step, clear it.
                self.current_plan = []
                self.current_task_prompt = None
                if plan:
                    logger.info(
                        "Plan found, but only had one step which was already executed.")

        except json.JSONDecodeError:
            logger.warning("Could not parse plan from decision response.")
            self.current_plan = []
            self.current_task_prompt = None

        # Update mood and reflect
        await self._update_mood_and_reflect(action_output)
        
        # Record iteration performance metrics
        iteration_duration = time.time() - iteration_start_time
        self.performance_tracker.record_metric(
            name="iteration_duration",
            value=iteration_duration,
            unit="seconds",
            source="agi_system",
            tags=["performance", "iteration"]
        )
        self.performance_tracker.increment_iteration_count()

    async def run_autonomous_loop(self):
        """The main autonomous loop of the AGI."""
        logger.info("Starting autonomous loop...")

        # Start background tasks
        self.background_tasks.append(
            asyncio.create_task(self.data_collection_task()))
        self.background_tasks.append(
            asyncio.create_task(self.event_detection_task()))
        self.background_tasks.append(
            asyncio.create_task(self.knowledge_compression_task()))
        self.background_tasks.append(
            asyncio.create_task(self.memory_consolidation_task()))

        # Start autonomous blog scheduler maintenance task
        if self.blog_scheduler:
            self.background_tasks.append(asyncio.create_task(
                self.autonomous_blog_maintenance_task()))

        # Start Mad Scientist System maintenance task
        if hasattr(self, 'mad_scientist_system') and self.mad_scientist_system:
            self.background_tasks.append(asyncio.create_task(
                self.mad_scientist_maintenance_task()))

        # Start Snake Agent if enabled
        if self.config.SNAKE_AGENT_ENABLED and self.snake_agent:
            await self.start_snake_agent()

        # Start Conversational AI if enabled
        if self.config.CONVERSATIONAL_AI_ENABLED and self.conversational_ai:
            await self.start_conversational_ai()

        while not self._shutdown.is_set():
            try:
                if self.experimentation_engine.active_experiment:
                    await self.experimentation_engine.run_experiment_step()
                else:
                    await self.run_iteration()

                logger.info(
                    f"End of loop iteration. Sleeping for {self.config.LOOP_SLEEP_DURATION} seconds.")
                await asyncio.sleep(self.config.LOOP_SLEEP_DURATION)
            except Exception as e:
                logger.critical(
                    f"Critical error in autonomous loop: {e}", exc_info=True)
                # Longer sleep after critical error
                await asyncio.sleep(self.config.LOOP_SLEEP_DURATION * 5)

        logger.info("Autonomous loop has been stopped.")

    async def mad_scientist_maintenance_task(self):
        """Background task to run mad scientist system activities."""
        while not self._shutdown.is_set():
            try:
                logger.info("Running mad scientist maintenance cycle...")
                
                # Occasionally run a mad scientist cycle to explore impossible projects
                if random.random() < 0.3:  # 30% chance to run a mad scientist cycle
                    # Select a random domain from recent activities
                    recent_domains = ["computer science", "physics", "mathematics", "ai research", "consciousness"]
                    if self.shared_state.recent_memories:
                        # Extract domains from recent memories
                        for memory in self.shared_state.recent_memories[:5]:
                            content = getattr(memory, 'content', str(memory))
                            if 'physics' in content.lower():
                                recent_domains.append('physics')
                            elif 'computer' in content.lower() or 'code' in content.lower():
                                recent_domains.append('computer science')
                            elif 'math' in content.lower():
                                recent_domains.append('mathematics')
                            elif 'ai' in content.lower() or 'intelligence' in content.lower():
                                recent_domains.append('ai research')
                    
                    # Select a random domain to focus on
                    selected_domain = random.choice(list(set(recent_domains)))
                    
                    logger.info(f"Starting mad scientist cycle in domain: {selected_domain}")
                    result = await self.mad_scientist_system.run_mad_scientist_cycle(selected_domain)
                    logger.info(f"Mad scientist cycle completed with {len(result.get('publications_created', []))} publications")
                
                # Periodically retry any pending publications
                retries = await self.mad_scientist_system.retry_publication_of_innovations()
                logger.info(f"Retried publication for {retries} innovations")

            except Exception as e:
                logger.error(
                    f"Error in mad scientist maintenance: {e}", exc_info=True)

            try:
                # Run maintenance every 2 hours
                await asyncio.sleep(7200)
            except asyncio.CancelledError:
                break
        logger.info("Mad scientist maintenance task shut down.")

    async def run_single_task(self, prompt: str):
        """Runs the AGI for a single task specified by the prompt."""
        logger.info(f"--- Running Single Task: {prompt} ---")
        self.shared_state.current_task = prompt

        max_iterations = Config().MAX_ITERATIONS
        for i in range(max_iterations):
            if self._shutdown.is_set():
                logger.info("Task appears to be complete. Ending run.")
                break
            await self.run_iteration()

            if not self.current_plan and not self.current_task_prompt:
                logger.info("Task appears to be complete. Ending run.")
                break

            await asyncio.sleep(1)  # Give a moment for async operations
        else:
            logger.warning(
                f"Task exceeded {max_iterations} iterations. Ending run.")

        logger.info("--- Single Task Finished ---")

    def _did_mood_improve(self, old_mood: Dict[str, float], new_mood: Dict[str, float]) -> bool:
        """
        Checks if the overall mood has improved based on positive and negative mood components.
        """
        config = Config()  # Get the config singleton instance
        old_score = sum(old_mood.get(m, 0) for m in config.POSITIVE_MOODS) - \
            sum(old_mood.get(m, 0) for m in config.NEGATIVE_MOODS)
        new_score = sum(new_mood.get(m, 0) for m in config.POSITIVE_MOODS) - \
            sum(new_mood.get(m, 0) for m in config.NEGATIVE_MOODS)

        logger.info(
            f"Mood score changed from {old_score:.2f} to {new_score:.2f}")
        return new_score > old_score

    async def data_collection_task(self):
        """Background task to fetch articles from RSS feeds every hour."""
        while not self._shutdown.is_set():
            try:
                logger.info("Fetching feeds...")
                num_saved = await asyncio.to_thread(self.data_service.fetch_and_save_articles)
                if num_saved > 0:
                    logger.info(
                        f"Feeds fetched and {num_saved} new articles saved.")
                else:
                    logger.info("No new articles found.")
            except Exception as e:
                logger.error(f"Error in data collection: {e}")

            try:
                # Use config value
                await asyncio.sleep(self.config.DATA_COLLECTION_INTERVAL)
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
                # Use config value
                await asyncio.sleep(self.config.EVENT_DETECTION_INTERVAL)
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
                await asyncio.sleep(86600)
            except asyncio.CancelledError:
                break
        logger.info("Knowledge compression task shut down.")

    async def memory_consolidation_task(self):
        """Periodically consolidates memories to optimize retrieval and relevance."""
        while not self._shutdown.is_set():
            try:
                logger.info("Starting memory consolidation...")
                
                # Perform enhanced memory consolidation
                consolidation_result = await self.memory_service.consolidate_old_memories()
                logger.info(
                    f"Memory consolidation finished. Report: {consolidation_result}")
                
                # Clean up expired memories from short-term and working memory
                if hasattr(self.memory_service, 'cleanup_expired_memories'):
                    await self.memory_service.cleanup_expired_memories()
                    logger.info("Expired memories cleaned up from short-term and working memory")
                    
            except Exception as e:
                logger.error(
                    f"Error during memory consolidation: {e}", exc_info=True)

            try:
                # Update sleep interval to be more frequent for better memory management
                await asyncio.sleep(10800)  # 3 hours instead of 6
            except asyncio.CancelledError:
                break
        logger.info("Memory consolidation task shut down.")

    async def autonomous_blog_maintenance_task(self):
        """Background task to maintain the autonomous blog scheduler."""
        while not self._shutdown.is_set():
            try:
                if self.blog_scheduler:
                    # Clear old events periodically
                    self.blog_scheduler.clear_old_events(hours=48)

                    # Log status periodically
                    status = self.blog_scheduler.get_status()
                    logger.info(
                        f"Blog scheduler status: {status['pending_events']} pending, {status['recent_posts']} recent posts")

            except Exception as e:
                logger.error(
                    f"Error in autonomous blog maintenance: {e}", exc_info=True)

            try:
                # Run maintenance every 6 hours
                await asyncio.sleep(21600)
            except asyncio.CancelledError:
                break
        logger.info("Autonomous blog maintenance task shut down.")

    def get_blog_scheduler_status(self) -> Dict[str, Any]:
        """Get autonomous blog scheduler status."""
        if not self.blog_scheduler:
            return {"enabled": False, "status": "not_available"}

        return {
            "enabled": True,
            "status": "active",
            **self.blog_scheduler.get_status()
        }

    async def trigger_learning_based_goal_setting(self, learning_event: Dict[str, Any]):
        """
        Trigger goal setting based on learning events.
        This method allows the AGI to set goals when it learns something significant.
        """
        try:
            logger.info(f"Triggering goal setting from learning event: {learning_event.get('event_type', 'unknown')}")
            
            # Create reflection insights from the learning event
            insights = {
                "capability_gaps": learning_event.get("capability_gaps", []),
                "learning_patterns": learning_event.get("learning_patterns", {}),
                "performance_insights": learning_event.get("performance_insights", {}),
                "learning_event_type": learning_event.get("event_type", "general"),
                "learning_content": learning_event.get("content", ""),
                "timestamp": learning_event.get("timestamp", None)
            }
            
            # Set goals based on the learning
            await self._set_learning_based_goals(insights)
            
            logger.info("Successfully triggered learning-based goal setting")
            
        except Exception as e:
            logger.error(f"Error in learning-based goal triggering: {e}")
            import traceback
            traceback.print_exc()


