#!/usr/bin/env python3
"""
AGI System - Main Integration Module
This module integrates all the specialized modules into a cohesive AGI system.
"""

import os
import sys
import time
import json
import logging
import argparse
from typing import Dict, List, Any, Optional
import threading
import queue
import importlib.util

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("agi_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AGI")

# Create a dedicated logger for interactions
interaction_logger = logging.getLogger("Interactions")
interaction_logger.setLevel(logging.INFO)
# Create a handler for the interactions logger - we will write to a JSONL file
interaction_handler = logging.FileHandler("interactions.jsonl")
interaction_handler.setFormatter(logging.Formatter('%(message)s'))
interaction_logger.addHandler(interaction_handler)

# Add modules directory to path
MODULES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules")
sys.path.append(MODULES_DIR)

# Import modules
try:
    # Add agent_self_reflection directory to path
    agent_self_reflection_dir = os.path.join(MODULES_DIR, "agent_self_reflection")
    sys.path.append(agent_self_reflection_dir)
    
    # Agent Self Reflection - Using absolute imports with importlib
    agent_reflection_path = os.path.join(agent_self_reflection_dir, "main.py")
    spec = importlib.util.spec_from_file_location("agent_reflection", agent_reflection_path)
    agent_reflection = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_reflection)
    reflect_on_task = agent_reflection.reflect_on_task
    
    # Import agent_self_reflection/llm.py for run_langchain_reflection with a unique name
    agent_self_reflection_llm_path = os.path.join(agent_self_reflection_dir, "llm.py")
    spec = importlib.util.spec_from_file_location("agent_self_reflection_llm", agent_self_reflection_llm_path)
    agent_self_reflection_llm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_self_reflection_llm)
    run_langchain_reflection = agent_self_reflection_llm.run_langchain_reflection
    
    # Add knowledge_compression directory to path
    knowledge_compression_dir = os.path.join(MODULES_DIR, "knowledge_compression")
    sys.path.append(knowledge_compression_dir)
    
    # Knowledge Compression - Using importlib to avoid naming conflicts
    knowledge_compression_path = os.path.join(knowledge_compression_dir, "main.py")
    spec = importlib.util.spec_from_file_location("knowledge_compression", knowledge_compression_path)
    knowledge_compression = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(knowledge_compression)
    compress_knowledge = knowledge_compression.compress_knowledge
    
    # Episodic Memory
    sys.path.append(os.path.join(MODULES_DIR, "episodic_memory"))
    from memory import app as memory_app
    import uvicorn
    
    # Emotional Intelligence
    sys.path.append(os.path.join(MODULES_DIR, "emotional_intellegence"))
    from emotional_intellegence import EmotionalIntelligence
    
    # Decision Engine
    sys.path.append(os.path.join(MODULES_DIR, "decision_engine"))
    from planner import GoalPlanner, plan_from_context
    from decision_maker import goal_driven_decision_maker_loop
    
    # Curiosity Trigger
    sys.path.append(os.path.join(MODULES_DIR, "curiosity_trigger"))
    from curiosity_trigger import CuriosityTrigger
    
    # Event Detection
    sys.path.append(os.path.join(MODULES_DIR, "event_detection"))
    from event_detector import process_data_for_events, load_models
    
    # Create a wrapper class for event detection functionality
    class EventDetector:
        def __init__(self):
            load_models()
            
        def process(self, texts: List[str]) -> List[Dict]:
            """Process a list of texts to detect events"""
            result = process_data_for_events(texts)
            return result.get("events", [])
    
    # Information Processing - YouTube Transcription
    youtube_transcription_dir = os.path.join(MODULES_DIR, "information_processing/youtube_transcription")
    sys.path.append(youtube_transcription_dir)
    
    # Use importlib to import module with hyphen in filename
    youtube_transcription_path = os.path.join(youtube_transcription_dir, "youtube-transcription.py")
    spec = importlib.util.spec_from_file_location("youtube_transcription_module", youtube_transcription_path)
    youtube_transcription_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(youtube_transcription_module)
    transcribe_youtube_video = youtube_transcription_module.transcribe_youtube_video
    
    # Trend Analysis
    sys.path.append(os.path.join(MODULES_DIR, "information_processing/trend_analysis"))
    from trend_engine import analyze_trends, fetch_feeds, setup_db
    
    # AGI Experimentation
    sys.path.append(os.path.join(MODULES_DIR, "agi_experimentation"))
    # Import agi_experimentation_engine from the correct module
    from modules.agi_experimentation.llm import agi_experimentation_engine
    
    # Situation Generator
    situation_generator_dir = os.path.join(MODULES_DIR, "situation_generator")
    sys.path.append(situation_generator_dir)
    # Import from the renamed file
    situation_generator_path = os.path.join(situation_generator_dir, "situation_generator.py")
    spec = importlib.util.spec_from_file_location("situation_generator_module", situation_generator_path)
    situation_generator_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(situation_generator_module)
    SituationGenerator = situation_generator_module.SituationGenerator
    
    logger.info("All modules imported successfully")
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    sys.exit(1)

class AGISystem:
    """Main AGI system that integrates all modules"""
    
    def __init__(self):
        self.emotional_intelligence = EmotionalIntelligence()
        self.goal_planner = GoalPlanner()
        try:
            self.event_detector = EventDetector()
        except Exception as e:
            logger.error(f"Failed to initialize EventDetector: {e}. Event detection will be disabled.")
            self.event_detector = None
        self.memory_queue = queue.Queue()
        self.task_queue = queue.Queue()
        self.situation_queue = queue.Queue()
        self.running = False
        self.memory_server_thread = None
        self.situation_generator = SituationGenerator()
        self.autonomous_mode = False
        self.autonomous_thread = None
        self.health_monitor_thread = None
        self.last_situation = None
        self.last_response = None
        
        # Setup trend analysis database
        setup_db()
        
        logger.info("AGI System initialized")
    
    def start_memory_server(self):
        """Start the memory server in a separate thread"""
        def run_server():
            uvicorn.run(memory_app, host="127.0.0.1", port=8000)
        
        self.memory_server_thread = threading.Thread(target=run_server)
        self.memory_server_thread.daemon = True
        self.memory_server_thread.start()
        logger.info("Memory server started on port 8000")
    
    def process_memory(self, text: str):
        """Process and store a memory"""
        import requests
        try:
            response = requests.post(
                "http://127.0.0.1:8000/extract_memories/",
                json={"user_input": text, "ai_output": ""}
            )
            if response.status_code == 200:
                memories = response.json().get("memories", [])
                if memories:
                    logging.info(f"Storing {len(memories)} memories: {memories}")
                    requests.post(
                        "http://127.0.0.1:8000/save_memories/",
                        json={"memories": memories}
                    )
                    logger.info(f"Stored {len(memories)} memories")
                    return memories
            return []
        except Exception as e:
            logger.error(f"Error processing memory: {e}")
            return []
    
    def retrieve_relevant_memories(self, query: str, top_n: int = 5):
        """Retrieve memories relevant to the given query"""
        import requests
        try:
            response = requests.post(
                "http://127.0.0.1:8000/get_relevant_memories/",
                json={"query_text": query, "top_n": top_n}
            )
            if response.status_code == 200:
                return response.json().get("relevant_memories", [])
            return []
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []
    
    def create_goal_from_context(self, context: str):
        """Create a new goal based on the given context"""
        goal_id = plan_from_context(context)
        logger.info(f"Created new goal with ID: {goal_id}")
        return goal_id
    
    def trigger_curiosity(self, recent_topics: List[str], lateralness: float = 0.7):
        """Trigger curiosity based on recent topics"""
        article, prompt = CuriosityTrigger.trigger(recent_topics, lateralness)
        logger.info(f"Curiosity triggered: {prompt}")
        return article, prompt
    
    def reflect_on_action(self, task: str, outcome: str):
        """Reflect on a completed task"""
        reflection = reflect_on_task(task, outcome)
        logger.info(f"Reflection generated for task: {task}")
        return reflection
    
    def reflect_with_langchain(self, task: str, outcome: str = None):
        """Reflect on a completed task using LangChain"""
        reflection = run_langchain_reflection(task, outcome)
        logger.info(f"LangChain reflection generated for task: {task}")
        return reflection
    
    def compress_knowledge_logs(self, logs: List[Dict]):
        """Compress knowledge from logs"""
        compressed = compress_knowledge(logs)
        logger.info("Knowledge compressed")
        return compressed
    
    def detect_events(self, texts: List[str]):
        """Detect events from a list of texts"""
        if self.event_detector:
            return self.event_detector.process(texts)
        else:
            logger.warning("Event detector is not available. Skipping event detection.")
            return []
    
    def transcribe_youtube(self, url: str):
        """Transcribe a YouTube video"""
        transcript = transcribe_youtube_video(url)
        logger.info(f"Transcribed YouTube video: {url}")
        return transcript
    
    def analyze_trends(self, hours: int = 24):
        """Analyze trends from the last n hours"""
        analyze_trends(hours)
        logger.info(f"Analyzed trends from the last {hours} hours")
    
    def run_experiment(self, idea: str):
        """Run an experiment based on the given idea"""
        result = agi_experimentation_engine(
            idea,
            use_chain_of_thought=True,
            online_validation=False,
            verbose=True
        )
        logger.info(f"Experiment run: {idea}")
        return result
    
    def process_input(self, user_input: str):
        """Process user input and determine appropriate actions"""
        # Update emotional state based on input
        self.emotional_intelligence.process_action_natural(user_input)
        
        # Store as memory
        memories = self.process_memory(user_input)
        
        # Get relevant memories for context
        relevant_memories = self.retrieve_relevant_memories(user_input)
        
        # Determine dominant mood and behavior suggestions
        mood = self.emotional_intelligence.get_dominant_mood()
        behavior = self.emotional_intelligence.influence_behavior()
        
        # If curiosity is triggered, get related information
        if behavior.get("curiosity_trigger", False):
            recent_topics = [m["text"] for m in relevant_memories]
            article, prompt = self.trigger_curiosity(recent_topics)
        
        # Create a goal if appropriate
        if "create goal" in user_input.lower():
            goal_id = self.create_goal_from_context(user_input)
            return f"Created new goal with ID: {goal_id}"
        
        # Run an experiment if requested
        if "experiment" in user_input.lower():
            result = self.run_experiment(user_input)
            return f"Experiment result: {result}"
        
        # Default response
        return f"Processed input. Current mood: {mood}, Memories stored: {len(memories)}"
    
    def process_situation(self, situation: Dict[str, Any]):
        """Process a situation generated by the SituationGenerator"""
        try:
            # Log the start of processing
            logger.debug(f"Starting to process situation: {situation}")
            
            # Store the situation for reference
            self.last_situation = situation
            
            # Simple processing for now - just log the situation
            logger.info(f"Processing situation: {situation}")
            
            # A more advanced implementation would involve the decision engine,
            # memory retrieval, and other modules to generate a response.
            
            # For now, let's create a simple response and log it.
            response = f"I have received the situation: {situation['type']}. I am thinking about it."
            self.last_response = response
            
            # Log the interaction with detailed error handling
            try:
                # First, verify that the situation is serializable
                logger.debug("Attempting to serialize situation for logging")
                interaction_data = {
                    "timestamp": time.time(),
                    "situation": situation,
                    "response": response
                }
                
                # Try to serialize the data first to catch any JSON errors
                json_str = json.dumps(interaction_data)
                logger.debug(f"Successfully serialized interaction data: {len(json_str)} bytes")
                
                # Now log it to the interactions.jsonl file
                logger.debug("Writing to interactions.jsonl")
                interaction_logger.info(json_str)
                logger.debug("Successfully wrote to interactions.jsonl")
                
                # Also store this interaction in memory
                try:
                    self.process_memory(f"Situation: {situation['type']} - Response: {response}")
                except Exception as mem_err:
                    logger.error(f"Error storing situation in memory: {mem_err}", exc_info=True)
                
            except TypeError as e:
                logger.error(f"JSON serialization error: {e}. Situation may contain non-serializable objects.", exc_info=True)
                # Try with a simplified version
                try:
                    simplified_situation = {
                        "type": situation.get("type", "unknown"),
                        "prompt": str(situation.get("prompt", "")),
                        "context": str(situation.get("context", ""))
                    }
                    logger.info(f"Using simplified situation for logging: {simplified_situation}")
                    interaction_logger.info(json.dumps({
                        "timestamp": time.time(),
                        "situation": simplified_situation,
                        "response": response
                    }))
                except Exception as inner_e:
                    logger.error(f"Even simplified JSON serialization failed: {inner_e}", exc_info=True)
            except Exception as e:
                logger.error(f"Error logging interaction: {e}", exc_info=True)
            
            logger.info(f"Generated response: {response}")
            
            # Reflect on the action with error handling
            try:
                reflection = self.reflect_on_action(
                    task=f"Processed situation of type: {situation['type']}",
                    outcome=response
                )
                logger.debug(f"Reflection generated: {reflection[:100]}...")
            except Exception as e:
                logger.error(f"Error during reflection: {e}", exc_info=True)
            
            return response
            
        except KeyError as e:
            error_msg = f"Missing key in situation: {e}"
            logger.error(error_msg, exc_info=True)
            return f"Error processing situation: {error_msg}"
        except Exception as e:
            error_msg = f"Unexpected error processing situation: {e}"
            logger.error(error_msg, exc_info=True)
            return f"Error processing situation: {error_msg}"
    
    def autonomous_thread_function(self):
        """A simpler, more robust thread function for autonomous operation."""
        logger.info("Starting autonomous operation with a simplified loop.")
        
        # Keep track of the last successful situation processing
        last_success_time = time.time()
        situation_count = 0
        error_count = 0
        consecutive_errors = 0
        
        while self.running and self.autonomous_mode:
            try:
                # Log heartbeat to show the thread is still alive
                current_time = time.time()
                elapsed_since_success = current_time - last_success_time
                
                if elapsed_since_success > 300:  # 5 minutes
                    logger.warning(f"No successful situation processing for {elapsed_since_success:.1f} seconds")
                
                # 1. Get a new situation
                logger.debug("Getting new situation from generator...")
                situation = self.situation_generator.get_situation(timeout=10)
                
                if situation:
                    situation_count += 1
                    logger.info(f"SITUATION ({situation_count}): {situation}")
                    
                    # Check if situation generator queue is growing too large
                    queue_size = self.situation_generator.situation_queue.qsize()
                    if queue_size > 10:
                        logger.warning(f"Situation queue is growing large: {queue_size} items")
                    
                    # 2. Process the situation with timing
                    start_time = time.time()
                    logger.debug(f"Processing situation at {start_time}")
                    
                    try:
                        response = self.process_situation(situation)
                        process_time = time.time() - start_time
                        logger.debug(f"Situation processed in {process_time:.2f} seconds")
                        
                        # Reset error counters on success
                        last_success_time = time.time()
                        consecutive_errors = 0
                        
                        # Verify that the response was properly logged to interactions.jsonl
                        try:
                            # Check if interactions.jsonl exists and was modified recently
                            if os.path.exists("interactions.jsonl"):
                                file_mod_time = os.path.getmtime("interactions.jsonl")
                                if time.time() - file_mod_time > 10:  # If file wasn't modified in the last 10 seconds
                                    logger.warning("interactions.jsonl wasn't updated recently despite processing a situation")
                            else:
                                logger.error("interactions.jsonl doesn't exist!")
                        except Exception as e:
                            logger.error(f"Error checking interactions.jsonl: {e}")
                            
                    except Exception as e:
                        error_count += 1
                        consecutive_errors += 1
                        logger.error(f"Error processing situation: {e}", exc_info=True)
                        
                        if consecutive_errors >= 3:
                            logger.critical(f"Too many consecutive errors ({consecutive_errors}), may need intervention")
                            # You could implement some recovery mechanism here
                else:
                    # If no situation is available, wait a bit
                    logger.debug("No situation available, sleeping for a moment.")
                    time.sleep(10)

                # Add a small sleep to prevent CPU hogging
                time.sleep(0.1)
                
            except Exception as e:
                error_count += 1
                consecutive_errors += 1
                logger.error(f"Error in autonomous thread: {e}", exc_info=True)
                
                if consecutive_errors >= 5:
                    logger.critical("Too many consecutive errors in autonomous thread, restarting situation generator")
                    try:
                        # Try to restart the situation generator
                        self.situation_generator.stop()
                        time.sleep(2)
                        self.situation_generator.start()
                        logger.info("Situation generator restarted")
                        consecutive_errors = 0
                    except Exception as restart_err:
                        logger.critical(f"Failed to restart situation generator: {restart_err}")
                
                time.sleep(10)
                
            # Periodically log statistics
            if situation_count > 0 and situation_count % 10 == 0:
                logger.info(f"Autonomous thread statistics: {situation_count} situations processed, {error_count} errors")
        
        logger.info("Autonomous thread function exiting")
    
    def start_autonomous_mode(self):
        """Start autonomous mode"""
        if self.autonomous_thread and self.autonomous_thread.is_alive():
            logger.warning("Autonomous mode thread is already running.")
            return
        
        self.autonomous_mode = True
        
        # Start the situation generator
        self.situation_generator.start()
        
        # Start the autonomous thread
        self.autonomous_thread = threading.Thread(target=self.autonomous_thread_function)
        self.autonomous_thread.daemon = True
        self.autonomous_thread.start()
        
        # Start the health monitor thread
        self.health_monitor_thread = threading.Thread(target=self.monitor_system_health)
        self.health_monitor_thread.daemon = True
        self.health_monitor_thread.start()
        logger.info("Health monitor thread started")
        
        logger.info("Autonomous mode started")
    
    def stop_autonomous_mode(self):
        """Stop autonomous mode"""
        if not self.autonomous_mode:
            logger.warning("Autonomous mode not running")
            return
        
        # Stop the autonomous thread
        self.autonomous_mode = False
        if self.autonomous_thread:
            self.autonomous_thread.join(timeout=5)
            logger.debug("Autonomous thread joined")
        
        # Stop the health monitor thread
        if hasattr(self, 'health_monitor_thread') and self.health_monitor_thread:
            self.health_monitor_thread.join(timeout=5)
            logger.debug("Health monitor thread joined")
        
        # Stop the situation generator
        self.situation_generator.stop()
        
        logger.info("Autonomous mode stopped")
    
    def start(self):
        """Start the AGI system"""
        self.running = True
        self.start_memory_server()
        
        # Wait for memory server to start
        time.sleep(2)
        
        logger.info("AGI System started")

        if self.autonomous_mode:
            self.start_autonomous_mode()
        
        # Unified main loop
        try:
            while self.running:
                if not self.autonomous_mode:
                    # Interactive mode
                    try:
                        user_input = input("> ")
                        if user_input.lower() == "exit":
                            self.running = False
                        elif user_input.lower() == "auto":
                            if not self.autonomous_mode:
                                self.start_autonomous_mode()
                                print("Autonomous mode started")
                        elif user_input.lower() == "manual":
                            if self.autonomous_mode:
                                self.stop_autonomous_mode()
                                print("Autonomous mode stopped")
                        else:
                            response = self.process_input(user_input)
                            print(response)
                    except (KeyboardInterrupt, EOFError):
                        self.running = False
                        logger.info("Interactive mode stopped.")
                    except Exception as e:
                        logger.error(f"Error processing input: {e}")
                else:
                    # Autonomous mode - main thread just waits
                    time.sleep(1)
        except KeyboardInterrupt:
            self.running = False
            logger.info("AGI system shutting down.")
        
        # Stop autonomous mode if it's running at the end
        if self.autonomous_mode:
            self.stop_autonomous_mode()
        
        logger.info("AGI System stopped")
    
    def monitor_system_health(self):
        """Monitor the health of the AGI system and log metrics."""
        logger.info("Starting system health monitor")
        
        health_check_interval = 60  # seconds
        last_check_time = time.time()
        
        try:
            import psutil
            has_psutil = True
        except ImportError:
            logger.warning("psutil not available, some health metrics will be limited")
            has_psutil = False
            
        while self.running and self.autonomous_mode:
            try:
                current_time = time.time()
                
                # Only check periodically
                if current_time - last_check_time < health_check_interval:
                    time.sleep(1)
                    continue
                    
                last_check_time = current_time
                
                # Log basic process information
                if has_psutil:
                    process = psutil.Process()
                    
                    # Memory usage
                    memory_info = process.memory_info()
                    logger.info(f"Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")
                    
                    # CPU usage
                    cpu_percent = process.cpu_percent(interval=1.0)
                    logger.info(f"CPU usage: {cpu_percent:.1f}%")
                    
                    # Check for zombie child processes
                    children = process.children(recursive=True)
                    if children:
                        logger.info(f"Child processes: {len(children)}")
                        for child in children:
                            if child.status() == psutil.STATUS_ZOMBIE:
                                logger.warning(f"Zombie child process detected: {child.pid}")
                
                # Check file sizes and growth
                log_files = ["agi_system.log", "situation_generator.log", "interactions.jsonl", "autonomous_agi.log"]
                for log_file in log_files:
                    if os.path.exists(log_file):
                        size_mb = os.path.getsize(log_file) / (1024 * 1024)
                        logger.debug(f"File size of {log_file}: {size_mb:.2f} MB")
                        
                        # Check if any log file is growing too large
                        if size_mb > 100:  # 100 MB
                            logger.warning(f"Log file {log_file} is very large: {size_mb:.2f} MB")
                    else:
                        logger.warning(f"Log file {log_file} does not exist")
                
                # Check if the situation generator is running
                if self.autonomous_mode and not self.situation_generator.running:
                    logger.error("Situation generator is not running but autonomous mode is enabled")
                    # Try to restart it
                    try:
                        self.situation_generator.start()
                        logger.info("Restarted situation generator")
                    except Exception as e:
                        logger.error(f"Failed to restart situation generator: {e}")
                
                # Check memory server
                try:
                    import requests
                    response = requests.get("http://127.0.0.1:8000/health", timeout=2)
                    if response.status_code == 200:
                        logger.debug("Memory server is responsive")
                    else:
                        logger.warning(f"Memory server returned status code {response.status_code}")
                except Exception as e:
                    logger.warning(f"Memory server health check failed: {e}")
                    
            except Exception as e:
                logger.error(f"Error in system health monitor: {e}", exc_info=True)
                time.sleep(health_check_interval)  # Wait before trying again
                
        logger.info("System health monitor stopped")

def main():
    parser = argparse.ArgumentParser(description="AGI System")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--auto", action="store_true", help="Start in autonomous mode")
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    agi = AGISystem()
    
    if args.auto:
        agi.autonomous_mode = True
    
    agi.start()

if __name__ == "__main__":
    main() 