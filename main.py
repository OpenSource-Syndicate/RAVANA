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
        self.event_detector = EventDetector()
        self.memory_queue = queue.Queue()
        self.task_queue = queue.Queue()
        self.situation_queue = queue.Queue()
        self.running = False
        self.memory_server_thread = None
        self.situation_generator = SituationGenerator()
        self.autonomous_mode = False
        self.autonomous_thread = None
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
        events = self.event_detector.process(texts)
        logger.info(f"Detected {len(events)} events")
        return events
    
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
        """Process a situation with nuanced logic for each type."""
        logger.info(f"Processing situation of type: {situation['type']}")
        
        prompt = situation.get("prompt", "No prompt provided.")
        context = situation.get("context", {})
        
        # Get relevant memories for context
        relevant_memories = self.retrieve_relevant_memories(prompt)
        context["memories"] = relevant_memories
        
        response = ""
        situation_type = situation.get("type")

        try:
            if situation_type == "trending_topic":
                articles = context.get("articles", [])
                events = self.detect_events(articles) if articles else []
                full_prompt = f"Analyze this trending topic: {prompt}\n\nRelevant context: {json.dumps(context)}\n\nDetected events: {json.dumps(events)}"
                response = run_langchain_reflection(full_prompt, str(context))

            elif situation_type == "curiosity_exploration":
                article = context.get("article", "")
                full_prompt = f"Explore this curiosity topic: {prompt}\n\nArticle: {article}"
                response = run_langchain_reflection(full_prompt, str(context))

            elif situation_type == "simple_reflection":
                # For simple reflections, the prompt is sufficient
                response = run_langchain_reflection(prompt, str(context))

            elif situation_type in ["hypothetical_scenario", "technical_challenge", "ethical_dilemma", "creative_task"]:
                full_prompt = f"Respond to this situation: {prompt}\n\nRelevant context: {json.dumps(context)}"
                response = run_langchain_reflection(full_prompt, str(context))

            else:
                logger.warning(f"Unknown situation type: {situation_type}. Using default processing.")
                response = run_langchain_reflection(prompt, str(context))
            
            # The reflection is now implicitly handled by the main loop's "Reflective" mood state
            # self.reflect_on_action(f"Responded to {situation_type}: {prompt}", response)

        except Exception as e:
            logger.error(f"Error processing situation type {situation_type}: {e}")
            response = f"I encountered an error while processing the situation: {e}"
            # Update emotional state to reflect the error
            self.emotional_intelligence.process_action_result({"error": True})

        return response

    def autonomous_thread_function(self):
        """Thread function for autonomous operation with decision-making."""
        logger.info("Starting autonomous operation with decision-making")
        
        while self.running and self.autonomous_mode:
            try:
                dominant_mood = self.emotional_intelligence.get_dominant_mood()
                logger.info(f"Dominant mood: {dominant_mood}")

                if dominant_mood == "Reflective" and self.last_situation and self.last_response:
                    logger.info("Reflecting on the last action...")
                    reflection = self.reflect_on_action(str(self.last_situation), self.last_response)
                    self.process_memory(f"Reflection on last action: {reflection}")
                    self.emotional_intelligence.process_action_result({"success": True, "major_completion": True})
                    time.sleep(10)

                elif dominant_mood == "Curious":
                    logger.info("Feeling curious, triggering curiosity module...")
                    # Using default topics for now
                    recent_topics = ["artificial intelligence", "philosophy of mind", "cognitive science"]
                    article, prompt = self.trigger_curiosity(recent_topics)
                    response = self.process_input(prompt)
                    self.emotional_intelligence.process_action_natural(response)
                    time.sleep(10)

                else: # Default behavior: process a new situation
                    logger.info("Getting new situation from generator...")
                    situation = self.situation_generator.get_situation(timeout=5)
                    
                    if situation:
                        self.last_situation = situation
                        response = self.process_situation(situation)
                        self.last_response = response
                        
                        logger.info(f"Situation: {situation.get('prompt')}")
                        logger.info(f"Response: {response}")
                        
                        self.process_memory(f"Response to situation '{situation.get('type')}': {response}")
                        self.emotional_intelligence.process_action_natural(response)
                        
                        time.sleep(5)
                    else:
                        logger.info("No situation available, sleeping for a moment.")
                        time.sleep(10)
                        self.emotional_intelligence.process_action_result({"inactivity": True})

            except Exception as e:
                logger.error(f"Error in autonomous thread: {e}")
                self.emotional_intelligence.process_action_result({"error": True})
                time.sleep(10)
    
    def start_autonomous_mode(self):
        """Start autonomous mode"""
        if self.autonomous_mode:
            logger.warning("Autonomous mode already running")
            return
        
        # Start the situation generator
        self.situation_generator.start()
        
        # Start the autonomous thread
        self.autonomous_mode = True
        self.autonomous_thread = threading.Thread(target=self.autonomous_thread_function)
        self.autonomous_thread.daemon = True
        self.autonomous_thread.start()
        
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
        print("AGI System is running. Type 'exit' to quit, 'auto' to start autonomous mode, 'manual' to stop autonomous mode.")
        
        while self.running:
            try:
                user_input = input("> ")
                if user_input.lower() == "exit":
                    self.running = False
                    break
                elif user_input.lower() == "auto":
                    self.start_autonomous_mode()
                    print("Autonomous mode started")
                elif user_input.lower() == "manual":
                    self.stop_autonomous_mode()
                    print("Autonomous mode stopped")
                else:
                    response = self.process_input(user_input)
                    print(response)
            except KeyboardInterrupt:
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error processing input: {e}")
        
        # Stop autonomous mode if it's running
        if self.autonomous_mode:
            self.stop_autonomous_mode()
        
        logger.info("AGI System stopped")
    
def main():
    parser = argparse.ArgumentParser(description="AGI System")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--auto", action="store_true", help="Start in autonomous mode")
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    agi = AGISystem()
    
    if args.auto:
        agi.start_autonomous_mode()
    
    agi.start()

if __name__ == "__main__":
    main() 