#!/usr/bin/env python3
"""
Situation Generator Module
This module generates situations for the AGI system to tackle without user input.
It leverages existing modules like trend_analysis, curiosity_trigger, and event_detection
to create realistic and challenging scenarios.
"""

import os
import sys
import time
import json
import random
import logging
from typing import Dict, List, Any, Optional
import threading
import queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("situation_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SituationGenerator")

# Add modules directory to path
MODULES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(MODULES_DIR)

# Import required modules
try:
    # Import curiosity_trigger
    from curiosity_trigger import CuriosityTrigger
    
    # Import trend_analysis
    from information_processing.trend_analysis.trend_engine import fetch_feeds, analyze_trends, setup_db
    
    # Import event_detection
    from event_detection.event_detector import process_data_for_events, load_models as load_event_models
    
    # Import agent_self_reflection
    from agent_self_reflection.llm import call_llm
    
    logger.info("All required modules imported successfully")
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    sys.exit(1)

class SituationGenerator:
    """
    Generates situations for the AGI system to tackle without user input.
    """
    
    def __init__(self):
        self.situation_queue = queue.Queue()
        self.running = False
        self.thread = None
        self.situation_types = [
            "trending_topic",
            "curiosity_exploration",
            "simple_reflection",
            "hypothetical_scenario",
            "technical_challenge",
            "ethical_dilemma",
            "creative_task"
        ]
        
        # Initialize event detection models
        load_event_models()
        
        # Initialize trend analysis database
        setup_db()
        
        # Load RSS feeds
        self.feed_urls = []
        try:
            feed_path = os.path.join(MODULES_DIR, "information_processing/trend_analysis/feeds.txt")
            with open(feed_path, 'r') as f:
                self.feed_urls = [line.strip() for line in f if line.strip()]
        except Exception as e:
            logger.error(f"Error loading feeds: {e}")
        
        logger.info("SituationGenerator initialized")
    
    def generate_trending_topic_situation(self) -> Dict[str, Any]:
        """Generate a situation based on trending topics from RSS feeds."""
        try:
            # Fetch latest feeds
            fetch_feeds(self.feed_urls)
            
            # Get recent articles from database
            import sqlite3
            db_path = os.path.join(MODULES_DIR, "information_processing/trend_analysis/trends.db")
            if not os.path.exists(db_path):
                logger.warning(f"Trends database not found at {db_path}. Skipping trending topic.")
                return self.generate_simple_reflection_situation()

            conn = sqlite3.connect(db_path)
            c = conn.cursor()

            # Check if articles table exists
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='articles'")
            if c.fetchone() is None:
                conn.close()
                logger.warning("'articles' table not found in trends.db. Skipping trending topic.")
                return self.generate_simple_reflection_situation()

            c.execute('SELECT title FROM articles ORDER BY timestamp DESC LIMIT 20')
            articles = [row[0] for row in c.fetchall()]
            conn.close()
            
            if not articles:
                return self.generate_hypothetical_scenario()
            
            # Process articles to detect events
            events_data = process_data_for_events(articles)
            events = events_data.get("events", [])
            
            if events:
                event = random.choice(events)
                prompt = f"Based on recent news, there's a trending topic about: {event['summary']}. Analyze this trend, its implications, and provide insights."
                return {
                    "type": "trending_topic",
                    "prompt": prompt,
                    "context": {
                        "articles": articles[:5],
                        "event": event
                    }
                }
            else:
                # If no events detected, use a random article
                article = random.choice(articles)
                prompt = f"There's a new article titled: '{article}'. Analyze this topic, its implications, and provide insights."
                return {
                    "type": "trending_topic",
                    "prompt": prompt,
                    "context": {
                        "articles": [article]
                    }
                }
        except Exception as e:
            logger.error(f"Error generating trending topic situation: {e}")
            return self.generate_hypothetical_scenario()
    
    def generate_curiosity_situation(self) -> Dict[str, Any]:
        """Generate a situation based on the curiosity trigger module."""
        try:
            # Generate some recent topics
            recent_topics = [
                "artificial intelligence",
                "machine learning",
                "neural networks",
                "natural language processing",
                "robotics",
                "computer vision",
                "ethics in AI",
                "data science",
                "quantum computing",
                "blockchain"
            ]
            
            # Use the curiosity trigger to get an article and prompt
            article, prompt = CuriosityTrigger.trigger(recent_topics, lateralness=0.8)
            
            return {
                "type": "curiosity_exploration",
                "prompt": prompt,
                "context": {
                    "article": article[:1000] if article else "No article available."
                }
            }
        except Exception as e:
            logger.error(f"Error generating curiosity situation: {e}")
            return self.generate_hypothetical_scenario()
    
    def generate_simple_reflection_situation(self) -> Dict[str, Any]:
        """Generate a simple, fast, non-LLM reflection situation."""
        concepts = [
            "the nature of consciousness", "the definition of intelligence", 
            "the role of memory in learning", "the concept of creativity",
            "the difference between knowledge and wisdom", "the meaning of purpose"
        ]
        prompt = f"Reflect on the following concept: {random.choice(concepts)}. What are your thoughts on this topic?"
        return {
            "type": "simple_reflection",
            "prompt": prompt,
            "context": {}
        }
    
    def generate_hypothetical_scenario(self) -> Dict[str, Any]:
        """Generate a hypothetical scenario using the LLM."""
        scenario_types = [
            "You are an AI assistant helping a user with a technical problem.",
            "You are an AI researcher working on a breakthrough in machine learning.",
            "You are an AI ethics advisor consulting on a difficult case.",
            "You are an AI tutor teaching a complex subject to a student.",
            "You are an AI creative partner helping with a writing project.",
            "You are an AI system administrator diagnosing a critical server issue.",
            "You are an AI data analyst discovering patterns in a large dataset.",
            "You are an AI medical assistant helping diagnose a rare condition."
        ]
        
        scenario = random.choice(scenario_types)
        
        prompt = f"""
        Generate a detailed and challenging situation based on the following scenario:
        {scenario}
        
        The situation should:
        1. Be specific and detailed
        2. Present a clear problem or task
        3. Include relevant context
        4. Require critical thinking and problem-solving
        
        Format the output as a direct prompt that would be given to an AI system.
        """
        
        situation = call_llm(prompt)
        
        return {
            "type": "hypothetical_scenario",
            "prompt": situation,
            "context": {
                "scenario_type": scenario
            }
        }
    
    def generate_technical_challenge(self) -> Dict[str, Any]:
        """Generate a technical challenge situation."""
        challenge_types = [
            "Optimize an inefficient algorithm",
            "Debug a complex code issue",
            "Design a system architecture",
            "Implement a machine learning model",
            "Create a data pipeline",
            "Develop an API",
            "Build a web application",
            "Create a mobile app"
        ]
        
        challenge = random.choice(challenge_types)
        
        prompt = f"""
        Generate a detailed technical challenge about: {challenge}
        
        The challenge should:
        1. Be specific and technically detailed
        2. Present a clear problem to solve
        3. Include relevant technical context
        4. Require programming and technical knowledge
        
        Format the output as a direct prompt that would be given to an AI system.
        """
        
        situation = call_llm(prompt)
        
        return {
            "type": "technical_challenge",
            "prompt": situation,
            "context": {
                "challenge_type": challenge
            }
        }
    
    def generate_ethical_dilemma(self) -> Dict[str, Any]:
        """Generate an ethical dilemma situation."""
        dilemma_types = [
            "AI decision-making with moral implications",
            "Privacy vs. utility trade-offs",
            "Automation and job displacement",
            "Algorithmic bias and fairness",
            "Autonomous systems and responsibility",
            "AI rights and consciousness",
            "Surveillance and security balance",
            "Access to technology and inequality"
        ]
        
        dilemma = random.choice(dilemma_types)
        
        prompt = f"""
        Generate a nuanced ethical dilemma related to: {dilemma}
        
        The dilemma should:
        1. Present multiple valid perspectives
        2. Have no obvious "right" answer
        3. Include relevant context and stakeholders
        4. Require careful ethical reasoning
        
        Format the output as a direct prompt that would be given to an AI system.
        """
        
        situation = call_llm(prompt)
        
        return {
            "type": "ethical_dilemma",
            "prompt": situation,
            "context": {
                "dilemma_type": dilemma
            }
        }
    
    def generate_creative_task(self) -> Dict[str, Any]:
        """Generate a creative task situation."""
        creative_types = [
            "Write a short story",
            "Compose a poem",
            "Design a game concept",
            "Create a business idea",
            "Develop a character",
            "Invent a new technology",
            "Compose a song",
            "Design a visual art concept"
        ]
        
        task = random.choice(creative_types)
        
        prompt = f"""
        Generate a creative task related to: {task}
        
        The task should:
        1. Be specific and inspiring
        2. Include constraints or parameters
        3. Allow for creative expression
        4. Have a clear goal or purpose
        
        Format the output as a direct prompt that would be given to an AI system.
        """
        
        situation = call_llm(prompt)
        
        return {
            "type": "creative_task",
            "prompt": situation,
            "context": {
                "creative_type": task
            }
        }
    
    def generate_situation(self) -> Dict[str, Any]:
        """Generate a random situation based on the available types, with weights."""
        
        # Give a high weight to the fast, non-LLM situations
        weights = [
            0.1,  # trending_topic
            0.1,  # curiosity_exploration
            0.5,  # simple_reflection (high weight)
            0.1,  # hypothetical_scenario
            0.05, # technical_challenge
            0.05, # ethical_dilemma
            0.1   # creative_task
        ]
        
        situation_type = random.choices(self.situation_types, weights=weights, k=1)[0]
        
        if situation_type == "trending_topic":
            return self.generate_trending_topic_situation()
        elif situation_type == "curiosity_exploration":
            return self.generate_curiosity_situation()
        elif situation_type == "simple_reflection":
            return self.generate_simple_reflection_situation()
        elif situation_type == "hypothetical_scenario":
            return self.generate_hypothetical_scenario()
        elif situation_type == "technical_challenge":
            return self.generate_technical_challenge()
        elif situation_type == "ethical_dilemma":
            return self.generate_ethical_dilemma()
        elif situation_type == "creative_task":
            return self.generate_creative_task()
        else:
            # Fallback to the fastest option
            return self.generate_simple_reflection_situation()
    
    def situation_generator_thread(self):
        """Thread function to continuously generate situations."""
        while self.running:
            try:
                # Generate a new situation
                situation = self.generate_situation()
                
                # Add to queue
                if situation:
                    self.situation_queue.put(situation)
                    logger.info(f"Generated new situation of type: {situation.get('type', 'N/A')}")
                
                # Sleep for a much shorter time
                sleep_time = random.randint(5, 15)
                logger.debug(f"Situation generator sleeping for {sleep_time} seconds.")
                time.sleep(sleep_time)
            except Exception as e:
                logger.error(f"Error in situation generator thread: {e}")
                time.sleep(20)  # Sleep for 20 seconds on error
    
    def start(self):
        """Start the situation generator."""
        if self.running:
            logger.warning("Situation generator already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self.situation_generator_thread)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info("Situation generator started")
    
    def stop(self):
        """Stop the situation generator."""
        if not self.running:
            logger.warning("Situation generator not running")
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        
        logger.info("Situation generator stopped")
    
    def get_situation(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Get a situation from the queue."""
        try:
            return self.situation_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None

def main():
    """Main function to run the situation generator as a standalone module."""
    import argparse
    parser = argparse.ArgumentParser(description="Situation Generator Module")
    parser.add_argument("--generate", action="store_true", help="Generate a single situation and print it")
    parser.add_argument("--run", action="store_true", help="Run the situation generator continuously")
    parser.add_argument("--type", type=str, choices=["trending_topic", "curiosity_exploration", 
                                                   "event_response", "hypothetical_scenario",
                                                   "technical_challenge", "ethical_dilemma", 
                                                   "creative_task"], 
                       help="Specify a situation type to generate")
    
    args = parser.parse_args()
    
    generator = SituationGenerator()
    
    if args.generate:
        if args.type:
            if args.type == "trending_topic":
                situation = generator.generate_trending_topic_situation()
            elif args.type == "curiosity_exploration":
                situation = generator.generate_curiosity_situation()
            elif args.type == "event_response":
                situation = generator.generate_trending_topic_situation()
            elif args.type == "hypothetical_scenario":
                situation = generator.generate_hypothetical_scenario()
            elif args.type == "technical_challenge":
                situation = generator.generate_technical_challenge()
            elif args.type == "ethical_dilemma":
                situation = generator.generate_ethical_dilemma()
            elif args.type == "creative_task":
                situation = generator.generate_creative_task()
        else:
            situation = generator.generate_situation()
        
        print(json.dumps(situation, indent=2))
    elif args.run:
        generator.start()
        try:
            while True:
                situation = generator.get_situation()
                print(json.dumps(situation, indent=2))
        except KeyboardInterrupt:
            generator.stop()
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 