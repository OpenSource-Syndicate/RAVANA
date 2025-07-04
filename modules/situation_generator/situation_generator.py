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
import asyncio

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

# Import required modules
try:
    # Import curiosity_trigger
    from ..curiosity_trigger.curiosity_trigger import CuriosityTrigger
    
    # Import trend_analysis
    from ..information_processing.trend_analysis.trend_engine import fetch_feeds, analyze_trends, setup_db
    
    # Import event_detection
    from ..event_detection.event_detector import process_data_for_events, load_models as load_event_models
    
    # Import agent_self_reflection
    from ..agent_self_reflection.llm import call_llm
    
    # Import config
    from core.config import Config
    
    logger.info("All required modules imported successfully")
except ImportError as e:
    logger.error(f"Error importing modules: {e}", exc_info=True)
    sys.exit(1)

class SituationGenerator:
    """
    Generates situations for the AGI system to tackle without user input.
    """
    
    def __init__(self, log_level=logging.INFO):
        # Set up logger
        self.logger = logging.getLogger("SituationGenerator")
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

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
        self.feed_urls = Config.FEED_URLS
        
        self.logger.info("SituationGenerator initialized")
    
    async def generate_trending_topic_situation(self) -> Dict[str, Any]:
        """Generate a situation based on trending topics from RSS feeds."""
        try:
            # Fetch latest feeds
            await asyncio.to_thread(fetch_feeds, self.feed_urls)
            
            # Get recent articles from database
            import sqlite3
            db_path = os.path.join(os.path.dirname(__file__), "../../trends.db")
            if not os.path.exists(db_path):
                self.logger.warning(f"Trends database not found at {db_path}. Skipping trending topic.")
                return self.generate_simple_reflection_situation()

            conn = sqlite3.connect(db_path)
            c = conn.cursor()

            # Check if articles table exists
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='articles'")
            if c.fetchone() is None:
                conn.close()
                self.logger.warning("'articles' table not found in trends.db. Skipping trending topic.")
                return self.generate_simple_reflection_situation()

            c.execute('SELECT title FROM articles ORDER BY timestamp DESC LIMIT 20')
            articles = [row[0] for row in c.fetchall()]
            conn.close()
            
            if not articles:
                return await self.generate_hypothetical_scenario()
            
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
            self.logger.error(f"Error generating trending topic situation: {e}")
            return await self.generate_hypothetical_scenario()
    
    async def generate_curiosity_situation(self, curiosity_topics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate a situation based on the curiosity trigger module."""
        try:
            # Generate some recent topics if not provided
            if not curiosity_topics:
                self.logger.info("No curiosity topics provided, using default list.")
                curiosity_topics = [
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
            article, prompt = await asyncio.to_thread(CuriosityTrigger.trigger, curiosity_topics, lateralness=0.8)
            
            return {
                "type": "curiosity_exploration",
                "prompt": prompt,
                "context": {
                    "article": article[:1000] if article else "No article available."
                }
            }
        except Exception as e:
            self.logger.error(f"Error generating curiosity situation: {e}")
            return await self.generate_hypothetical_scenario()
    
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
    
    async def generate_hypothetical_scenario(self) -> Dict[str, Any]:
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
        
        situation = await asyncio.to_thread(call_llm, prompt)
        
        if situation is None:
            self.logger.warning("LLM call failed. Falling back to simple reflection.")
            return self.generate_simple_reflection_situation()
            
        return {
            "type": "hypothetical_scenario",
            "prompt": situation,
            "context": {
                "scenario_type": scenario
            }
        }
    
    async def generate_technical_challenge(self) -> Dict[str, Any]:
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
        
        situation = await asyncio.to_thread(call_llm, prompt)
        
        if situation is None:
            self.logger.warning("LLM call failed. Falling back to simple reflection.")
            return self.generate_simple_reflection_situation()

        return {
            "type": "technical_challenge",
            "prompt": situation,
            "context": {
                "challenge_type": challenge
            }
        }
    
    async def generate_ethical_dilemma(self) -> Dict[str, Any]:
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
        
        situation = await asyncio.to_thread(call_llm, prompt)
        
        if situation is None:
            self.logger.warning("LLM call failed. Falling back to simple reflection.")
            return self.generate_simple_reflection_situation()

        return {
            "type": "ethical_dilemma",
            "prompt": situation,
            "context": {
                "dilemma_type": dilemma
            }
        }
    
    async def generate_creative_task(self) -> Dict[str, Any]:
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
        
        situation = await asyncio.to_thread(call_llm, prompt)
        
        if situation is None:
            self.logger.warning("LLM call failed. Falling back to simple reflection.")
            return self.generate_simple_reflection_situation()

        return {
            "type": "creative_task",
            "prompt": situation,
            "context": {
                "task_type": task
            }
        }
    
    async def generate_situation(self, curiosity_topics: Optional[List[str]] = None, behavior_modifiers: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generates a new situation for the AGI to handle, optionally influenced by behavior modifiers.
        """
        if behavior_modifiers is None:
            behavior_modifiers = {}

        # Prioritize situation types based on behavior modifiers
        if behavior_modifiers.get('activate_self_reflection'):
            return self.generate_simple_reflection_situation()
        if behavior_modifiers.get('explore_more'):
            return await self.generate_curiosity_situation(curiosity_topics)
        if behavior_modifiers.get('try_simpler_task'):
            # Choose from a list of simpler tasks
            situation_type = random.choice(['simple_reflection', 'creative_task'])
        elif behavior_modifiers.get('take_on_harder_challenges'):
            # Choose from a list of more complex tasks
            situation_type = random.choice(['technical_challenge', 'hypothetical_scenario', 'ethical_dilemma'])
        else:
            situation_type = random.choice(self.situation_types)

        self.logger.info(f"Generating situation of type: {situation_type}")

        if situation_type == "trending_topic":
            return await self.generate_trending_topic_situation()
        elif situation_type == "curiosity_exploration":
            return await self.generate_curiosity_situation(curiosity_topics)
        elif situation_type == "simple_reflection":
            return self.generate_simple_reflection_situation()
        elif situation_type == "hypothetical_scenario":
            return await self.generate_hypothetical_scenario()
        elif situation_type == "technical_challenge":
            return await self.generate_technical_challenge()
        elif situation_type == "ethical_dilemma":
            return await self.generate_ethical_dilemma()
        elif situation_type == "creative_task":
            return await self.generate_creative_task()
        
        # Fallback to a default situation
        return self.generate_simple_reflection_situation()

def main():
    """
    Main function to test the SituationGenerator.
    This will generate a few situations and print them.
    """
    async def test_generator():
        generator = SituationGenerator()
        for _ in range(5):
            situation = await generator.generate_situation()
            print(json.dumps(situation, indent=2))
            print("-" * 20)

    asyncio.run(test_generator())

if __name__ == '__main__':
    main() 