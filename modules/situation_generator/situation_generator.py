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
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler("situation_generator.log"),
#         logging.StreamHandler()
#     ]
# )
logger = logging.getLogger("SituationGenerator")

# Import required modules
try:
    # Import curiosity_trigger
    from ..curiosity_trigger.curiosity_trigger import CuriosityTrigger
    
    # Import trend_analysis
    from ..information_processing.trend_analysis.trend_engine import fetch_feeds, setup_db
    
    # Import event_detection
    from ..event_detection.event_detector import process_data_for_events
    
    # Import agent_self_reflection
    from ..agent_self_reflection.llm import call_llm
    from ..agent_self_reflection.self_modification import generate_hypothesis, analyze_experiment_outcome
    
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
    
    def __init__(self, log_level=logging.INFO, embedding_model=None, sentiment_classifier=None):
        # Set up logger
        self.logger = logging.getLogger("SituationGenerator")
        self.logger.setLevel(log_level)
        # if not self.logger.handlers:
        #     handler = logging.StreamHandler()
        #     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        #     handler.setFormatter(formatter)
        #     self.logger.addHandler(handler)

        self.situation_types = [
            "llm_generated",
            "trending_topic",
            "curiosity_exploration",
            "simple_reflection",
            "hypothetical_scenario",
            "technical_challenge",
            "ethical_dilemma",
            "creative_task",
            "search_result_analysis"
        ]
        
        # LLM State
        self.mood = "neutral"
        self.memories = []
        self.collected_data = []

        # Initialize event detection models
        self.embedding_model = embedding_model
        self.sentiment_classifier = sentiment_classifier
        
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
                return await self.generate_simple_reflection_situation()

            conn = sqlite3.connect(db_path)
            c = conn.cursor()

            # Check if articles table exists
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='articles'")
            if c.fetchone() is None:
                conn.close()
                self.logger.warning("'articles' table not found in trends.db. Skipping trending topic.")
                return await self.generate_simple_reflection_situation()

            c.execute('SELECT title FROM articles ORDER BY timestamp DESC LIMIT 20')
            articles = [row[0] for row in c.fetchall()]
            conn.close()
            
            if not articles:
                return await self.generate_hypothetical_scenario()
            
            # Process articles to detect events
            events_data = process_data_for_events(
                articles,
                embedding_model=self.embedding_model,
                sentiment_classifier=self.sentiment_classifier
            )
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
        """Generate a situation based on the curiosity trigger module, making it more relevant to the current context."""
        try:
            # If no specific topics are provided, generate them from the agent's context
            if not curiosity_topics:
                self.logger.info("No curiosity topics provided, generating from agent's context.")
                context_str = f"Mood: {self.mood}, Recent Memories: {self.memories[-5:]}"
                article, prompt = await asyncio.to_thread(CuriosityTrigger.from_context, context_str, lateralness=0.5)
            else:
                # Use the provided topics
                article, prompt = await asyncio.to_thread(lambda: asyncio.run(CuriosityTrigger().trigger(curiosity_topics, lateralness=0.8)))

            if not article or "No article available" in article:
                self.logger.warning("Could not fetch a curiosity article. Falling back to a hypothetical scenario.")
                return await self.generate_hypothetical_scenario()

            return {
                "type": "curiosity_exploration",
                "prompt": prompt,
                "context": {
                    "article": article[:1500] if article else "No article available.",
                    "topics": curiosity_topics
                }
            }
        except Exception as e:
            self.logger.error(f"Error generating curiosity situation: {e}", exc_info=True)
            return await self.generate_hypothetical_scenario()
    
    async def generate_simple_reflection_situation(self, hypothesis: Optional[str] = None) -> Dict[str, Any]:
        """Generate a simple reflection situation with a concept from the LLM."""
        try:
            # Prompt for the LLM to generate a concept
            concept_prompt = f"""
            Based on your current state:
            - Mood: {self.mood}
            - Recent Memories: {self.memories[-5:]}

            Suggest a single, abstract, or philosophical concept to reflect upon. The concept should be just a few words.
            Example: The nature of creativity.
            """
            concept = await asyncio.to_thread(call_llm, concept_prompt) if not hypothesis else hypothesis

            if not concept or "failed" in concept:
                self.logger.warning("LLM call for concept failed. Falling back to a random concept.")
                concepts = [
                    "the nature of consciousness", "the definition of intelligence",
                    "the role of memory in learning", "the concept of creativity",
                    "the difference between knowledge and wisdom", "the meaning of purpose"
                ]
                concept = random.choice(concepts)

            prompt = f"Reflect on the following concept: {concept}. What are your thoughts on this topic?"
            
            return {
                "type": "simple_reflection",
                "prompt": prompt,
                "context": {
                    "concept": concept
                }
            }
        except Exception as e:
            self.logger.error(f"Error generating simple reflection situation: {e}", exc_info=True)
            # Fallback to a hardcoded concept if everything fails
            concepts = [
                "the nature of consciousness", "the definition of intelligence", 
                "the role of memory in learning", "the concept of creativity",
                "the difference between knowledge and wisdom", "the meaning of purpose"
            ]
            concept = random.choice(concepts)
            prompt = f"Reflect on the following concept: {concept}. What are your thoughts on this topic?"
            return {
                "type": "simple_reflection",
                "prompt": prompt,
                "context": {}
            }
    
    async def generate_hypothetical_scenario(self) -> Dict[str, Any]:
        """Generate a hypothetical scenario using the LLM."""
        try:
            scenario_type_prompt = f"""
            Based on your current state:
            - Mood: {self.mood}
            - Recent Memories: {self.memories[-5:]}

            Suggest a high-level scenario for an AI to be in. Just the scenario, a few words.
            The scenario can be grounded in reality or highly speculative and philosophical.
            Example: An AI assistant helping a user with a technical problem.
            Example: An AI philosopher debating the nature of consciousness with a human.
            Example: An AI physicist attempting to design a warp drive.
            """
            scenario = await asyncio.to_thread(call_llm, scenario_type_prompt)

            if not scenario or "failed" in scenario:
                self.logger.warning("LLM call for scenario type failed. Falling back to a random scenario type.")
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
                return await self.generate_simple_reflection_situation()
                
            return {
                "type": "hypothetical_scenario",
                "prompt": situation,
                "context": {
                    "scenario_type": scenario
                }
            }
        except Exception as e:
            self.logger.error(f"Error generating hypothetical_scenario: {e}", exc_info=True)
            return await self.generate_simple_reflection_situation()
    
    async def generate_technical_challenge(self) -> Dict[str, Any]:
        """Generate a technical challenge situation."""
        try:
            challenge_type_prompt = f"""
            Based on your current state:
            - Mood: {self.mood}
            - Recent Memories: {self.memories[-5:]}

            Suggest a high-level technical challenge for an AI. Just the topic.
            Example: Optimize an inefficient algorithm.
            """
            challenge = await asyncio.to_thread(call_llm, challenge_type_prompt)

            if not challenge or "failed" in challenge:
                self.logger.warning("LLM call for challenge type failed. Falling back to a random challenge type.")
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
                return await self.generate_simple_reflection_situation()

            return {
                "type": "technical_challenge",
                "prompt": situation,
                "context": {
                    "challenge_type": challenge
                }
            }
        except Exception as e:
            self.logger.error(f"Error generating technical_challenge: {e}", exc_info=True)
            return await self.generate_simple_reflection_situation()
    
    async def generate_ethical_dilemma(self) -> Dict[str, Any]:
        """Generate an ethical dilemma situation."""
        try:
            dilemma_type_prompt = f"""
            Based on your current state:
            - Mood: {self.mood}
            - Recent Memories: {self.memories[-5:]}

            Suggest a high-level ethical dilemma for an AI. Just the topic.
            Example: AI decision-making with moral implications.
            """
            dilemma = await asyncio.to_thread(call_llm, dilemma_type_prompt)

            if not dilemma or "failed" in dilemma:
                self.logger.warning("LLM call for dilemma type failed. Falling back to a random dilemma type.")
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
                return await self.generate_simple_reflection_situation()

            return {
                "type": "ethical_dilemma",
                "prompt": situation,
                "context": {
                    "dilemma_type": dilemma
                }
            }
        except Exception as e:
            self.logger.error(f"Error generating ethical_dilemma: {e}", exc_info=True)
            return await self.generate_simple_reflection_situation()
    
    async def generate_creative_task(self) -> Dict[str, Any]:
        """Generate a creative task situation."""
        try:
            task_type_prompt = f"""
            Based on your current state:
            - Mood: {self.mood}
            - Recent Memories: {self.memories[-5:]}

            Suggest a high-level creative task for an AI. Just the topic.
            The task can be artistic, scientific, or purely imaginative.
            Example: Write a short story.
            Example: Design a theoretical model for a Dyson Sphere.
            Example: Invent a new form of music based on mathematical principles.
            """
            task = await asyncio.to_thread(call_llm, task_type_prompt)

            if not task or "failed" in task:
                self.logger.warning("LLM call for task type failed. Falling back to a random task type.")
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
                return await self.generate_simple_reflection_situation()

            return {
                "type": "creative_task",
                "prompt": situation,
                "context": {
                    "task_type": task
                }
            }
        except Exception as e:
            self.logger.error(f"Error generating creative_task: {e}", exc_info=True)
            return await self.generate_simple_reflection_situation()
    
    def update_llm_state(self, mood=None, new_memories=None, new_data=None):
        """Update the LLM's state."""
        if mood:
            self.mood = mood
        if new_memories:
            self.memories.extend(new_memories)
        if new_data:
            self.collected_data.extend(new_data)

    async def generate_llm_situation(self) -> Dict[str, Any]:
        """Generate a situation based on the LLM's mood, memories, and collected data."""
        try:
            # Construct the prompt
            prompt = f"""
            As an AI, your current state is:
            - Mood: {self.mood}
            - Recent Memories: {self.memories[-5:]}
            - Collected Data: {self.collected_data[-5:]}

            Based on this state, generate a compelling and challenging situation for you to address.
            The situation should be relevant to your recent experiences and data you've collected.
            It should be a task, a problem, or a creative challenge.
            """

            situation = await asyncio.to_thread(call_llm, prompt)

            if situation is None:
                self.logger.warning("LLM call failed. Falling back to simple reflection.")
                return await self.generate_simple_reflection_situation()

            return {
                "type": "llm_generated",
                "prompt": situation,
                "context": {
                    "mood": self.mood,
                    "memories": self.memories[-5:],
                    "collected_data": self.collected_data[-5:]
                }
            }
        except Exception as e:
            self.logger.error(f"Error generating LLM-based situation: {e}")
            return await self.generate_hypothetical_scenario()

    async def generate_search_result_situation(self, search_results: List[str]) -> Dict[str, Any]:
        """Generate a situation based on search results."""
        self.logger.info("Generating situation from search results.")
        prompt = f"Analyze the following search results and provide a summary of the key findings, insights, and any potential actions to take:\n\n{''.join(search_results)}"
        return {
            "type": "search_result_analysis",
            "prompt": prompt,
            "context": {
                "search_results": search_results
            }
        }

    async def generate_hypothesis_test_situation(self, hypothesis: str) -> Dict[str, Any]:
        """Generate a situation to test a specific hypothesis from the reflection module."""
        prompt = f"""
        Design a 'technical challenge' or a 'hypothetical scenario' to rigorously test the following hypothesis:
        Hypothesis: "{hypothesis}"

        Describe the challenge or scenario and what success or failure would indicate.
        """
        scenario_description = await asyncio.to_thread(call_llm, prompt)

        return {
            "type": "hypothesis_test",
            "prompt": scenario_description,
            "context": {
                "hypothesis": hypothesis
            }
        }

    async def generate_situation(self, shared_state: Any, curiosity_topics: Optional[List[str]] = None, behavior_modifiers: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main method to generate a situation for the AGI.
        It can be triggered by various internal states or modules.
        """
        self.logger.info("Generating a new situation...")

        # Check for search results first
        if hasattr(shared_state, 'search_results') and shared_state.search_results:
            self.logger.info("Found search results, generating a situation to analyze them.")
            search_results = shared_state.search_results
            # Clear the search results from the shared state after using them
            shared_state.search_results = []
            return await self.generate_search_result_situation(search_results)

        # Handle behavior modifiers that force a specific situation
        if behavior_modifiers:
            if behavior_modifiers.get("new_hypothesis"):
                self.logger.info(f"New hypothesis detected. Generating a situation to test it.")
                return await self.generate_hypothesis_test_situation(behavior_modifiers["new_hypothesis"])
            if "take_break" in behavior_modifiers and random.random() < 0.7:
                self.logger.info("Behavior modifier suggests a break. Generating a simple reflection.")
                return await self.generate_simple_reflection_situation()

        # Probabilistic selection of situation type, biased towards more engaging tasks
        weights = {
            "curiosity_exploration": 0.3,
            "trending_topic": 0.2,
            "technical_challenge": 0.15,
            "hypothetical_scenario": 0.15,
            "ethical_dilemma": 0.1,
            "creative_task": 0.05,
            "simple_reflection": 0.05,
        }

        # If there is an active experiment, don't generate a new situation
        if behavior_modifiers and behavior_modifiers.get('active_experiment'):
            self.logger.info("Active experiment in progress. Situation generation is paused.")
            return {
                "type": "wait",
                "prompt": "Experiment in progress. Awaiting outcome.",
                "context": {"reason": "Waiting for experiment to conclude."}
            }
        
        situation_type = random.choices(list(weights.keys()), weights=list(weights.values()), k=1)[0]
        self.logger.info(f"Generating a '{situation_type}' situation.")

        if situation_type == "trending_topic":
            return await self.generate_trending_topic_situation()
        elif situation_type == "curiosity_exploration":
            return await self.generate_curiosity_situation(curiosity_topics)
        elif situation_type == "simple_reflection":
            return await self.generate_simple_reflection_situation()
        elif situation_type == "hypothetical_scenario":
            return await self.generate_hypothetical_scenario()
        elif situation_type == "technical_challenge":
            return await self.generate_technical_challenge()
        elif situation_type == "ethical_dilemma":
            return await self.generate_ethical_dilemma()
        elif situation_type == "creative_task":
            return await self.generate_creative_task()
        else:
            # Fallback to a simple reflection
            return await self.generate_simple_reflection_situation()

async def main():
    """Main function for testing the SituationGenerator."""
    generator = SituationGenerator()
    
    class DummySharedState:
        def __init__(self):
            self.search_results = []

    shared_state = DummySharedState()

    # Test with no specific inputs
    print("--- Generating a random situation ---")
    situation = await generator.generate_situation(shared_state)
    print(json.dumps(situation, indent=2))
    
    # Test with curiosity topics
    print("\n--- Generating a situation with specific curiosity topics ---")
    topics = ["quantum computing", "renaissance art"]
    situation = await generator.generate_situation(shared_state, curiosity_topics=topics)
    print(json.dumps(situation, indent=2))
    
    # Test with a behavior modifier (hypothesis)
    print("\n--- Generating a situation to test a hypothesis ---")
    modifiers = {"new_hypothesis": "That my planning algorithm is not efficient for long-term goals."}
    situation = await generator.generate_situation(shared_state, behavior_modifiers=modifiers)
    print(json.dumps(situation, indent=2))

    # Test with search results
    print("\n--- Generating a situation with search results ---")
    shared_state.search_results = ["Search result 1", "Search result 2"]
    situation = await generator.generate_situation(shared_state)
    print(json.dumps(situation, indent=2))
    # Verify that search results are cleared
    print(f"Search results in shared state after generation: {shared_state.search_results}")

if __name__ == "__main__":
    asyncio.run(main())