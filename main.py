import asyncio
import logging
import sys
import os

# Add modules directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.decision_engine.llm import decision_maker_loop, generate_hypothetical_scenarios
from modules.knowledge_compression.main import compress_knowledge
from modules.information_processing.youtube_transcription.youtube_transcription import transcribe_youtube_video
from modules.information_processing.trend_analysis.trend_engine import fetch_feeds, analyze_trends
from modules.event_detection.event_detector import process_data_for_events, load_models as load_event_models
from modules.episodic_memory.memory import (
    extract_memories_api,
    save_memories,
    get_relevant_memories_api,
    consolidate_memories_api,
    startup_event as init_memory_db,
    app as memory_app
)
from modules.agent_self_reflection.self_modification import run_self_modification
from modules.situation_generator.situation_generator import SituationGenerator
from modules.emotional_intellegence.emotional_intellegence import EmotionalIntelligence
from modules.curiosity_trigger.curiosity_trigger import CuriosityTrigger
from modules.agi_experimentation.llm import agi_experimentation_engine
from sentence_transformers import SentenceTransformer


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AGISystem:
    def __init__(self):
        logger.info("Initializing Ravana AGI System...")
        
        # Load shared models
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        load_event_models(embedding_model_instance=self.embedding_model)
        memory_app.embedding_model = self.embedding_model


        # Initialize modules
        self.situation_generator = SituationGenerator()
        self.emotional_intelligence = EmotionalIntelligence()
        self.episodic_memory = init_memory_db()


        # Shared state
        self.shared_state = {
            "mood": self.emotional_intelligence.get_mood_vector(),
            "current_situation": None,
            "recent_memories": [],
            "long_term_goals": [],
        }
    
    async def run_autonomous_loop(self):
        logger.info("Starting autonomous loop...")
        while True:
            try:
                # 1. Sense: Generate a new situation
                situation = self.situation_generator.generate_situation()
                self.shared_state['current_situation'] = situation
                logger.info(f"New Situation: {situation['type']} - {situation['prompt']}")

                # 2. Recall: Get relevant memories
                relevant_memories = await get_relevant_memories_api({"query_text": situation['prompt']})
                self.shared_state['recent_memories'] = relevant_memories.relevant_memories

                # 3. Think: Decide on a plan
                decision_input = {
                    "situation": situation['prompt'],
                    "memory": [mem.text for mem in relevant_memories.relevant_memories],
                    "rag_context": situation.get('context'),
                }
                decision = await asyncio.to_thread(decision_maker_loop, **decision_input)
                logger.info(f"Decision: {decision}")
                
                # 4. Act: Execute the plan
                action_output = await self.execute_action(decision)
                logger.info(f"Action Output: {action_output}")

                # 5. Feel: Update emotional state
                self.emotional_intelligence.process_action_natural(action_output)
                self.shared_state['mood'] = self.emotional_intelligence.get_mood_vector()
                logger.info(f"Updated Mood: {self.shared_state['mood']}")

                # 6. Memorize: Store the interaction
                interaction_summary = f"Situation: {situation['prompt']}\nDecision: {decision}\nAction Output: {action_output}"
                memories_to_save = await extract_memories_api({"user_input": interaction_summary, "ai_output": ""})
                await asyncio.to_thread(save_memories, memories_to_save.memories)
                
                # 7. Reflect: Run self-reflection
                if self.emotional_intelligence.get_dominant_mood() == "Reflective":
                    await asyncio.to_thread(run_self_modification)

                await asyncio.sleep(10) # Pause between loops
            except Exception as e:
                logger.error(f"Error in autonomous loop: {e}", exc_info=True)
                await asyncio.sleep(60) # Wait a minute before retrying
                
    async def execute_action(self, decision):
        raw_response = decision.get('raw_response', "")
        # Basic parsing to find if a tool should be used.
        # This is a simplified approach. A more robust solution would use function calling from the LLM.
        if "experiment" in raw_response.lower():
            return await asyncio.to_thread(agi_experimentation_engine, raw_response)
        elif "youtube.com" in raw_response.lower():
            # Extract URL
            url = raw_response.split(" ")[-1]
            return await asyncio.to_thread(transcribe_youtube_video, url)
        elif "trends" in raw_response.lower():
            return await asyncio.to_thread(analyze_trends)
        elif "curiosity" in raw_response.lower():
            return await asyncio.to_thread(CuriosityTrigger.trigger, [])
        else:
            return "No specific action taken from decision."

async def main():
    agi = AGISystem()
    await agi.run_autonomous_loop()

if __name__ == "__main__":
    asyncio.run(main()) 