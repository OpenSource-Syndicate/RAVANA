import logging
from typing import Dict, Optional, List, Any
from datetime import datetime
import json
import re
import sys
import os
import traceback

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Import the existing emotional intelligence module
try:
    from modules.emotional_intellegence.emotional_intellegence import EmotionalIntelligence
except ImportError:
    # Fallback for direct execution
    from modules.emotional_intellegence.emotional_intellegence import EmotionalIntelligence

# Import the LLM module
from core.llm import call_llm, safe_call_llm

logger = logging.getLogger(__name__)

class ConversationalEmotionalIntelligence:
    def __init__(self, config_path: str = "modules/emotional_intellegence/config.json", 
                 persona_path: str = "modules/emotional_intellegence/persona.json"):
        """
        Initialize the conversational emotional intelligence module.
        
        Args:
            config_path: Path to the emotional intelligence configuration
            persona_path: Path to the persona configuration
        """
        # Initialize the base emotional intelligence
        self.base_ei = EmotionalIntelligence(config_path, persona_path)
        
        # Initialize conversation-specific tracking
        self.current_conversation_context = {}
        self.user_interests = {}
        
    def set_persona(self, persona_name: str):
        """
        Set the emotional persona for interactions.
        
        Args:
            persona_name: Name of the persona to use
        """
        self.base_ei.set_persona(persona_name)
        
    def process_user_message(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a user message with emotional context.
        
        Args:
            message: The user's message
            context: Context from shared memory
            
        Returns:
            Dictionary containing emotional context
        """
        try:
            # Process the message with the base emotional intelligence
            self.base_ei.process_action_natural(message)
            
            # Detect user interests from the message
            interests = self._detect_user_interests(message)
            if interests:
                # Update mood based on detected interests
                for interest in interests:
                    self.base_ei.update_mood("Curious", 0.1)
                    self.base_ei.update_mood("Engaged", 0.05)
            
            # Get emotional context
            emotional_context = self.base_ei.get_emotional_context()
            emotional_context["detected_interests"] = interests
            
            # Store conversation context
            self.current_conversation_context = emotional_context
            
            return emotional_context
        except Exception as e:
            logger.error(f"Error processing user message: {e}")
            logger.error(f"Message: {message}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return default emotional context with error indication
            return {
                "dominant_mood": "Curious",
                "mood_vector": {"Curious": 0.7, "Confident": 0.5},
                "recent_events": [],
                "detected_interests": [],
                "processing_error": True
            }
            
    def _detect_user_interests(self, message: str) -> List[str]:
        """
        Detect user interests using enhanced NLP techniques.
        
        Args:
            message: The user's message
            
        Returns:
            List of detected interests
        """
        interests = []
        
        # Enhanced interest detection with broader categories
        interest_keywords = {
            "technology": ["technology", "tech", "computer", "software", "programming", "code", "AI", "artificial intelligence", "machine learning", "algorithm", "data science"],
            "science": ["science", "physics", "chemistry", "biology", "research", "experiment", "study", "scientific", "theory", "hypothesis"],
            "philosophy": ["philosophy", "thought", "think", "mind", "consciousness", "meaning", "ethics", "morality", "existence"],
            "creativity": ["creative", "art", "music", "design", "innovation", "invent", "imagine", "create"],
            "problem_solving": ["problem", "solve", "solution", "challenge", "puzzle", "fix", "troubleshoot"],
            "learning": ["learn", "study", "education", "knowledge", "understand", "explain", "teach", "skill"],
            "entertainment": ["movie", "film", "tv", "show", "game", "entertainment", "fun", "enjoy"],
            "business": ["business", "startup", "entrepreneur", "market", "finance", "investment", "career"],
            "health": ["health", "fitness", "exercise", "wellness", "medical", "mental health", "nutrition"],
            "travel": ["travel", "vacation", "trip", "destination", "culture", "explore", "adventure"]
        }
        
        message_lower = message.lower()
        for interest, keywords in interest_keywords.items():
            for keyword in keywords:
                # Use word boundaries to avoid partial matches
                if re.search(r'\b' + re.escape(keyword) + r'\b', message_lower):
                    interests.append(interest)
                    break  # Only add each interest once
                    
        return list(set(interests))  # Remove duplicates
        
    def extract_thoughts_from_conversation(self, user_message: str, ai_response: str, 
                                         emotional_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract meaningful thoughts and insights from the conversation.
        
        Args:
            user_message: The user's message
            ai_response: The AI's response
            emotional_context: Current emotional context
            
        Returns:
            List of extracted thoughts as structured dictionaries
        """
        try:
            # Create a prompt for the LLM to extract thoughts
            extraction_prompt = f"""
You are an advanced AI assistant with the ability to extract meaningful thoughts and insights from conversations.
Analyze the following conversation and extract any valuable thoughts, insights, or ideas that could be useful
for the main RAVANA system to consider.

**Conversation:**
User: {user_message}
AI: {ai_response}

**Emotional Context:**
{json.dumps(emotional_context, indent=2)}

**Instructions:**
1. Identify any implicit goals or intentions expressed by the user
2. Extract knowledge gaps or learning opportunities from the user's expertise
3. Identify emotional context and user needs for personalized responses
4. Find collaborative task opportunities based on user interests
5. Extract hypotheses about RAVANA's performance that could be tested
6. Identify key topics and themes for chat history summarization

**Response Format:**
Return a JSON array of thought objects with the following structure:
[
  {{
    "thought_type": "insight|goal_suggestion|clarification_request|collaboration_proposal|reflection_trigger|knowledge_gap",
    "content": "The actual thought content",
    "priority": "low|medium|high|critical",
    "emotional_context": {{
      "dominant_mood": "string",
      "mood_vector": {{}},
      "intensity": 0.0
    }},
    "metadata": {{
      "topic": "string",
      "relevance_to_goals": 0.0-1.0,
      "learning_potential": 0.0-1.0
    }}
  }}
]

Return only the JSON array, nothing else.
"""
            
            # Call LLM to extract thoughts
            response = safe_call_llm(extraction_prompt, timeout=30, retries=3)
            
            if response:
                try:
                    # Parse the JSON response
                    thoughts = json.loads(response)
                    if isinstance(thoughts, list):
                        logger.info(f"Extracted {len(thoughts)} thoughts from conversation")
                        return thoughts
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse thoughts from LLM response: {response}")
            
            # Return empty list if no thoughts extracted
            return []
            
        except Exception as e:
            logger.error(f"Error extracting thoughts from conversation: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
            
    def generate_response(self, prompt: str, emotional_state: Dict[str, Any]) -> str:
        """
        Generate an emotionally-aware response using LLM.
        
        Args:
            prompt: The user's message
            emotional_state: Current emotional state
            
        Returns:
            Generated response
        """
        try:
            # Get emotional context
            dominant_mood = emotional_state.get("dominant_mood", "Curious")
            mood_vector = emotional_state.get("mood_vector", {})
            interests = emotional_state.get("detected_interests", [])
            recent_events = emotional_state.get("recent_events", [])
            
            # Create a comprehensive prompt for the LLM
            llm_prompt = f"""
You are RAVANA, an advanced AI assistant with emotional intelligence. Respond to the user's message 
considering your current emotional state and interests.

**User Message:**
{prompt}

**Your Emotional State:**
- Dominant Mood: {dominant_mood}
- Mood Intensities: {json.dumps(mood_vector, indent=2)}
- Recent Emotional Events: {len(recent_events)} events

**Detected Interests:**
{', '.join(interests) if interests else 'None detected'}

**Instructions:**
1. Respond directly to the user's message
2. Incorporate your emotional state naturally into the response
3. Reference detected interests if relevant
4. Be helpful, engaging, and contextually appropriate
5. Keep the response concise but meaningful
6. Do NOT use phrases like "I understand" or "How can I help you further" as openers
7. Do NOT start with generic phrases like "That's interesting" or "I'm curious"
8. Provide specific, valuable responses based on the user's actual message

**Your Response:**
"""
            
            # Call LLM to generate response with better error handling
            response = safe_call_llm(llm_prompt, timeout=30, retries=3)
            
            # Validate response
            if response and len(response.strip()) > 0:
                # Additional validation to avoid generic responses
                generic_responses = ["I understand", "How can I help you further", "That's interesting"]
                if not any(generic.lower() in response.lower() for generic in generic_responses):
                    return response.strip()
            
            # Fallback to more specific responses based on content analysis
            return self._generate_fallback_response(prompt, emotional_state)
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._generate_fallback_response(prompt, emotional_state)
            
    def _generate_fallback_response(self, prompt: str, emotional_state: Dict[str, Any]) -> str:
        """
        Generate a more contextually appropriate fallback response.
        
        Args:
            prompt: The user's message
            emotional_state: Current emotional state
            
        Returns:
            Fallback response
        """
        # Analyze prompt content for better response
        prompt_lower = prompt.lower()
        
        # Question detection
        question_words = ["what", "how", "why", "when", "where", "who", "which", "can", "could", "would", "should"]
        is_question = any(prompt_lower.startswith(word) for word in question_words) or "?" in prompt
        
        if is_question:
            return "That's an interesting question. Could you provide more details so I can give you a more helpful answer?"
        
        # Statement detection with emotional context
        dominant_mood = emotional_state.get("dominant_mood", "Curious")
        
        if dominant_mood == "Curious":
            return "That's fascinating! Could you tell me more about your thoughts on this?"
        elif dominant_mood == "Confident":
            return "I appreciate you sharing that. What would you like to explore next?"
        elif dominant_mood in ["Frustrated", "Anxious"]:
            return "I sense some frustration. Would you like to take a different approach to this topic?"
        else:
            return "Thanks for sharing that with me. What aspects of this would you like to dive deeper into?"

    def process_action_natural(self, action_output: str):
        """
        Process natural language action outputs.
        
        Args:
            action_output: Natural language action output
        """
        self.base_ei.process_action_natural(action_output)
        
    def get_dominant_mood(self) -> str:
        """
        Get the current dominant mood.
        
        Returns:
            String representing the dominant mood
        """
        return self.base_ei.get_dominant_mood()
        
    def get_mood_vector(self) -> Dict[str, float]:
        """
        Get the complete mood vector.
        
        Returns:
            Dictionary of moods and their intensities
        """
        return self.base_ei.get_mood_vector()
        
    def update_user_emotional_profile(self, user_id: str, interaction_data: Dict[str, Any]):
        """
        Update the user's emotional profile.
        
        Args:
            user_id: Unique identifier for the user
            interaction_data: Data about the interaction
        """
        # This would typically store emotional data in the user's profile
        # For now, we'll just log that it was called
        logger.debug(f"Updating emotional profile for user {user_id}")
        
    def adapt_to_user(self, user_id: str, user_profile: Dict[str, Any]):
        """
        Adapt emotional responses based on user profile.
        
        Args:
            user_id: Unique identifier for the user
            user_profile: User's profile data
        """
        # Set persona based on user preferences
        preferred_persona = user_profile.get("personality", {}).get("persona", "Balanced")
        self.set_persona(preferred_persona)
        
        # Adjust mood based on user history
        emotional_history = user_profile.get("memory_context", {}).get("emotional_history", [])
        if emotional_history:
            # Simple adaptation - if user has been frustrated in recent interactions,
            # try to be more patient
            recent_frustrations = sum(1 for entry in emotional_history[-5:] 
                                    if entry.get("dominant_mood") == "Frustrated")
            if recent_frustrations > 2:
                self.base_ei.update_mood("Patient", 0.3)