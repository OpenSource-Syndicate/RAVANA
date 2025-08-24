import json
import logging
import traceback
from typing import Dict, List, Any

# Import required modules
from core.llm import safe_call_llm

logger = logging.getLogger(__name__)


class ConversationalEmotionalIntelligence:
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
            
    async def extract_thoughts_from_conversation_async(self, user_message: str, ai_response: str, 
                                                     emotional_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract meaningful thoughts and insights from the conversation asynchronously.
        
        Args:
            user_message: The user's message
            ai_response: The AI's response
            emotional_context: Current emotional context
            
        Returns:
            List of extracted thoughts as structured dictionaries
        """
        try:
            # Import async LLM function
            from core.llm import async_safe_call_llm
            
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
            response = await async_safe_call_llm(extraction_prompt, timeout=30, retries=3)
            
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
            
    async def generate_response_async(self, prompt: str, emotional_state: Dict[str, Any]) -> str:
        """
        Generate an emotionally-aware response using async LLM calls.
        
        Args:
            prompt: The user's message
            emotional_state: Current emotional state
            
        Returns:
            Generated response
        """
        try:
            # Import async LLM function
            from core.llm import async_safe_call_llm
            
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
            response = await async_safe_call_llm(llm_prompt, timeout=30, retries=3)
            
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
        Generate a fallback response when LLM fails or returns inappropriate content.
        
        Args:
            prompt: The user's message
            emotional_state: Current emotional state
            
        Returns:
            Generated fallback response
        """
        try:
            dominant_mood = emotional_state.get("dominant_mood", "Curious")
            interests = emotional_state.get("detected_interests", [])
            
            # Simple rule-based fallback responses based on message type and mood
            if "?" in prompt:
                # Question-based responses
                if dominant_mood == "Curious":
                    return "That's an interesting question. I'd love to explore that topic with you further."
                elif dominant_mood == "Helpful":
                    return "I'd be happy to help you with that. Could you tell me more about what you're looking for?"
                else:
                    return "I'm not certain about that, but I'm curious to learn more. What aspects interest you most?"
            else:
                # Statement-based responses
                if interests:
                    return f"I find that fascinating, especially in relation to {interests[0] if interests else 'your interests'}. What aspects of this topic intrigue you the most?"
                else:
                    return "That's really interesting. I'd love to dive deeper into this with you. What would you like to explore further?"
        except Exception as e:
            logger.error(f"Error generating fallback response: {e}")
            # Ultimate fallback
            return "I'm processing that thought. Could you share a bit more about what you're considering?"

    def process_user_message(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a user message and update emotional context.
        
        Args:
            message: The user's message
            context: Current context
            
        Returns:
            Updated emotional context
        """
        # Simple emotional context generation (in a real implementation, this would be more complex)
        emotional_context = {
            "dominant_mood": "Curious",
            "mood_vector": {"Curious": 0.8, "Engaged": 0.7, "Helpful": 0.6},
            "detected_interests": self._detect_user_interests(message),
            "recent_events": []
        }
        
        return emotional_context

    def _detect_user_interests(self, message: str) -> List[str]:
        """
        Detect user interests from their message.
        
        Args:
            message: The user's message
            
        Returns:
            List of detected interests
        """
        # Simple keyword-based interest detection (in a real implementation, this would be more sophisticated)
        interests = []
        message_lower = message.lower()
        
        # Technology-related keywords
        tech_keywords = ["ai", "artificial intelligence", "machine learning", "programming", "code", "algorithm", "data"]
        if any(keyword in message_lower for keyword in tech_keywords):
            interests.append("technology")
            
        # Philosophy-related keywords
        philosophy_keywords = ["philosophy", "consciousness", "meaning", "ethics", "morality"]
        if any(keyword in message_lower for keyword in philosophy_keywords):
            interests.append("philosophy")
            
        # Science-related keywords
        science_keywords = ["physics", "chemistry", "biology", "science", "research", "experiment"]
        if any(keyword in message_lower for keyword in science_keywords):
            interests.append("science")
            
        return interests

    def set_persona(self, persona: str):
        """
        Set the persona for the emotional intelligence module.
        
        Args:
            persona: The persona to set
        """
        # In a real implementation, this would configure the emotional responses based on persona
        pass