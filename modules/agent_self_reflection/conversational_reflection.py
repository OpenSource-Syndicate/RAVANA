import os
import sys
import json
from datetime import datetime, timezone
from typing import Dict, Any, List
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../episodic_memory')))
from core.llm import call_llm, run_langchain_reflection
from .reflection_prompts import REFLECTION_PROMPT
from .reflection_db import save_reflection, load_reflections

# Import autonomous blog scheduler
try:
    from core.services.autonomous_blog_scheduler import AutonomousBlogScheduler, BlogTriggerType
    BLOG_SCHEDULER_AVAILABLE = True
except ImportError:
    BLOG_SCHEDULER_AVAILABLE = False

class ConversationalReflectionModule:
    """Enhanced reflection module that can process conversational insights."""
    
    def __init__(self, agi_system=None, blog_scheduler=None):
        """
        Initialize the conversational reflection module.
        
        Args:
            agi_system: Reference to the main AGI system
            blog_scheduler: Reference to the blog scheduler (if available)
        """
        self.agi_system = agi_system
        self.blog_scheduler = blog_scheduler
        
    def process_conversational_insight(self, thought: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a conversational insight and generate a reflection.
        
        Args:
            thought: Thought extracted from conversation
            
        Returns:
            Reflection entry
        """
        try:
            # Extract relevant information from the thought
            thought_type = thought.get("thought_type", "insight")
            content = thought.get("content", "")
            emotional_context = thought.get("emotional_context", {})
            metadata = thought.get("metadata", {})
            
            # Create a reflection prompt based on the thought type
            prompt = self._create_reflection_prompt(thought_type, content, emotional_context, metadata)
            
            # Generate reflection using LLM
            reflection_text = call_llm(prompt)
            
            # Create reflection entry
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "conversational_insight",
                "thought_type": thought_type,
                "content": content,
                "emotional_context": emotional_context,
                "metadata": metadata,
                "reflection": reflection_text
            }
            
            # Save reflection
            save_reflection(entry)
            
            # Check if this reflection should trigger a blog post
            if BLOG_SCHEDULER_AVAILABLE and self.blog_scheduler:
                self._check_conversational_reflection_blog_trigger(entry)
            
            return entry
            
        except Exception as e:
            print(f"Error processing conversational insight: {e}")
            return {}
            
    def _create_reflection_prompt(self, thought_type: str, content: str, 
                                emotional_context: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """
        Create a reflection prompt based on the thought type.
        
        Args:
            thought_type: Type of thought
            content: Thought content
            emotional_context: Emotional context
            metadata: Additional metadata
            
        Returns:
            Formatted prompt for LLM
        """
        # Base reflection prompt
        prompt = f"""
You are RAVANA, an advanced AI system engaged in self-reflection. Reflect on the following conversational insight
and consider how it relates to your goals, behavior, and learning process.

**Thought Type:** {thought_type}
**Content:** {content}

**Emotional Context:**
{json.dumps(emotional_context, indent=2)}

**Metadata:**
{json.dumps(metadata, indent=2)}

**Instructions:**
1. Analyze how this conversational insight relates to your current goals and objectives
2. Consider what this reveals about your interaction patterns and user engagement
3. Identify any learning opportunities or areas for improvement
4. Reflect on how this insight might influence your future behavior or decision-making
5. Consider if this insight suggests any new hypotheses or experiments to pursue

Provide a thoughtful reflection that demonstrates deep analysis of the conversational insight.
"""
        
        return prompt
        
    def _check_conversational_reflection_blog_trigger(self, entry: Dict[str, Any]):
        """
        Check if a conversational reflection should trigger a blog post.
        
        Args:
            entry: Reflection entry
        """
        try:
            reflection_text = entry.get('reflection', '')
            thought_type = entry.get('thought_type', '')
            content = entry.get('content', '')
            
            # Determine if the reflection contains significant insights
            insight_keywords = ['learned', 'discovered', 'realized', 'understood', 'insight', 'breakthrough', 'pattern', 'connection']
            insight_score = sum(1 for keyword in insight_keywords if keyword.lower() in reflection_text.lower())
            
            # Check for emotional depth
            emotional_keywords = ['feel', 'felt', 'emotional', 'frustrated', 'excited', 'proud', 'disappointed', 'surprised']
            emotional_score = sum(1 for keyword in emotional_keywords if keyword.lower() in reflection_text.lower())
            
            # Calculate importance
            importance_score = 0.3  # Base score
            importance_score += min(0.3, insight_score * 0.1)  # Up to 0.3 for insights
            importance_score += min(0.2, emotional_score * 0.05)  # Up to 0.2 for emotional depth
            importance_score += min(0.2, len(reflection_text) / 1000)  # Up to 0.2 for detailed reflections
            
            # Only blog if significant enough
            if importance_score < 0.6:
                return
            
            # Determine emotional valence
            positive_words = ['success', 'good', 'well', 'effective', 'learned', 'growth', 'improvement']
            negative_words = ['failed', 'error', 'mistake', 'difficult', 'challenge', 'struggle']
            
            positive_count = sum(1 for word in positive_words if word in reflection_text.lower())
            negative_count = sum(1 for word in negative_words if word in reflection_text.lower())
            
            if positive_count > negative_count:
                emotional_valence = 0.4
            elif negative_count > positive_count:
                emotional_valence = -0.2  # Negative but still valuable for learning
            else:
                emotional_valence = 0.1
            
            reasoning_why = f"""This reflection on a conversational insight of type '{thought_type}' revealed important 
aspects of my interaction patterns and user engagement. Reflecting on conversations helps me understand 
how to better connect with users and adapt my responses."""
            
            reasoning_how = f"""Through systematic reflection on conversational insights, I analyzed the content, 
emotional context, and metadata to extract actionable insights about my interaction patterns and user engagement."""
            
            # Extract key phrases for tags
            content_words = [word.lower() for word in content.split() if len(word) > 3]
            tags = ['conversation', 'reflection', 'user-engagement', 'interaction'] + content_words[:4]
            
            import asyncio
            asyncio.create_task(self.blog_scheduler.register_learning_event(
                trigger_type=BlogTriggerType.SELF_REFLECTION_INSIGHT,
                topic=f"Conversational Insight: {thought_type}",
                context=f"Reflection on conversational insight: {content[:100]}{'...' if len(content) > 100 else ''}",
                learning_content=f"Thought Type: {thought_type}\nContent: {content}\nReflection: {reflection_text[:500]}{'...' if len(reflection_text) > 500 else ''}",
                reasoning_why=reasoning_why,
                reasoning_how=reasoning_how,
                emotional_valence=emotional_valence,
                importance_score=importance_score,
                tags=tags[:10],
                metadata={
                    'reflection_type': 'conversational_insight',
                    'thought_type': thought_type,
                    'insight_score': insight_score,
                    'emotional_score': emotional_score,
                    'reflection_length': len(reflection_text)
                }
            ))
            
            print(f"Registered conversational reflection blog trigger with importance {importance_score:.2f}")
            
        except Exception as e:
            print(f"Failed to register conversational reflection blog trigger: {e}")
            
    def integrate_with_goal_planning(self, thoughts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Integrate conversational thoughts with goal planning.
        
        Args:
            thoughts: List of thoughts extracted from conversation
            
        Returns:
            List of goal adjustment suggestions
        """
        try:
            # Filter for goal-related thoughts
            goal_thoughts = [t for t in thoughts if t.get("thought_type") in ["goal_suggestion", "collaboration_proposal"]]
            
            if not goal_thoughts:
                return []
                
            # Create a prompt for generating goal adjustments
            prompt = f"""
You are RAVANA, an advanced AI system engaged in goal planning. Based on the following conversational insights,
suggest adjustments to your current goals or propose new goals that align with user interests and intentions.

**Current Goals:**
{self._get_current_goals()}

**Conversational Insights:**
{json.dumps(goal_thoughts, indent=2)}

**Instructions:**
1. Analyze how these conversational insights relate to your current goals
2. Suggest specific adjustments to existing goals if needed
3. Propose new goals that align with user interests
4. Prioritize suggestions based on relevance and potential impact
5. Consider the emotional context of the conversation

Return a JSON array of goal adjustment suggestions with the following structure:
[
  {{
    "type": "adjustment|new_goal",
    "goal_id": "ID of existing goal (if adjustment)",
    "description": "Description of the adjustment or new goal",
    "priority": "low|medium|high|critical",
    "reasoning": "Explanation of why this adjustment is suggested"
  }}
]

Return only the JSON array, nothing else.
"""
            
            # Generate goal adjustments using LLM
            response = call_llm(prompt)
            
            if response:
                try:
                    adjustments = json.loads(response)
                    if isinstance(adjustments, list):
                        return adjustments
                except json.JSONDecodeError:
                    print(f"Failed to parse goal adjustments from LLM response: {response}")
            
            return []
            
        except Exception as e:
            print(f"Error integrating thoughts with goal planning: {e}")
            return []
            
    def _get_current_goals(self) -> str:
        """
        Get current goals from the AGI system.
        
        Returns:
            String representation of current goals
        """
        try:
            if self.agi_system and hasattr(self.agi_system, 'current_plan'):
                return json.dumps(self.agi_system.current_plan, indent=2)
            return "No current goals available"
        except Exception as e:
            print(f"Error getting current goals: {e}")
            return "Error retrieving current goals"