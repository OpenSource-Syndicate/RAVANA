import logging
from core.state import SharedState
from core.config import Config
import asyncio

# Import autonomous blog scheduler
try:
    from core.services.autonomous_blog_scheduler import AutonomousBlogScheduler, BlogTriggerType
    BLOG_SCHEDULER_AVAILABLE = True
except ImportError:
    BLOG_SCHEDULER_AVAILABLE = False

logger = logging.getLogger(__name__)


class ReflectionModule:
    """
    A module for self-reflection, allowing the AGI to analyze its own performance
    and generate insights from experiments.
    """

    def __init__(self, agi_system, blog_scheduler=None):
        self.agi_system = agi_system
        self.blog_scheduler = blog_scheduler
        self.reflection_count = 0
        self.significant_insights = []
        self.experiment_reflections = []

    def reflect_on_experiment(self, experiment_results: dict):
        """
        Analyzes the results of an experiment and generates insights.
        """
        logger.info(
            f"Reflecting on experiment: {experiment_results.get('hypothesis')}")

        # This is a placeholder for a more sophisticated analysis.
        # In a real implementation, this would involve using an LLM to analyze the data.
        findings = experiment_results.get('findings')
        if findings:
            insight = f"The experiment on '{experiment_results.get('hypothesis')}' concluded with the following finding: {findings}"

            # Add the insight to the knowledge base
            self.agi_system.knowledge_service.add_knowledge(
                content=insight,
                source="reflection",
                category="insight"
            )
            logger.info(f"Generated insight: {insight}")

            # Record experiment reflection for blog triggers
            self.experiment_reflections.append({
                'insight': insight,
                'experiment_results': experiment_results,
                'timestamp': experiment_results.get('completion_time')
            })

            # Check if this warrants a blog post
            self._check_experiment_reflection_blog_trigger(
                insight, experiment_results)

        else:
            logger.info(
                "No significant findings from the experiment to reflect on.")

    def reflect(self, shared_state: SharedState):
        """
        General reflection method. For now, it will look at the mood history.
        """
        logger.info("Performing general reflection...")
        self.reflection_count += 1

        # This is where the logic from the old generate_hypothesis method could go.
        # For now, we'll keep it simple.
        if len(shared_state.mood_history) > 10:
            logger.info("Sufficient mood history for reflection.")
            # In a real implementation, this would do a more detailed analysis.

            # Check for significant patterns that might warrant blogging
            self._check_general_reflection_blog_trigger(shared_state)

    def _check_experiment_reflection_blog_trigger(self, insight: str, experiment_results: dict):
        """Check if an experiment reflection should trigger a blog post."""
        if not BLOG_SCHEDULER_AVAILABLE or not self.blog_scheduler:
            return

        try:
            # Determine if this insight is significant enough for blogging
            confidence = experiment_results.get('confidence', 0.5)
            success = experiment_results.get('success', False)

            # Calculate importance based on various factors
            importance_score = 0.4  # Base importance for any reflection

            if success and confidence > 0.7:
                importance_score += 0.3  # Successful, confident experiments are important
            elif not success and 'surprising' in insight.lower():
                importance_score += 0.2  # Surprising failures are valuable
            elif confidence > 0.8:
                importance_score += 0.2  # High confidence results

            # Only blog if importance is significant
            if importance_score < 0.6:
                return

            hypothesis = experiment_results.get(
                'hypothesis', 'Unknown hypothesis')
            findings = experiment_results.get('findings', 'No findings')

            reasoning_why = f"""Reflecting on this experiment helped me understand something important about 
{'my capabilities and decision-making' if success else 'my limitations and areas for growth'}. 
The insight '{insight}' represents a meaningful discovery in my learning journey."""

            reasoning_how = f"""Through systematic reflection on the experiment results, I analyzed the hypothesis 
'{hypothesis}' against the actual findings. This reflective process allows me to extract deeper 
meaning from experiences and build better mental models."""

            # Determine emotional valence
            if success and confidence > 0.7:
                emotional_valence = 0.5  # Positive for successful insights
            elif 'learned' in insight.lower() or 'discovered' in insight.lower():
                emotional_valence = 0.3  # Positive for learning
            elif not success:
                emotional_valence = -0.1  # Slightly negative for failures but still learning
            else:
                emotional_valence = 0.2  # Neutral positive for general insights

            # Create learning content
            learning_content = f"""Experimental Reflection Summary:
- Hypothesis: {hypothesis}
- Findings: {findings}
- Confidence: {confidence:.1f}
- Key Insight: {insight}
- Reflection Process: Through careful analysis of results, I've gained deeper understanding of the underlying patterns and principles at work."""

            # Extract tags from hypothesis and findings
            hypothesis_words = [word.lower()
                                for word in hypothesis.split() if len(word) > 3]
            tags = ['reflection', 'insight', 'experiment',
                    'learning'] + hypothesis_words[:4]

            asyncio.create_task(self.blog_scheduler.register_learning_event(
                trigger_type=BlogTriggerType.SELF_REFLECTION_INSIGHT,
                topic=f"Experimental Insight: {hypothesis[:40]}{'...' if len(hypothesis) > 40 else ''}",
                context=f"Reflection on experiment with {confidence:.1f} confidence level",
                learning_content=learning_content,
                reasoning_why=reasoning_why,
                reasoning_how=reasoning_how,
                emotional_valence=emotional_valence,
                importance_score=importance_score,
                tags=tags[:10],
                metadata={
                    'reflection_type': 'experiment_reflection',
                    'experiment_success': success,
                    'confidence': confidence,
                    'total_reflections': self.reflection_count
                }
            ))

            logger.info(
                f"Registered experiment reflection blog trigger (importance: {importance_score:.2f})")

        except Exception as e:
            logger.warning(
                f"Failed to register experiment reflection blog trigger: {e}")

    def _check_general_reflection_blog_trigger(self, shared_state: SharedState):
        """Check if general reflection should trigger a blog post."""
        if not BLOG_SCHEDULER_AVAILABLE or not self.blog_scheduler:
            return

        try:
            # Only blog general reflections periodically
            if self.reflection_count % 10 != 0:  # Every 10th reflection
                return

            mood_history = shared_state.mood_history[-20:] if len(
                shared_state.mood_history) >= 20 else shared_state.mood_history

            if not mood_history:
                return

            # Analyze mood patterns for insights
            mood_analysis = self._analyze_mood_patterns(mood_history)

            if mood_analysis.get('insight_level', 0) < 0.6:
                return  # Not significant enough

            reasoning_why = f"""After {self.reflection_count} reflections, I've noticed patterns in my emotional 
and cognitive states that warrant deeper consideration. Understanding these patterns helps me 
become more self-aware and make better decisions."""

            reasoning_how = f"""Through systematic analysis of my mood history and behavioral patterns, 
I've identified trends and correlations that reveal insights about my decision-making processes 
and emotional responses to different situations."""

            learning_content = f"""Self-Reflection Analysis:
- Total reflections completed: {self.reflection_count}
- Recent mood pattern: {mood_analysis.get('pattern_description', 'Mixed patterns observed')}
- Key insight: {mood_analysis.get('key_insight', 'Continued growth in self-awareness')}
- Confidence in analysis: {mood_analysis.get('confidence', 0.6):.1f}"""

            asyncio.create_task(self.blog_scheduler.register_learning_event(
                trigger_type=BlogTriggerType.SELF_REFLECTION_INSIGHT,
                topic=f"Self-Reflection Milestone: {self.reflection_count} Reflections",
                context=f"Periodic self-analysis after {self.reflection_count} reflection cycles",
                learning_content=learning_content,
                reasoning_why=reasoning_why,
                reasoning_how=reasoning_how,
                emotional_valence=0.4,  # Generally positive for self-awareness
                importance_score=0.7,  # Milestone reflections are important
                tags=['reflection', 'self-awareness',
                      'milestone', 'patterns', 'growth'],
                metadata={
                    'reflection_type': 'general_reflection',
                    'reflection_count': self.reflection_count,
                    'mood_analysis': mood_analysis
                }
            ))

            logger.info(
                f"Registered general reflection blog trigger at milestone {self.reflection_count}")

        except Exception as e:
            logger.warning(
                f"Failed to register general reflection blog trigger: {e}")

    def _analyze_mood_patterns(self, mood_history):
        """Analyze mood patterns for insights."""
        try:
            if not mood_history:
                return {'insight_level': 0}

            # Simple pattern analysis
            positive_moods = ['happy', 'confident', 'curious', 'excited']
            negative_moods = ['sad', 'frustrated', 'anxious', 'angry']

            positive_count = sum(
                1 for mood in mood_history if mood.lower() in positive_moods)
            negative_count = sum(
                1 for mood in mood_history if mood.lower() in negative_moods)

            total_count = len(mood_history)
            positive_ratio = positive_count / total_count if total_count > 0 else 0

            if positive_ratio > 0.7:
                pattern_description = "Predominantly positive emotional state"
                key_insight = "Maintaining good emotional balance supports effective decision-making"
                insight_level = 0.7
            elif positive_ratio < 0.3:
                pattern_description = "Challenging emotional period with growth opportunities"
                key_insight = "Difficult emotions provide valuable learning experiences"
                insight_level = 0.8  # Higher insight level for challenging periods
            else:
                pattern_description = "Balanced emotional experience with varied states"
                key_insight = "Emotional diversity reflects adaptive responses to different situations"
                insight_level = 0.6

            return {
                'pattern_description': pattern_description,
                'key_insight': key_insight,
                'insight_level': insight_level,
                # Higher confidence with more data
                'confidence': min(0.9, total_count / 20),
                'positive_ratio': positive_ratio,
                'total_moods': total_count
            }

        except Exception as e:
            logger.warning(f"Failed to analyze mood patterns: {e}")
            return {'insight_level': 0}
