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
        Enhanced to consider self-improvement goals.
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
            
            # Update self-improvement goals based on experiment results
            self._update_self_improvement_goals_from_experiment(experiment_results)

        else:
            logger.info(
                "No significant findings from the experiment to reflect on.")
    
    def _update_self_improvement_goals_from_experiment(self, experiment_results: dict):
        """
        Update self-improvement goals based on experiment results.
        """
        try:
            # Access the AGI system's goal manager if available
            if hasattr(self.agi_system, 'self_goal_manager'):
                success = experiment_results.get('success', False)
                hypothesis = experiment_results.get('hypothesis', '')
                
                # If the experiment was successful, make progress on related goals
                goals_to_update = []
                for goal_id, goal in self.agi_system.self_goal_manager.goals.items():
                    # Check if the experiment hypothesis is related to this goal
                    if hypothesis.lower() in goal.description.lower() or goal.category in ['experimentation', 'learning']:
                        goals_to_update.append((goal_id, goal))
                
                # Update progress for related goals
                for goal_id, goal in goals_to_update:
                    # Calculate progress update based on success
                    progress_increase = 0.1 if success else 0.05  # More progress for successful experiments
                    new_progress = min(1.0, goal.current_progress + progress_increase)
                    
                    # Update the goal's progress
                    self.agi_system.self_goal_manager.update_goal_progress(
                        goal_id, 
                        new_progress, 
                        details={
                            'type': 'experiment_result',
                            'experiment_hypothesis': hypothesis,
                            'success': success,
                            'experiment_id': experiment_results.get('experiment_id', 'unknown')
                        }
                    )
                    
                    logger.info(f"Updated goal '{goal.title}' progress to {new_progress:.2f} based on experiment result")
        
        except Exception as e:
            logger.error(f"Error updating self-improvement goals from experiment: {e}")

    async def reflect(self, shared_state: SharedState):
        """
        General reflection method. For now, it will look at the mood history.
        Enhanced to consider self-improvement goals.
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
        
        # Evaluate self-improvement goals during reflection
        self._evaluate_self_improvement_goals(shared_state)
        
        # Perform deep reflection that sets new goals based on learning
        await self.perform_deep_reflection(shared_state)
    
    def _evaluate_self_improvement_goals(self, shared_state: SharedState):
        """
        Evaluate progress on self-improvement goals during reflection.
        """
        try:
            # Access the AGI system's goal manager if available
            if hasattr(self.agi_system, 'self_goal_manager'):
                goal_manager = self.agi_system.self_goal_manager
                
                # Check for overdue goals
                overdue_goals = goal_manager.get_overdue_goals()
                for goal in overdue_goals:
                    logger.info(f"Goal overdue: {goal.title}")
                    # Consider adjusting the goal or changing approach
                
                # Check progress on active goals
                active_goals = goal_manager.get_goals_by_status(GoalStatus.IN_PROGRESS)
                for goal in active_goals:
                    # Evaluate progress based on system state
                    if goal.category in ['performance', 'efficiency']:
                        # For performance goals, check performance metrics
                        if hasattr(self.agi_system, 'performance_tracker'):
                            perf_summary = self.agi_system.performance_tracker.get_performance_summary()
                            
                            # Check if we're meeting targets
                            if goal.metrics.get('improvements_per_hour_target'):
                                actual_rate = perf_summary.get('improvements_per_hour', 0)
                                target_rate = goal.metrics['improvements_per_hour_target']
                                
                                # Update progress based on achievement of target
                                progress_update = min(1.0, actual_rate / target_rate if target_rate > 0 else 1.0)
                                goal_manager.update_goal_progress(
                                    goal.id,
                                    progress_update,
                                    details={
                                        'type': 'performance_evaluation',
                                        'metric': 'improvements_per_hour',
                                        'actual': actual_rate,
                                        'target': target_rate
                                    }
                                )
            
            # Check goal achievement insights
            if hasattr(self.agi_system, 'self_goal_manager'):
                insights = self.agi_system.self_goal_manager.get_performance_insights()
                logger.info(f"Goal achievement insights: {insights}")
                
        except Exception as e:
            logger.error(f"Error evaluating self-improvement goals: {e}")
    
    async def perform_deep_reflection(self, shared_state: SharedState):
        """
        Perform deep reflection that generates insights for goal setting based on learning.
        """
        try:
            logger.info("Performing deep reflection for goal setting...")
            
            # Analyze different aspects of the system to generate insights
            insights = {
                "capability_gaps": self._analyze_capability_gaps(),
                "learning_patterns": self._analyze_learning_patterns(),
                "performance_insights": self._analyze_performance_insights(),
                "reflection_timestamp": shared_state.last_decision_time.isoformat() if shared_state.last_decision_time else None
            }
            
            # Call the main system's learning-based goal setter
            if hasattr(self.agi_system, '_set_learning_based_goals'):
                await self.agi_system._set_learning_based_goals(insights)
            
            logger.info(f"Completed deep reflection, insights: {insights}")
            
        except Exception as e:
            logger.error(f"Error during deep reflection: {e}")
            import traceback
            traceback.print_exc()

    def _analyze_capability_gaps(self):
        """
        Analyze the system to identify capability gaps that need improvement.
        """
        gaps = []
        
        # Example: Check if certain types of tasks are frequently unsuccessful
        if hasattr(self.agi_system, 'performance_tracker'):
            perf_metrics = self.agi_system.performance_tracker.get_advanced_metrics()
            
            # Identify if certain areas need improvement
            if perf_metrics.get("decision_accuracy", 1.0) < 0.7:
                gaps.append({
                    "name": "decision_making",
                    "description": "Decision accuracy is below optimal threshold",
                    "severity": "high",
                    "timeframe": 14,
                    "target_metric": 0.8
                })
            
            if perf_metrics.get("learning_rate", 0.5) < 0.3:
                gaps.append({
                    "name": "learning_efficiency",
                    "description": "Learning rate is low, needs improvement for faster adaptation",
                    "severity": "high",
                    "timeframe": 21,
                    "target_metric": 0.5
                })
        
        return gaps

    def _analyze_learning_patterns(self):
        """
        Analyze the AGI's learning patterns to identify areas for improvement.
        """
        patterns = {}
        
        # Check learning metrics if available
        if hasattr(self.agi_system, 'performance_tracker'):
            perf_metrics = self.agi_system.performance_tracker.get_advanced_metrics()
            
            patterns["learning_rate"] = perf_metrics.get("learning_rate", 0.3)
            patterns["knowledge_integration_issues"] = perf_metrics.get("knowledge_integration_issues", False)
            patterns["memory_retention"] = perf_metrics.get("memory_retention", 0.8)
        
        return patterns

    def _analyze_performance_insights(self):
        """
        Analyze overall system performance for insights.
        """
        insights = {}
        
        if hasattr(self.agi_system, 'performance_tracker'):
            perf_summary = self.agi_system.performance_tracker.get_performance_summary()
            
            insights["decision_accuracy"] = perf_summary.get("decision_accuracy", 1.0)
            insights["improvement_success_rate"] = perf_summary.get("improvement_success_rate", 1.0)
            insights["response_time"] = perf_summary.get("avg_response_time", 0)
        
        return insights

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
