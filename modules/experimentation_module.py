import logging
from typing import Dict, Any

# Import autonomous blog scheduler
try:
    from core.services.autonomous_blog_scheduler import AutonomousBlogScheduler, BlogTriggerType
    BLOG_SCHEDULER_AVAILABLE = True
except ImportError:
    BLOG_SCHEDULER_AVAILABLE = False

logger = logging.getLogger(__name__)

class ExperimentationModule:
    """
    A module for designing and running experiments to validate hypotheses.
    """
    def __init__(self, agi_system, blog_scheduler=None):
        self.agi_system = agi_system
        self.blog_scheduler = blog_scheduler
        self.completed_experiments = []
        self.experiment_count = 0

    def design_and_run_experiment(self, hypothesis: str, shared_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Designs an experiment and passes it to the experimentation engine.
        """
        logger.info(f"Designing experiment for hypothesis: {hypothesis}")
        
        self.experiment_count += 1
        experiment_id = f"exp_{self.experiment_count:03d}"
        
        experiment_plan = {
            "experiment_id": experiment_id,
            "hypothesis": hypothesis,
            "plan": [
                {"step": 1, "action": "set_mood", "parameters": {"mood": "Confident"}},
                {"step": 2, "action": "run_task", "parameters": {"prompt": "Create a plan to launch a new product."}},
                {"step": 3, "action": "record_metric", "parameters": {"name": "plan_quality"}},
                {"step": 4, "action": "set_mood", "parameters": {"mood": "Frustrated"}},
                {"step": 5, "action": "run_task", "parameters": {"prompt": "Create a plan to launch a new product."}},
                {"step": 6, "action": "record_metric", "parameters": {"name": "plan_quality"}},
                {"step": 7, "action": "analyze_results"}
            ],
            "start_time": None,
            "end_time": None,
            "context": shared_state
        }
        
        self.agi_system.experimentation_engine.start_experiment(experiment_plan)
        
        return {"status": "started", "plan": experiment_plan}
    
    async def complete_experiment(self, experiment_id: str, results: Dict[str, Any]):
        """Handle experiment completion and trigger blog post if appropriate."""
        try:
            # Record completed experiment
            experiment_record = {
                'experiment_id': experiment_id,
                'completion_time': results.get('completion_time'),
                'hypothesis': results.get('hypothesis'),
                'findings': results.get('findings'),
                'success': results.get('success', False),
                'confidence': results.get('confidence', 0.5),
                'context': results.get('context', {})
            }
            
            self.completed_experiments.append(experiment_record)
            
            # Trigger blog post for significant experiments
            await self._register_experiment_completion_blog_trigger(experiment_record)
            
            logger.info(f"Completed experiment {experiment_id}: {results.get('findings', 'No findings')}")
            
        except Exception as e:
            logger.error(f"Failed to complete experiment {experiment_id}: {e}")

    def run_experiment_from_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Designs and runs an experiment from a given prompt.
        """
        # This would use an LLM to generate a hypothesis and plan from the prompt.
        # For now, we'll use a hardcoded example.
        hypothesis = "A positive mood improves planning ability."
        return self.design_and_run_experiment(hypothesis, {})
    
    async def _register_experiment_completion_blog_trigger(self, experiment_record: Dict[str, Any]):
        """Register a blog trigger for experiment completion."""
        if not BLOG_SCHEDULER_AVAILABLE or not self.blog_scheduler:
            return
        
        try:
            experiment_id = experiment_record.get('experiment_id', 'unknown')
            hypothesis = experiment_record.get('hypothesis', 'Unknown hypothesis')
            findings = experiment_record.get('findings', 'No findings recorded')
            success = experiment_record.get('success', False)
            confidence = experiment_record.get('confidence', 0.5)
            
            # Determine importance based on experiment outcomes
            importance_score = 0.5  # Base importance
            
            # Increase importance for successful experiments
            if success:
                importance_score += 0.2
            
            # Increase importance for high-confidence results
            importance_score += confidence * 0.2
            
            # Increase importance for surprising or counterintuitive results
            if 'surprising' in findings.lower() or 'unexpected' in findings.lower():
                importance_score += 0.1
            
            # Cap importance
            importance_score = min(0.9, importance_score)
            
            # Create reasoning
            reasoning_why = f"""I completed experiment '{experiment_id}' to test the hypothesis: '{hypothesis}'. 
Experimentation is crucial for validating assumptions and advancing understanding. 
{'The successful outcome' if success else 'Even though the results were inconclusive,'} this experiment 
provides valuable insights for future decision-making."""
            
            reasoning_how = f"""This experiment was conducted systematically through my experimentation engine. 
I followed a structured plan with measurable steps, recorded metrics, and analyzed the results. 
The confidence level of {confidence:.1f} reflects the reliability of the findings."""
            
            # Determine emotional valence
            if success and confidence > 0.7:
                emotional_valence = 0.6  # Positive for successful experiments
            elif success:
                emotional_valence = 0.3  # Mildly positive
            elif 'failed' in findings.lower():
                emotional_valence = -0.2  # Slightly negative for failures
            else:
                emotional_valence = 0.1  # Neutral for inconclusive
            
            # Create learning content
            learning_content = f"""Experiment Results Summary:
- Experiment ID: {experiment_id}
- Hypothesis: {hypothesis}
- Outcome: {'Success' if success else 'Inconclusive/Failed'}
- Confidence: {confidence:.1f}
- Key Findings: {findings}
- Learning Value: This experiment {'validated' if success else 'challenged'} my assumptions and {'confirmed' if success else 'revealed gaps in'} my understanding."""
            
            # Extract tags from hypothesis and findings
            hypothesis_words = [word.lower() for word in hypothesis.split() if len(word) > 3]
            findings_words = [word.lower() for word in findings.split() if len(word) > 3]
            tags = ['experiment', 'research', 'hypothesis', 'findings'] + hypothesis_words[:3] + findings_words[:2]
            
            await self.blog_scheduler.register_learning_event(
                trigger_type=BlogTriggerType.EXPERIMENT_COMPLETION,
                topic=f"Experiment Results: {hypothesis[:50]}{'...' if len(hypothesis) > 50 else ''}",
                context=f"Systematic experiment {experiment_id} completed with {confidence:.1f} confidence",
                learning_content=learning_content,
                reasoning_why=reasoning_why,
                reasoning_how=reasoning_how,
                emotional_valence=emotional_valence,
                importance_score=importance_score,
                tags=tags[:10],  # Limit tags
                metadata={
                    'experiment_id': experiment_id,
                    'success': success,
                    'confidence': confidence,
                    'hypothesis': hypothesis,
                    'total_experiments': len(self.completed_experiments)
                }
            )
            
            logger.info(f"Registered experiment completion blog trigger for {experiment_id} (importance: {importance_score:.2f})")
            
        except Exception as e:
            logger.warning(f"Failed to register experiment completion blog trigger: {e}")
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get a summary of completed experiments."""
        try:
            if not self.completed_experiments:
                return {'total_experiments': 0, 'message': 'No experiments completed yet'}
            
            successful_experiments = [exp for exp in self.completed_experiments if exp.get('success', False)]
            avg_confidence = sum(exp.get('confidence', 0) for exp in self.completed_experiments) / len(self.completed_experiments)
            
            return {
                'total_experiments': len(self.completed_experiments),
                'successful_experiments': len(successful_experiments),
                'success_rate': len(successful_experiments) / len(self.completed_experiments),
                'average_confidence': avg_confidence,
                'recent_experiments': self.completed_experiments[-5:],  # Last 5
                'blog_triggers_created': len([exp for exp in self.completed_experiments if exp.get('confidence', 0) > 0.6])
            }
            
        except Exception as e:
            logger.error(f"Failed to get experiment summary: {e}")
            return {'error': str(e)}