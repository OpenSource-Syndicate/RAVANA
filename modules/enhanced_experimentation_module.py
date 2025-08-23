import logging
from typing import Dict, Any, List
import json
import re

# Import autonomous blog scheduler
try:
    from core.services.autonomous_blog_scheduler import AutonomousBlogScheduler, BlogTriggerType
    BLOG_SCHEDULER_AVAILABLE = True
except ImportError:
    BLOG_SCHEDULER_AVAILABLE = False

# Import LLM module for hypothesis generation
from core.llm import call_llm, safe_call_llm

logger = logging.getLogger(__name__)

class EnhancedExperimentationModule:
    """
    An enhanced module for designing and running experiments, including hypothesis generation from conversational insights.
    """
    def __init__(self, agi_system, blog_scheduler=None):
        self.agi_system = agi_system
        self.blog_scheduler = blog_scheduler
        self.completed_experiments = []
        self.experiment_count = 0
        self.conversational_hypotheses = []  # Store hypotheses generated from conversations

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
    
    async def generate_hypotheses_from_conversation(self, conversational_insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate testable hypotheses from conversational insights.
        
        Args:
            conversational_insights: List of insights extracted from conversations
            
        Returns:
            List of generated hypotheses
        """
        try:
            if not conversational_insights:
                return []
            
            # Filter for insights that might lead to hypotheses
            relevant_insights = [
                insight for insight in conversational_insights 
                if insight.get("thought_type") in ["insight", "knowledge_gap", "reflection_trigger"]
            ]
            
            if not relevant_insights:
                return []
            
            # Create a prompt for hypothesis generation
            prompt = f"""
You are RAVANA, an advanced AI system conducting scientific research. Based on the following conversational insights,
generate testable hypotheses that could be investigated through experiments.

Conversational Insights:
{json.dumps(relevant_insights, indent=2)}

Instructions:
1. Identify patterns or claims in the insights that could be tested
2. Formulate each hypothesis as a clear, testable statement
3. For each hypothesis, suggest a potential experimental approach
4. Estimate the potential impact and confidence level for each hypothesis
5. Consider how these hypotheses relate to your core capabilities

Return a JSON array of hypothesis objects with the following structure:
[
  {{
    "hypothesis": "Clear, testable statement",
    "rationale": "Why this hypothesis is worth testing",
    "experimental_approach": "Brief description of how to test this hypothesis",
    "confidence": 0.0-1.0,
    "potential_impact": 0.0-1.0,
    "related_insights": ["insight_id_1", "insight_id_2"],
    "category": "capability|performance|interaction|learning"
  }}
]

Return only the JSON array, nothing else.
"""
            
            # Generate hypotheses using LLM
            response = safe_call_llm(prompt, timeout=30, retries=3)
            
            if response:
                try:
                    hypotheses = json.loads(response)
                    if isinstance(hypotheses, list):
                        # Add metadata to each hypothesis
                        for i, hypothesis in enumerate(hypotheses):
                            hypothesis["generated_from"] = "conversation"
                            hypothesis["timestamp"] = __import__('datetime').datetime.utcnow().isoformat()
                            hypothesis["hypothesis_id"] = f"conv_hyp_{len(self.conversational_hypotheses) + i + 1:03d}"
                        
                        # Store generated hypotheses
                        self.conversational_hypotheses.extend(hypotheses)
                        
                        logger.info(f"Generated {len(hypotheses)} hypotheses from conversational insights")
                        return hypotheses
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse hypotheses from LLM response: {response}")
            
            return []
            
        except Exception as e:
            logger.error(f"Error generating hypotheses from conversation: {e}")
            return []
    
    async def prioritize_conversational_hypotheses(self) -> List[Dict[str, Any]]:
        """
        Prioritize conversational hypotheses based on impact, confidence, and relevance.
        
        Returns:
            List of prioritized hypotheses
        """
        try:
            if not self.conversational_hypotheses:
                return []
            
            # Sort hypotheses by a combination of impact and confidence
            prioritized = sorted(
                self.conversational_hypotheses,
                key=lambda h: h.get("potential_impact", 0.5) * h.get("confidence", 0.5),
                reverse=True
            )
            
            return prioritized
            
        except Exception as e:
            logger.error(f"Error prioritizing conversational hypotheses: {e}")
            return []
    
    async def design_experiment_from_conversational_hypothesis(self, hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Design an experiment specifically for a conversational hypothesis.
        
        Args:
            hypothesis: Hypothesis generated from conversation
            
        Returns:
            Experiment design dictionary
        """
        try:
            hypothesis_text = hypothesis.get("hypothesis", "")
            experimental_approach = hypothesis.get("experimental_approach", "")
            
            # Create a prompt for detailed experiment design
            design_prompt = f"""
You are RAVANA, an advanced AI system designing scientific experiments. Based on the following hypothesis
generated from conversational insights, create a detailed experimental plan.

Hypothesis: {hypothesis_text}
Suggested Approach: {experimental_approach}

Instructions:
1. Design a controlled experiment with clear variables
2. Define measurable metrics for success
3. Specify the experimental conditions and control conditions
4. Outline the steps needed to conduct the experiment
5. Identify potential confounding factors
6. Estimate the resources and time needed

Return a JSON object with the following structure:
{{
  "experiment_design": {{
    "hypothesis": "The original hypothesis",
    "variables": {{
      "independent": "What you will manipulate",
      "dependent": "What you will measure",
      "controlled": ["List of controlled variables"]
    }},
    "methodology": "Detailed description of the experimental method",
    "metrics": [
      {{
        "name": "metric_name",
        "description": "What this metric measures",
        "measurement_approach": "How to measure this metric"
      }}
    ],
    "steps": [
      {{
        "step": 1,
        "description": "What to do in this step",
        "expected_outcome": "What you expect to happen"
      }}
    ],
    "control_conditions": "Description of control conditions",
    "confounding_factors": ["List of potential confounding factors"],
    "estimated_duration": "Time estimate for the experiment",
    "required_resources": ["List of required resources"]
  }}
}}

Return only the JSON object, nothing else.
"""
            
            # Generate experiment design using LLM
            response = safe_call_llm(design_prompt, timeout=45, retries=3)
            
            if response:
                try:
                    design = json.loads(response)
                    if isinstance(design, dict) and "experiment_design" in design:
                        # Add metadata
                        design["experiment_design"]["source_hypothesis"] = hypothesis
                        design["experiment_design"]["generated_from"] = "conversational_insight"
                        design["experiment_design"]["timestamp"] = __import__('datetime').datetime.utcnow().isoformat()
                        
                        logger.info(f"Designed experiment for conversational hypothesis: {hypothesis_text[:50]}...")
                        return design
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse experiment design from LLM response: {response}")
            
            # Fallback to basic design if LLM approach fails
            return self._create_basic_experiment_design(hypothesis)
            
        except Exception as e:
            logger.error(f"Error designing experiment from conversational hypothesis: {e}")
            # Fallback to basic design
            return self._create_basic_experiment_design(hypothesis)
    
    def _create_basic_experiment_design(self, hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a basic experiment design as a fallback.
        
        Args:
            hypothesis: Hypothesis to design experiment for
            
        Returns:
            Basic experiment design dictionary
        """
        hypothesis_text = hypothesis.get("hypothesis", "Unknown hypothesis")
        
        basic_design = {
            "experiment_design": {
                "hypothesis": hypothesis_text,
                "variables": {
                    "independent": "Condition (with vs. without conversational context)",
                    "dependent": "Performance metric related to hypothesis",
                    "controlled": ["Time of day", "Task complexity", "System configuration"]
                },
                "methodology": f"Test the hypothesis '{hypothesis_text}' by comparing performance with and without conversational context",
                "metrics": [
                    {
                        "name": "performance_score",
                        "description": "Measure of task performance",
                        "measurement_approach": "Quantitative assessment of output quality"
                    }
                ],
                "steps": [
                    {
                        "step": 1,
                        "description": "Establish baseline performance without conversational context",
                        "expected_outcome": "Standard performance metrics recorded"
                    },
                    {
                        "step": 2,
                        "description": "Introduce conversational context and repeat task",
                        "expected_outcome": "Performance metrics with conversational context recorded"
                    },
                    {
                        "step": 3,
                        "description": "Compare results and analyze differences",
                        "expected_outcome": "Identification of conversational context impact"
                    }
                ],
                "control_conditions": "Standard operation without conversational context",
                "confounding_factors": ["User expertise", "Task familiarity", "System load"],
                "estimated_duration": "2 hours",
                "required_resources": ["Task execution environment", "Performance monitoring tools"],
                "source_hypothesis": hypothesis,
                "generated_from": "conversational_insight",
                "timestamp": __import__('datetime').datetime.utcnow().isoformat()
            }
        }
        
        return basic_design
    
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
                'context': results.get('context', {}),
                'source': results.get('source', 'standard')
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
            source = experiment_record.get('source', 'standard')
            
            # Determine importance based on experiment outcomes
            importance_score = 0.5  # Base importance
            
            # Increase importance for successful experiments
            if success:
                importance_score += 0.2
            
            # Increase importance for high-confidence results
            importance_score += confidence * 0.2
            
            # Increase importance for conversational insight experiments
            if source == "conversational_insight":
                importance_score += 0.1
            
            # Increase importance for surprising or counterintuitive results
            if 'surprising' in findings.lower() or 'unexpected' in findings.lower():
                importance_score += 0.1
            
            # Cap importance
            importance_score = min(0.9, importance_score)
            
            # Create reasoning
            reasoning_why = f"""I completed experiment '{experiment_id}' to test the hypothesis: '{hypothesis}'. 
Experimentation is crucial for validating assumptions and advancing understanding. 
{'The successful outcome' if success else 'Even though the results were inconclusive,'} this experiment 
provides valuable insights for future decision-making. {'This experiment was particularly valuable as it originated from user conversations, showing how human interaction can generate meaningful research questions.' if source == 'conversational_insight' else ''}"""
            
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
- Source: {'Conversational Insight' if source == 'conversational_insight' else 'Standard Research'}
- Outcome: {'Success' if success else 'Inconclusive/Failed'}
- Confidence: {confidence:.1f}
- Key Findings: {findings}
- Learning Value: This experiment {'validated' if success else 'challenged'} my assumptions and {'confirmed' if success else 'revealed gaps in'} my understanding."""
            
            # Extract tags from hypothesis and findings
            hypothesis_words = [word.lower() for word in hypothesis.split() if len(word) > 3]
            findings_words = [word.lower() for word in findings.split() if len(word) > 3]
            tags = ['experiment', 'research', 'hypothesis', 'findings'] + hypothesis_words[:3] + findings_words[:2]
            
            # Add source-specific tags
            if source == "conversational_insight":
                tags.append("conversational-research")
                tags.append("user-insights")
            
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
                    'source': source,
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
            conversational_experiments = [exp for exp in self.completed_experiments if exp.get('source') == 'conversational_insight']
            avg_confidence = sum(exp.get('confidence', 0) for exp in self.completed_experiments) / len(self.completed_experiments)
            
            return {
                'total_experiments': len(self.completed_experiments),
                'successful_experiments': len(successful_experiments),
                'conversational_insight_experiments': len(conversational_experiments),
                'success_rate': len(successful_experiments) / len(self.completed_experiments) if self.completed_experiments else 0,
                'average_confidence': avg_confidence,
                'recent_experiments': self.completed_experiments[-5:],  # Last 5
                'blog_triggers_created': len([exp for exp in self.completed_experiments if exp.get('confidence', 0) > 0.6])
            }
            
        except Exception as e:
            logger.error(f"Failed to get experiment summary: {e}")
            return {'error': str(e)}