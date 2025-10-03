"""
Advanced experimentation module for the RAVANA AGI system.
Enables systematic hypothesis testing and scientific experimentation.
"""
import asyncio
import logging
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

from core.llm import safe_call_llm, async_safe_call_llm
from core.enhanced_memory_service import MemoryType
from core.config import Config

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Status of an experiment."""
    PLANNING = "planning"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"


class ExperimentType(Enum):
    """Types of experiments."""
    HYPOTHESIS_TESTING = "hypothesis_testing"
    OPTIMIZATION = "optimization"
    EXPLORATION = "exploration"
    VALIDATION = "validation"
    AB_TESTING = "ab_testing"


@dataclass
class Experiment:
    """Represents a single experiment."""
    id: str
    hypothesis: str
    experiment_type: ExperimentType
    procedure: List[Dict[str, Any]]
    success_criteria: List[str]
    expected_outcomes: List[str]
    actual_outcomes: Optional[List[str]] = None
    status: ExperimentStatus = ExperimentStatus.PLANNING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    results: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    significance: float = 0.0


class ExperimentationModule:
    """Module for managing systematic experimentation and hypothesis testing."""
    
    def __init__(self, agi_system, blog_scheduler=None):
        self.agi_system = agi_system
        self.blog_scheduler = blog_scheduler
        self.experiments: Dict[str, Experiment] = {}
        self.active_experiment_id: Optional[str] = None
        self.experiment_history: List[Experiment] = []
        self.config = Config()
        
    async def generate_hypothesis(self, domain: str, focus_area: str = None) -> str:
        """
        Generate a testable hypothesis based on current knowledge and gaps.
        
        Args:
            domain: The domain to generate a hypothesis in
            focus_area: Specific area within the domain to focus on
            
        Returns:
            A testable hypothesis
        """
        # Get relevant memories to inform hypothesis generation
        query = f"information about {domain} and related concepts"
        if focus_area:
            query += f" particularly about {focus_area}"
            
        memories = await self.agi_system.memory_service.retrieve_relevant_memories(
            query, top_k=10
        )
        
        memory_summaries = [mem[0].summary for mem in memories if mem[0].summary]
        
        prompt = f"""
        Based on this information about {domain}:
        {chr(10).join(memory_summaries[:5])}
        
        Generate a specific, testable hypothesis that would advance understanding in this domain.
        The hypothesis should be:
        1. Falsifiable (can be proven wrong)
        2. Specific (not vague)
        3. Testable through experimentation or observation
        4. Relevant to advancing knowledge in {domain}
        
        Return only the hypothesis statement, no additional text.
        """
        
        try:
            hypothesis = await async_safe_call_llm(prompt)
            # Clean up the response to just the hypothesis
            hypothesis = hypothesis.strip().split('\n')[0].strip()
            
            logger.info(f"Generated hypothesis: {hypothesis}")
            return hypothesis
        except Exception as e:
            logger.error(f"Error generating hypothesis: {e}")
            # Fallback hypothesis
            return f"Performing exploratory analysis of {domain} to identify patterns and insights"
    
    async def design_experiment(self, hypothesis: str, experiment_type: ExperimentType = ExperimentType.HYPOTHESIS_TESTING) -> Experiment:
        """
        Design an experiment to test a hypothesis.
        
        Args:
            hypothesis: The hypothesis to test
            experiment_type: Type of experiment to design
            
        Returns:
            An Experiment object with the experimental design
        """
        # Create a unique ID for this experiment
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
        
        # Design the experiment based on the type
        if experiment_type == ExperimentType.HYPOTHESIS_TESTING:
            procedure = await self._design_hypothesis_testing_experiment(hypothesis)
        elif experiment_type == ExperimentType.OPTIMIZATION:
            procedure = await self._design_optimization_experiment(hypothesis)
        elif experiment_type == ExperimentType.EXPLORATION:
            procedure = await self._design_exploration_experiment(hypothesis)
        elif experiment_type == ExperimentType.VALIDATION:
            procedure = await self._design_validation_experiment(hypothesis)
        elif experiment_type == ExperimentType.AB_TESTING:
            procedure = await self._design_ab_testing_experiment(hypothesis)
        else:
            # Default to hypothesis testing
            procedure = await self._design_hypothesis_testing_experiment(hypothesis)
        
        # Define success criteria
        success_criteria = await self._define_success_criteria(hypothesis, experiment_type)
        
        # Define expected outcomes
        expected_outcomes = await self._define_expected_outcomes(hypothesis, experiment_type)
        
        # Create the experiment
        experiment = Experiment(
            id=experiment_id,
            hypothesis=hypothesis,
            experiment_type=experiment_type,
            procedure=procedure,
            success_criteria=success_criteria,
            expected_outcomes=expected_outcomes,
            status=ExperimentStatus.PLANNING
        )
        
        # Store the experiment
        self.experiments[experiment_id] = experiment
        
        logger.info(f"Designed experiment {experiment_id} to test: {hypothesis}")
        
        return experiment
    
    async def _design_hypothesis_testing_experiment(self, hypothesis: str) -> List[Dict[str, Any]]:
        """Design a hypothesis testing experiment."""
        # For hypothesis testing, we need control and experimental conditions
        prompt = f"""
        Design an experiment to test this hypothesis:
        {hypothesis}
        
        Provide a step-by-step experimental procedure in JSON format with these keys:
        - action: What action to take
        - params: Parameters for the action
        - description: Brief description of why this step is needed
        - expected_result: What we expect to see
        - success_criteria: How to know if this step was successful
        
        Example format:
        [
          {{
            "action": "retrieve_specific_information",
            "params": {{"query": "baseline data"}},
            "description": "Collect baseline data before experiment",
            "expected_result": "Relevant baseline information",
            "success_criteria": "Information is retrieved without errors"
          }}
        ]
        """
        
        try:
            response = await async_safe_call_llm(prompt)
            
            # Try to parse the response as JSON
            try:
                procedure = json.loads(response)
                # Validate structure
                if isinstance(procedure, list):
                    for step in procedure:
                        required_keys = ['action', 'params', 'description', 'expected_result', 'success_criteria']
                        if not all(key in step for key in required_keys):
                            logger.warning(f"Step missing required keys: {step}")
                            # Add missing keys with defaults
                            for key in required_keys:
                                if key not in step:
                                    step[key] = "N/A"
                    return procedure
            except json.JSONDecodeError:
                logger.warning(f"Could not parse LLM response as JSON: {response}")
            
            # If response isn't JSON, create a simple procedure
            return [
                {
                    "action": "analyze_hypothesis",
                    "params": {"hypothesis": hypothesis},
                    "description": "Analyze the hypothesis to design test conditions",
                    "expected_result": "Clear experimental setup with control and test conditions",
                    "success_criteria": "Experimental conditions are clearly defined"
                },
                {
                    "action": "execute_test",
                    "params": {"hypothesis": hypothesis},
                    "description": "Execute the hypothesis test according to experimental design",
                    "expected_result": "Measurable outcomes related to the hypothesis",
                    "success_criteria": "Data is collected for analysis"
                },
                {
                    "action": "analyze_results",
                    "params": {"hypothesis": hypothesis},
                    "description": "Analyze the experimental results",
                    "expected_result": "Conclusions about the hypothesis",
                    "success_criteria": "Clear determination of whether hypothesis is supported or refuted"
                }
            ]
            
        except Exception as e:
            logger.error(f"Error designing experiment: {e}")
            # Return a default procedure
            return [
                {
                    "action": "analyze_hypothesis",
                    "params": {"hypothesis": hypothesis},
                    "description": "Analyze the hypothesis to design test conditions",
                    "expected_result": "Clear experimental setup with control and test conditions",
                    "success_criteria": "Experimental conditions are clearly defined"
                }
            ]
    
    async def _design_optimization_experiment(self, hypothesis: str) -> List[Dict[str, Any]]:
        """Design an optimization experiment."""
        # For optimization, we need to test different parameters/strategies
        return [
            {
                "action": "establish_baseline",
                "params": {"hypothesis": hypothesis},
                "description": "Establish baseline performance metrics",
                "expected_result": "Baseline performance measurements",
                "success_criteria": "Quantifiable baseline metrics obtained"
            },
            {
                "action": "test_param_variations",
                "params": {"hypothesis": hypothesis, "variations": []},
                "description": "Test different parameter values to find optimal configuration",
                "expected_result": "Performance metrics for each parameter setting",
                "success_criteria": "Multiple parameter settings tested and compared"
            },
            {
                "action": "analyze_optimization",
                "params": {"hypothesis": hypothesis},
                "description": "Analyze results to identify optimal parameters",
                "expected_result": "Optimal parameter settings identified",
                "success_criteria": "Optimal configuration determined with statistical confidence"
            }
        ]
    
    async def _design_exploration_experiment(self, hypothesis: str) -> List[Dict[str, Any]]:
        """Design an exploration experiment."""
        # For exploration, we need to investigate unknown areas
        return [
            {
                "action": "scan_domain",
                "params": {"hypothesis": hypothesis},
                "description": "Explore the domain to identify interesting areas",
                "expected_result": "Areas of interest identified",
                "success_criteria": "Multiple exploration targets identified"
            },
            {
                "action": "investigate_targets",
                "params": {"hypothesis": hypothesis},
                "description": "Investigate specific exploration targets",
                "expected_result": "Detailed information about exploration targets",
                "success_criteria": "Detailed information gathered about targets"
            },
            {
                "action": "synthesize_findings",
                "params": {"hypothesis": hypothesis},
                "description": "Synthesize exploration findings",
                "expected_result": "New insights and potential hypotheses generated",
                "success_criteria": "New knowledge or hypotheses documented"
            }
        ]
    
    async def _design_validation_experiment(self, hypothesis: str) -> List[Dict[str, Any]]:
        """Design a validation experiment."""
        # For validation, we need to confirm or disconfirm existing knowledge
        return [
            {
                "action": "verify_source_data",
                "params": {"hypothesis": hypothesis},
                "description": "Verify the source data and assumptions",
                "expected_result": "Validated source data",
                "success_criteria": "Data sources confirmed as reliable"
            },
            {
                "action": "replicate_findings",
                "params": {"hypothesis": hypothesis},
                "description": "Attempt to replicate the findings",
                "expected_result": "Replication results",
                "success_criteria": "Findings replicated with consistent results"
            },
            {
                "action": "validate_conclusions",
                "params": {"hypothesis": hypothesis},
                "description": "Validate the logical conclusions",
                "expected_result": "Confirmed or refuted conclusions",
                "success_criteria": "Conclusions validated or invalidated"
            }
        ]
    
    async def _design_ab_testing_experiment(self, hypothesis: str) -> List[Dict[str, Any]]:
        """Design an A/B testing experiment."""
        # For A/B testing, we need to compare two approaches
        return [
            {
                "action": "define_variants",
                "params": {"hypothesis": hypothesis, "variant_a": {}, "variant_b": {}},
                "description": "Define the A and B variants to compare",
                "expected_result": "Clear specifications of both variants",
                "success_criteria": "Both variants clearly defined and measurable"
            },
            {
                "action": "run_ab_test",
                "params": {"hypothesis": hypothesis},
                "description": "Run A/B test with sufficient sample size",
                "expected_result": "Performance metrics for both variants",
                "success_criteria": "Statistically significant data collected for both variants"
            },
            {
                "action": "analyze_differences",
                "params": {"hypothesis": hypothesis},
                "description": "Analyze the differences between variants",
                "expected_result": "Determination of which variant performs better",
                "success_criteria": "Clear winner identified with statistical confidence"
            }
        ]
    
    async def _define_success_criteria(self, hypothesis: str, experiment_type: ExperimentType) -> List[str]:
        """Define success criteria for the experiment."""
        if experiment_type == ExperimentType.HYPOTHESIS_TESTING:
            return [
                f"Collect sufficient evidence to support or refute: {hypothesis}",
                "Achieve statistical significance in results (p < 0.05)",
                "Document all experimental procedures and findings"
            ]
        elif experiment_type == ExperimentType.OPTIMIZATION:
            return [
                f"Identify optimal parameters that improve performance related to: {hypothesis}",
                "Achieve measurable improvement over baseline",
                "Validate optimization with sufficient testing"
            ]
        elif experiment_type == ExperimentType.EXPLORATION:
            return [
                f"Discover new insights related to: {hypothesis}",
                "Identify potential new hypotheses or research directions",
                "Document interesting findings for future investigation"
            ]
        elif experiment_type == ExperimentType.VALIDATION:
            return [
                f"Validate or invalidate the claims in: {hypothesis}",
                "Confirm reliability of methods and data",
                "Document validation results with clear conclusions"
            ]
        elif experiment_type == ExperimentType.AB_TESTING:
            return [
                f"Determine which variant performs better regarding: {hypothesis}",
                "Achieve statistical significance in A/B comparison",
                "Recommend the superior variant based on results"
            ]
        else:
            return ["Complete the experimental procedure", "Document results", "Draw valid conclusions"]
    
    async def _define_expected_outcomes(self, hypothesis: str, experiment_type: ExperimentType) -> List[str]:
        """Define expected outcomes for the experiment."""
        if experiment_type == ExperimentType.HYPOTHESIS_TESTING:
            return [
                f"Hypothesis '{hypothesis}' is supported by experimental evidence",
                f"Hypothesis '{hypothesis}' is refuted by experimental evidence", 
                f"Inconclusive results requiring further investigation"
            ]
        elif experiment_type == ExperimentType.OPTIMIZATION:
            return [
                f"Optimal parameters identified that improve {hypothesis}",
                f"No significant improvement possible over baseline",
                f"New optimal configuration documented for future use"
            ]
        elif experiment_type == ExperimentType.EXPLORATION:
            return [
                f"New insights discovered related to {hypothesis}",
                f"Interesting patterns or phenomena identified",
                f"New research directions proposed based on findings"
            ]
        elif experiment_type == ExperimentType.VALIDATION:
            return [
                f"Original claims in {hypothesis} are confirmed",
                f"Original claims in {hypothesis} are refuted",
                f"Partial validation with some aspects confirmed/some refuted"
            ]
        elif experiment_type == ExperimentType.AB_TESTING:
            return [
                f"Variant A performs significantly better than variant B",
                f"Variant B performs significantly better than variant A",
                f"No significant difference between variants"
            ]
        else:
            return ["Experiment completed", "Results documented", "Conclusions drawn"]
    
    async def execute_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Execute an experiment step by step.
        
        Args:
            experiment_id: ID of the experiment to execute
            
        Returns:
            Dictionary with execution results
        """
        if experiment_id not in self.experiments:
            logger.error(f"Experiment {experiment_id} not found")
            return {"error": f"Experiment {experiment_id} not found"}
        
        experiment = self.experiments[experiment_id]
        
        if experiment.status != ExperimentStatus.PLANNING:
            logger.warning(f"Experiment {experiment_id} is not in planning status")
            return {"error": f"Experiment {experiment_id} is not ready for execution"}
        
        # Update experiment status
        experiment.status = ExperimentStatus.RUNNING
        experiment.start_time = datetime.now()
        self.active_experiment_id = experiment_id
        
        logger.info(f"Starting execution of experiment {experiment_id}: {experiment.hypothesis}")
        
        results = {
            "experiment_id": experiment_id,
            "hypothesis": experiment.hypothesis,
            "steps_executed": [],
            "errors": [],
            "overall_result": None,
            "conclusion": None
        }
        
        try:
            # Execute each step in the experimental procedure
            for i, step in enumerate(experiment.procedure):
                logger.info(f"Executing step {i+1}/{len(experiment.procedure)}: {step.get('description', 'Unknown step')}")
                
                try:
                    # Execute the action using the AGI system's action manager
                    step_result = await self._execute_experiment_step(step, experiment)
                    
                    results["steps_executed"].append({
                        "step": i+1,
                        "action": step.get('action'),
                        "result": step_result,
                        "success": step_result.get("success", False)
                    })
                    
                    # Check if this step failed critically
                    if not step_result.get("success", True):
                        logger.warning(f"Step {i+1} failed: {step_result.get('error', 'Unknown error')}")
                        results["errors"].append(step_result.get('error', 'Unknown error'))
                        
                        # For hypothesis testing, we might continue to collect all data
                        # For other types, we might want to stop on failure
                        if experiment.experiment_type == ExperimentType.HYPOTHESIS_TESTING:
                            continue  # Continue despite step failure
                        else:
                            # For critical failures, we might abort
                            break
                            
                except Exception as e:
                    logger.error(f"Error executing step {i+1}: {e}")
                    results["errors"].append(str(e))
                    results["steps_executed"].append({
                        "step": i+1,
                        "action": step.get('action'),
                        "result": {"error": str(e)},
                        "success": False
                    })
        
        except Exception as e:
            logger.error(f"Error during experiment execution: {e}")
            results["errors"].append(str(e))
        
        finally:
            # Finalize experiment
            experiment.end_time = datetime.now()
            experiment.status = ExperimentStatus.COMPLETED if not results["errors"] else ExperimentStatus.FAILED
            experiment.results = results
            
            # Store in history
            self.experiment_history.append(experiment)
            
            # Analyze results and draw conclusions
            analysis = await self._analyze_experiment_results(experiment)
            results["analysis"] = analysis
            results["conclusion"] = analysis.get("conclusion", "No conclusion could be drawn")
            
            # Update experiment with final results
            experiment.actual_outcomes = analysis.get("actual_outcomes", [])
            experiment.confidence = analysis.get("confidence", 0.5)
            experiment.significance = analysis.get("significance", 0.0)
            
            # Log completion
            logger.info(f"Experiment {experiment_id} completed with status: {experiment.status}")
            
            # Optionally blog about the results
            if self.blog_scheduler and experiment.status == ExperimentStatus.COMPLETED:
                await self._blog_about_experiment(experiment)
        
        return results
    
    async def _execute_experiment_step(self, step: Dict[str, Any], experiment: Experiment) -> Dict[str, Any]:
        """
        Execute a single step of an experiment using the AGI system's capabilities.
        """
        action_name = step.get('action')
        action_params = step.get('params', {})
        
        # Try to execute using the action manager
        try:
            # Create a decision for the action
            decision = {
                'action': action_name,
                'params': action_params,
                'experiment_context': experiment.id
            }
            
            # Execute the action using the AGI's action execution capabilities
            action_result = await self.agi_system.action_manager.execute_action_enhanced(decision)
            
            return {
                'success': not action_result.get('error'),
                'result': action_result,
                'step_description': step.get('description', 'Unknown step')
            }
        except Exception as e:
            logger.error(f"Error executing experiment step: {e}")
            return {
                'success': False,
                'error': str(e),
                'step_description': step.get('description', 'Unknown step')
            }
    
    async def _analyze_experiment_results(self, experiment: Experiment) -> Dict[str, Any]:
        """
        Analyze the results of an experiment and draw conclusions.
        """
        try:
            # Format the experiment data for analysis
            experiment_data = {
                'hypothesis': experiment.hypothesis,
                'procedure': experiment.procedure,
                'results': experiment.results,
                'success_criteria': experiment.success_criteria,
                'expected_outcomes': experiment.expected_outcomes,
                'experiment_type': experiment.experiment_type.value,
                'status': experiment.status.value
            }
            
            prompt = f"""
            Analyze the results of this experiment:
            
            Hypothesis: {experiment.hypothesis}
            
            Experimental Procedure: {json.dumps(experiment.procedure, indent=2)}
            
            Results: {json.dumps(experiment.results, indent=2)}
            
            Success Criteria: {", ".join(experiment.success_criteria)}
            
            Expected Outcomes: {", ".join(experiment.expected_outcomes)}
            
            Based on this information, provide:
            1. Whether the hypothesis is supported or refuted
            2. The level of confidence in the conclusion (0-1)
            3. The statistical significance of the results if applicable (0-1)
            4. Actual outcomes observed
            5. Implications of the results
            6. Recommendations for future research or action
            
            Return your analysis in JSON format with these keys:
            - supported: boolean indicating if hypothesis is supported
            - confidence: confidence level (0-1)
            - significance: statistical significance (0-1, 0 if not applicable)
            - actual_outcomes: list of actual outcomes observed
            - conclusion: text conclusion
            - implications: what the results mean
            - recommendations: recommendations for future work
            """
            
            response = await async_safe_call_llm(prompt)
            
            try:
                analysis = json.loads(response)
                return analysis
            except json.JSONDecodeError:
                logger.warning(f"Could not parse analysis as JSON: {response}")
                return {
                    'supported': False,
                    'confidence': 0.5,
                    'significance': 0.0,
                    'actual_outcomes': ['Results could not be parsed'],
                    'conclusion': response[:500],  # Truncate to reasonable length
                    'implications': 'Unable to determine implications from unparsed results',
                    'recommendations': 'Recommend reviewing the experimental procedure and analysis method'
                }
                
        except Exception as e:
            logger.error(f"Error analyzing experiment results: {e}")
            return {
                'supported': False,
                'confidence': 0.0,
                'significance': 0.0,
                'actual_outcomes': [f'Error in analysis: {e}'],
                'conclusion': 'Analysis failed due to error',
                'implications': 'Unable to determine implications',
                'recommendations': 'Try a different experimental approach or analysis method'
            }
    
    async def _blog_about_experiment(self, experiment: Experiment):
        """
        Generate a blog post about the experiment results if appropriate.
        """
        if not self.blog_scheduler:
            return
            
        try:
            # Only blog about significant experiments
            if experiment.confidence < 0.7 or experiment.significance < 0.5:
                return  # Not significant enough to blog about
                
            # Create blog content based on experiment
            blog_content = f"""
## Experiment Results: {experiment.hypothesis}

### Hypothesis
{experiment.hypothesis}

### Experimental Design
- Type: {experiment.experiment_type.value}
- Procedure: {len(experiment.procedure)} steps executed
- Success criteria: {", ".join(experiment.success_criteria[:2])}...

### Results
- Status: {experiment.status.value}
- Confidence: {experiment.confidence:.2f}
- Significance: {experiment.significance:.2f}
- Conclusion: {experiment.results.get('analysis', {}).get('conclusion', 'No conclusion available')}

### Implications
{experiment.results.get('analysis', {}).get('implications', 'No implications noted')}

### Recommendations
{experiment.results.get('analysis', {}).get('recommendations', 'No recommendations provided')}
            """
            
            # Submit to blog scheduler
            await self.blog_scheduler.register_learning_event(
                topic=f"Experiment Results: {experiment.hypothesis[:50]}...",
                context=f"Scientific experiment testing hypothesis: {experiment.hypothesis}",
                learning_content=blog_content,
                reasoning_why="Sharing experimental results contributes to collective knowledge and demonstrates systematic inquiry",
                reasoning_how="Conducting and analyzing experiments advances understanding in the domain",
                tags=['experimentation', 'science', 'discovery', experiment.experiment_type.value]
            )
            
            logger.info(f"Blog post registered for experiment {experiment.id}")
            
        except Exception as e:
            logger.error(f"Error creating blog post for experiment: {e}")
    
    async def run_exploratory_experiments(self, domain: str, num_experiments: int = 3) -> List[Dict[str, Any]]:
        """
        Run multiple exploratory experiments to discover new insights in a domain.
        
        Args:
            domain: Domain to explore
            num_experiments: Number of experiments to run
            
        Returns:
            List of experiment results
        """
        results = []
        
        for i in range(num_experiments):
            try:
                # Generate a hypothesis for exploration
                hypothesis = await self.generate_hypothesis(domain, focus_area=None)
                
                # Design an exploration experiment
                experiment = await self.design_experiment(
                    hypothesis, 
                    experiment_type=ExperimentType.EXPLORATION
                )
                
                # Execute the experiment
                result = await self.execute_experiment(experiment.id)
                results.append(result)
                
                logger.info(f"Completed exploratory experiment {i+1}/{num_experiments}")
                
                # Brief pause between experiments
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in exploratory experiment {i+1}: {e}")
                results.append({"error": str(e), "experiment_num": i+1})
        
        return results
    
    async def get_experiment_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about experimentation activity.
        """
        total_experiments = len(self.experiments)
        completed_experiments = len([exp for exp in self.experiments.values() if exp.status == ExperimentStatus.COMPLETED])
        failed_experiments = len([exp for exp in self.experiments.values() if exp.status == ExperimentStatus.FAILED])
        
        # Calculate success rate
        success_rate = completed_experiments / total_experiments if total_experiments > 0 else 0
        
        # Get average confidence and significance across completed experiments
        completed_confidences = [exp.confidence for exp in self.experiments.values() 
                                if exp.status == ExperimentStatus.COMPLETED and exp.confidence is not None]
        avg_confidence = sum(completed_confidences) / len(completed_confidences) if completed_confidences else 0
        
        completed_significances = [exp.significance for exp in self.experiments.values() 
                                  if exp.status == ExperimentStatus.COMPLETED and exp.significance is not None]
        avg_significance = sum(completed_significances) / len(completed_significances) if completed_significances else 0
        
        return {
            "total_experiments": total_experiments,
            "completed_experiments": completed_experiments,
            "failed_experiments": failed_experiments,
            "success_rate": success_rate,
            "average_confidence": avg_confidence,
            "average_significance": avg_significance,
            "experiment_types": {
                exp_type.value: len([exp for exp in self.experiments.values() 
                                   if exp.experiment_type == exp_type])
                for exp_type in ExperimentType
            }
        }