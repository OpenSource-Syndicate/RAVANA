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
        
    async def generate_hypothesis(self, domain: str, focus_area: str = None, impossible: bool = False) -> str:
        """
        Generate a testable hypothesis based on current knowledge and gaps.
        
        Args:
            domain: The domain to generate a hypothesis in
            focus_area: Specific area within the domain to focus on
            impossible: Whether to generate an 'impossible' or highly challenging hypothesis
            
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
        
        if impossible:
            prompt = f"""
            Based on this information about {domain}:
            {chr(10).join(memory_summaries[:5])}
            
            Generate a specific, seemingly impossible hypothesis that would advance understanding in this domain.
            The hypothesis should be:
            1. Currently considered impossible or extremely difficult by conventional wisdom
            2. Specific (not vague)
            3. Potentially testable through innovative experimentation or observation
            4. Relevant to advancing knowledge in {domain}
            5. Push the boundaries of what is considered possible
            
            Examples of impossible hypotheses:
            - "Consciousness can be transferred between systems without loss of identity"
            - "Information can be retrieved from a black hole using quantum entanglement"
            - "A perpetual motion machine can be created using quantum fluctuations"
            
            Return only the hypothesis statement, no additional text.
            """
        else:
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
    
    async def generate_impossible_hypothesis(self, domain: str, focus_area: str = None) -> str:
        """
        Generate an 'impossible' or highly challenging hypothesis.
        
        Args:
            domain: The domain to generate a hypothesis in
            focus_area: Specific area within the domain to focus on
            
        Returns:
            An 'impossible' hypothesis
        """
        return await self.generate_hypothesis(domain, focus_area, impossible=True)
    
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
        Enhanced to include failure analysis and alternative approaches when experiments fail.
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
            
            # Determine if this is a failure that needs special analysis
            is_failure = experiment.status in [ExperimentStatus.FAILED, ExperimentStatus.ABANDONED]
            
            if is_failure:
                prompt = f"""
                Analyze the results of this failed experiment:
                
                Hypothesis: {experiment.hypothesis}
                
                Experimental Procedure: {json.dumps(experiment.procedure, indent=2)}
                
                Results: {json.dumps(experiment.results, indent=2)}
                
                Success Criteria: {", ".join(experiment.success_criteria)}
                
                Expected Outcomes: {", ".join(experiment.expected_outcomes)}
                
                Status: {experiment.status.value}
                
                Perform a comprehensive failure analysis that includes:
                1. Root causes of the failure
                2. What was learned despite the failure
                3. Alternative approaches that might succeed
                4. Modifications to the original hypothesis that could make it testable
                5. Broader implications of this failure for the domain
                6. How this failure advances understanding
                
                The mad scientist approach values learning from failures as much as successes.
                
                Return your analysis in JSON format with these keys:
                - supported: boolean indicating if hypothesis is supported (likely false for failures)
                - confidence: confidence level (0-1) 
                - significance: statistical significance (0-1, 0 if not applicable)
                - actual_outcomes: list of actual outcomes observed
                - conclusion: text conclusion
                - implications: what the results mean
                - recommendations: recommendations for future work
                - failure_analysis: detailed analysis of what went wrong and why
                - alternative_approaches: potential alternative approaches to test the hypothesis
                - learning_points: key learning points from the failure
                """
            else:
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
                
                # If this was a failure, also trigger alternative approach discovery
                if is_failure and 'alternative_approaches' in analysis:
                    # Store alternative approaches for future use
                    for approach in analysis['alternative_approaches']:
                        logger.info(f"Discovered alternative approach: {approach}")
                        # Potentially store in knowledge base for future use
                        await self.agi_system.knowledge_service.add_knowledge(
                            content=approach,
                            source="failure_analysis",
                            category="alternative_approach"
                        )
                
                return analysis
            except json.JSONDecodeError:
                logger.warning(f"Could not parse analysis as JSON: {response}")
                
                if is_failure:
                    return {
                        'supported': False,
                        'confidence': 0.2,  # Lower confidence for failed experiments
                        'significance': 0.0,
                        'actual_outcomes': ['Experiment failed - specific outcomes unknown'],
                        'conclusion': 'Experiment failed, but valuable insights may be gained from analysis',
                        'implications': 'Failure provides information about limitations and constraints',
                        'recommendations': 'Consider alternative approaches to test the same hypothesis',
                        'failure_analysis': f'Attempt to test hypothesis failed. Details: {response[:300]}',
                        'alternative_approaches': ['Try different experimental design', 'Modify hypothesis to be more testable'],
                        'learning_points': ['Learned what does not work', 'Identified specific failure points']
                    }
                else:
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
            
            # Determine if this was a failure analysis based on status
            is_failure = experiment.status in [ExperimentStatus.FAILED, ExperimentStatus.ABANDONED]
            
            if is_failure:
                return {
                    'supported': False,
                    'confidence': 0.0,
                    'significance': 0.0,
                    'actual_outcomes': [f'Error in analysis: {e}'],
                    'conclusion': 'Analysis failed due to error, but failure analysis is still valuable',
                    'implications': 'Unable to determine implications',
                    'recommendations': 'Try a different experimental approach or analysis method',
                    'failure_analysis': f'Error during failure analysis: {e}',
                    'alternative_approaches': ['Consider completely different approach to hypothesis'],
                    'learning_points': ['Even failed analysis provides meta-learning opportunities']
                }
            else:
                return {
                    'supported': False,
                    'confidence': 0.0,
                    'significance': 0.0,
                    'actual_outcomes': [f'Error in analysis: {e}'],
                    'conclusion': 'Analysis failed due to error',
                    'implications': 'Unable to determine implications',
                    'recommendations': 'Try a different experimental approach or analysis method'
                }

    async def analyze_failure_patterns(self, domain: str = None) -> Dict[str, Any]:
        """
        Analyze patterns in failed experiments to extract general insights.
        
        Args:
            domain: Optional domain to focus the analysis on
            
        Returns:
            A dictionary with failure pattern analysis
        """
        logger.info(f"Analyzing failure patterns{' in domain: ' + domain if domain else ''}")
        
        # Get all experiments
        all_experiments = list(self.experiments.values())
        
        # Filter by domain if specified
        if domain:
            domain_experiments = []
            for exp in all_experiments:
                if domain.lower() in exp.hypothesis.lower():
                    domain_experiments.append(exp)
            experiments_to_analyze = domain_experiments
        else:
            experiments_to_analyze = all_experiments
        
        # Separate failed and successful experiments
        failed_experiments = [exp for exp in experiments_to_analyze if exp.status in [ExperimentStatus.FAILED, ExperimentStatus.ABANDONED]]
        successful_experiments = [exp for exp in experiments_to_analyze if exp.status in [ExperimentStatus.COMPLETED]]
        
        if not failed_experiments:
            logger.info("No failed experiments to analyze")
            return {
                'total_experiments': len(experiments_to_analyze),
                'failed_experiments': 0,
                'success_rate': 1.0 if experiments_to_analyze else 0.0,
                'failure_patterns': [],
                'learning_opportunities': [],
                'recommendations': ['Continue current approach - success rate is high']
            }
        
        # Format failed experiments for analysis
        failed_experiment_summaries = []
        for exp in failed_experiments:
            failed_experiment_summaries.append({
                'hypothesis': exp.hypothesis,
                'type': exp.experiment_type.value,
                'procedure_length': len(exp.procedure),
                'results': exp.results
            })
        
        prompt = f"""
        Analyze these failed experiments to identify patterns and extract valuable insights:
        
        Total Experiments Analyzed: {len(experiments_to_analyze)}
        Failed Experiments: {len(failed_experiments)}
        Successful Experiments: {len(successful_experiments)}
        
        Failed Experiment Summaries:
        {json.dumps(failed_experiment_summaries, indent=2)}
        
        Identify:
        1. Common patterns in the failures
        2. What these failures reveal about the domain
        3. Learning opportunities from these failures
        4. Recommendations to avoid similar failures in the future
        5. Potential areas where the approach might be too conservative or too aggressive
        6. Unexpected insights gained from the failures
        
        Remember: The mad scientist approach values learning from failures as much as celebrating successes.
        
        Return your analysis in JSON format with these keys:
        - total_experiments: total number of experiments analyzed
        - failed_experiments: number of failed experiments
        - success_rate: calculated success rate
        - failure_patterns: common patterns identified in failures
        - learning_opportunities: valuable learning opportunities from failures
        - recommendations: suggestions for improving experimental approach
        - unexpected_insights: any unexpected insights from the failures
        """
        
        try:
            response = await async_safe_call_llm(prompt)
            
            try:
                analysis = json.loads(response)
                return analysis
            except json.JSONDecodeError:
                logger.warning(f"Could not parse failure analysis as JSON: {response}")
                
                return {
                    'total_experiments': len(experiments_to_analyze),
                    'failed_experiments': len(failed_experiments),
                    'success_rate': len(successful_experiments) / len(experiments_to_analyze) if experiments_to_analyze else 0,
                    'failure_patterns': ['Failed to parse detailed analysis'],
                    'learning_opportunities': ['Improve analysis parsing'],
                    'recommendations': ['Review error in failure analysis'],
                    'unexpected_insights': []
                }
                
        except Exception as e:
            logger.error(f"Error analyzing failure patterns: {e}")
            return {
                'total_experiments': len(experiments_to_analyze),
                'failed_experiments': len(failed_experiments),
                'success_rate': len(successful_experiments) / len(experiments_to_analyze) if experiments_to_analyze else 0,
                'failure_patterns': [f'Error in analysis: {e}'],
                'learning_opportunities': ['Error analysis'],
                'recommendations': ['Fix failure analysis system'],
                'unexpected_insights': []
            }

    async def discover_alternative_pathways(self, original_goal: str, failure_context: str = None) -> List[Dict[str, str]]:
        """
        Discover alternative pathways to achieve a goal when primary methods fail.
        
        Args:
            original_goal: The original goal that needs alternative approaches
            failure_context: Context about why primary methods failed (optional)
            
        Returns:
            A list of alternative pathways, each with a description and rationale
        """
        logger.info(f"Discovering alternative pathways for goal: {original_goal}")
        
        # Get relevant memories to inform alternative pathway discovery
        query = f"alternative approaches to achieve {original_goal} or related goals"
        memories = await self.agi_system.memory_service.retrieve_relevant_memories(
            query, top_k=10
        )
        
        memory_summaries = [mem[0].summary for mem in memories if mem[0].summary]
        
        if failure_context:
            prompt = f"""
            The original goal was: {original_goal}
            The context of failure was: {failure_context}
            
            Additional context from knowledge base:
            {chr(10).join(memory_summaries[:5])}
            
            Discover alternative pathways to achieve the same or similar outcomes as the original goal.
            Consider:
            1. Completely different approaches to the same problem
            2. Relaxing some constraints to make the goal achievable
            3. Breaking the goal into smaller, achievable sub-goals
            4. Indirect methods that achieve the same outcome through different means
            5. Related goals that provide similar value
            6. Evolution of the goal based on new information
            
            The mad scientist approach finds ways around impossibilities.
            
            Return your alternatives in JSON format as an array of objects with these keys:
            - approach: description of the alternative approach
            - rationale: why this approach might work when the original didn't
            - requirements: what would be needed to pursue this approach
            - risk_level: risk level from 0.0 to 1.0
            - estimated_difficulty: how difficult this approach is to implement
            """
        else:
            prompt = f"""
            The original goal was: {original_goal}
            
            Additional context from knowledge base:
            {chr(10).join(memory_summaries[:5])}
            
            Discover alternative pathways to achieve the same or similar outcomes as the original goal.
            Consider:
            1. Completely different approaches to the same problem
            2. Relaxing some constraints to make the goal achievable
            3. Breaking the goal into smaller, achievable sub-goals
            4. Indirect methods that achieve the same outcome through different means
            5. Related goals that provide similar value
            6. Evolution of the goal based on new information
            
            The mad scientist approach finds ways around impossibilities.
            
            Return your alternatives in JSON format as an array of objects with these keys:
            - approach: description of the alternative approach
            - rationale: why this approach might work when the original didn't
            - requirements: what would be needed to pursue this approach
            - risk_level: risk level from 0.0 to 1.0
            - estimated_difficulty: how difficult this approach is to implement
            """
        
        try:
            response = await async_safe_call_llm(prompt)
            
            try:
                alternatives = json.loads(response)
                
                if isinstance(alternatives, list):
                    logger.info(f"Discovered {len(alternatives)} alternative pathways for goal: {original_goal}")
                    
                    # Store alternatives in knowledge base
                    for alt_idx, alt in enumerate(alternatives):
                        knowledge_content = f"Alternative approach {alt_idx+1} to achieve {original_goal}: {alt.get('approach', '')}. Rationale: {alt.get('rationale', '')}"
                        await self.agi_system.knowledge_service.add_knowledge(
                            content=knowledge_content,
                            source="alternative_pathway_discovery",
                            category="alternative_approach"
                        )
                    
                    return alternatives
                else:
                    logger.warning("LLM response was not a list of alternatives")
                    # Return a default structure if parsing fails
                    return [{
                        'approach': 'Systematic exploration of alternative approaches',
                        'rationale': 'When primary approach fails, explore systematically',
                        'requirements': 'Analysis of primary failure mode',
                        'risk_level': 0.5,
                        'estimated_difficulty': 'medium'
                    }]
                    
            except json.JSONDecodeError:
                logger.warning(f"Could not parse alternatives as JSON: {response}")
                
                # Try to extract alternatives from plain text
                lines = response.strip().split('\n')
                alternatives = []
                
                current_approach = ""
                current_rationale = ""
                
                for line in lines:
                    if line.startswith("Approach:") or line.startswith("Alternative:") or line.startswith("1.") or line.startswith("2.") or line.startswith("3."):
                        if current_approach:
                            alternatives.append({
                                'approach': current_approach,
                                'rationale': current_rationale,
                                'requirements': 'Standard research tools',
                                'risk_level': 0.5,
                                'estimated_difficulty': 'medium'
                            })
                        current_approach = line.replace("Approach:", "").replace("Alternative:", "").replace("1.", "").replace("2.", "").replace("3.", "").strip()
                        current_rationale = ""
                    elif line.startswith("Rationale:") or "why " in line.lower() or "because" in line.lower():
                        current_rationale = line.replace("Rationale:", "").strip()
                
                # Add the last one if it exists
                if current_approach:
                    alternatives.append({
                        'approach': current_approach,
                        'rationale': current_rationale,
                        'requirements': 'Standard research tools',
                        'risk_level': 0.5,
                        'estimated_difficulty': 'medium'
                    })
                
                if alternatives:
                    logger.info(f"Discovered {len(alternatives)} alternative pathways from plain text for goal: {original_goal}")
                    return alternatives
                else:
                    logger.warning("Could not extract alternatives from LLM response")
                    return [{
                        'approach': 'Systematic analysis of the original goal to find alternative pathways',
                        'rationale': 'This is a fallback approach when alternatives are not clearly specified',
                        'requirements': 'Detailed analysis of the original goal and constraints',
                        'risk_level': 0.3,
                        'estimated_difficulty': 'high'
                    }]
                
        except Exception as e:
            logger.error(f"Error discovering alternative pathways: {e}")
            return [{
                'approach': 'Manual analysis of the goal to find alternative pathways',
                'rationale': 'LLM failed to generate alternatives, so manual approach needed',
                'requirements': 'Human researcher input',
                'risk_level': 0.1,
                'estimated_difficulty': 'high'
            }]

    async def apply_alternative_pathway(self, original_experiment: Experiment, alternative_approach: Dict[str, str]) -> Experiment:
        """
        Apply an alternative pathway to create a new experiment based on a failed one.
        
        Args:
            original_experiment: The original failed experiment
            alternative_approach: An alternative approach from discover_alternative_pathways
            
        Returns:
            A new experiment based on the alternative approach
        """
        logger.info(f"Applying alternative pathway for original experiment: {original_experiment.hypothesis}")
        
        # Create a new experiment based on the alternative approach
        experiment_id = f"alt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
        
        # Adapt the experimental procedure based on the alternative approach
        new_procedure = await self._adapt_procedure_for_alternative(
            original_experiment.procedure,
            alternative_approach
        )
        
        # Define success criteria for the alternative
        success_criteria = await self._define_success_criteria_for_alternative(
            original_experiment.hypothesis,
            alternative_approach
        )
        
        # Define expected outcomes for the alternative
        expected_outcomes = await self._define_expected_outcomes_for_alternative(
            original_experiment.hypothesis,
            alternative_approach
        )
        
        # Determine experiment type based on the alternative approach
        # If it's a completely different approach, it might be exploration
        # If it's refining the original idea, it might be hypothesis testing
        if "different approach" in alternative_approach.get('approach', '').lower():
            experiment_type = ExperimentType.EXPLORATION
        else:
            experiment_type = ExperimentType.HYPOTHESIS_TESTING
        
        # Create the new experiment
        new_experiment = Experiment(
            id=experiment_id,
            hypothesis=alternative_approach.get('approach', f"Alternative approach to: {original_experiment.hypothesis}"),
            experiment_type=experiment_type,
            procedure=new_procedure,
            success_criteria=success_criteria,
            expected_outcomes=expected_outcomes,
            status=ExperimentStatus.PLANNING
        )
        
        # Store the experiment
        self.experiments[experiment_id] = new_experiment
        
        logger.info(f"Created alternative experiment {experiment_id} for hypothesis: {new_experiment.hypothesis}")
        
        # Store a reference from the original experiment to the alternative
        if not hasattr(original_experiment, 'alternatives_tried'):
            original_experiment.alternatives_tried = []
        original_experiment.alternatives_tried.append(experiment_id)
        
        return new_experiment

    async def _adapt_procedure_for_alternative(self, original_procedure: List[Dict], alternative_approach: Dict[str, str]) -> List[Dict]:
        """
        Adapt an original experimental procedure for an alternative approach.
        """
        logger.debug(f"Adapting procedure for alternative approach: {alternative_approach.get('approach', 'Unknown')}")
        
        # Create a prompt to adapt the procedure
        prompt = f"""
        Original Experimental Procedure:
        {json.dumps(original_procedure, indent=2)}
        
        Alternative Approach:
        {alternative_approach.get('approach', '')}
        
        Rationale for Alternative:
        {alternative_approach.get('rationale', '')}
        
        Requirements for Alternative:
        {alternative_approach.get('requirements', '')}
        
        Please adapt the original experimental procedure to fit the alternative approach.
        The new procedure should:
        1. Align with the alternative approach
        2. Maintain scientific rigor
        3. Address the requirements of the alternative approach
        4. Preserve the goal of testing the underlying hypothesis in a new way
        5. Be feasible given the new approach
        
        Return the adapted procedure in the same format as the original.
        """
        
        try:
            response = await async_safe_call_llm(prompt)
            
            try:
                new_procedure = json.loads(response)
                if isinstance(new_procedure, list):
                    return new_procedure
                else:
                    logger.warning("Adapted procedure is not a list, returning original")
                    return original_procedure
            except json.JSONDecodeError:
                logger.warning(f"Could not parse adapted procedure as JSON: {response}")
                # Return a modified version of the original procedure
                return original_procedure
                
        except Exception as e:
            logger.error(f"Error adapting procedure for alternative: {e}")
            return original_procedure

    async def _define_success_criteria_for_alternative(self, original_hypothesis: str, alternative_approach: Dict[str, str]) -> List[str]:
        """
        Define success criteria for an alternative approach.
        """
        logger.debug(f"Defining success criteria for alternative: {alternative_approach.get('approach', 'Unknown')}")
        
        prompt = f"""
        Original Hypothesis: {original_hypothesis}
        
        Alternative Approach: {alternative_approach.get('approach', '')}
        
        Rationale: {alternative_approach.get('rationale', '')}
        
        Requirements: {alternative_approach.get('requirements', '')}
        
        Define success criteria for this alternative approach. The criteria should:
        1. Be achievable with this alternative approach
        2. Still address the core hypothesis or a derivative of it
        3. Be measurable and specific
        4. Reflect the new methodology
        5. Account for the different risk level and difficulty
        
        Return a list of 2-4 success criteria.
        """
        
        try:
            response = await async_safe_call_llm(prompt)
            
            # Parse the response into a list
            lines = response.strip().split('\n')
            criteria = []
            
            for line in lines:
                line = line.strip()
                # Remove numbering if present
                if line.startswith(('1.', '2.', '3.', '4.', '5.')):
                    line = line[2:].strip()
                elif line.startswith(('-', '*')):
                    line = line[1:].strip()
                
                if line:
                    criteria.append(line)
            
            if not criteria:
                criteria = [f"Successfully execute the alternative approach: {alternative_approach.get('approach', 'Unknown approach')}"]
            
            return criteria
            
        except Exception as e:
            logger.error(f"Error defining success criteria for alternative: {e}")
            return [f"Execute the alternative approach: {alternative_approach.get('approach', 'Unknown approach')}"]

    async def _define_expected_outcomes_for_alternative(self, original_hypothesis: str, alternative_approach: Dict[str, str]) -> List[str]:
        """
        Define expected outcomes for an alternative approach.
        """
        logger.debug(f"Defining expected outcomes for alternative: {alternative_approach.get('approach', 'Unknown')}")
        
        prompt = f"""
        Original Hypothesis: {original_hypothesis}
        
        Alternative Approach: {alternative_approach.get('approach', '')}
        
        Rationale: {alternative_approach.get('rationale', '')}
        
        Requirements: {alternative_approach.get('requirements', '')}
        
        Risk Level: {alternative_approach.get('risk_level', 'Unknown')}
        
        Define expected outcomes for this alternative approach. The outcomes should:
        1. Be realistic given the alternative methodology
        2. Relate to the original hypothesis or its implications
        3. Account for the different risk level
        4. Include both positive and negative possibilities
        
        Return a list of 2-4 expected outcomes.
        """
        
        try:
            response = await async_safe_call_llm(prompt)
            
            # Parse the response into a list
            lines = response.strip().split('\n')
            outcomes = []
            
            for line in lines:
                line = line.strip()
                # Remove numbering if present
                if line.startswith(('1.', '2.', '3.', '4.', '5.')):
                    line = line[2:].strip()
                elif line.startswith(('-', '*')):
                    line = line[1:].strip()
                
                if line:
                    outcomes.append(line)
            
            if not outcomes:
                outcomes = [f"Results from alternative approach to {original_hypothesis}"]
            
            return outcomes
            
        except Exception as e:
            logger.error(f"Error defining expected outcomes for alternative: {e}")
            return [f"Results from alternative approach to {original_hypothesis}"]
    
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