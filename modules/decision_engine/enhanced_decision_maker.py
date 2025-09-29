"""
Enhanced Decision Maker with Adaptive Reasoning Capabilities
"""

import logging
import json
import re
import random
from typing import Dict, Any, List, Optional
from modules.decision_engine.planner import GoalPlanner, plan_from_context
from core.llm import call_llm
from ..agent_self_reflection.self_modification import generate_hypothesis, analyze_experiment_outcome

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class AdaptiveReasoningEngine:
    """Enhanced decision engine with meta-cognitive capabilities for self-monitoring and self-regulation"""

    def __init__(self, agi_system=None):
        self.agi_system = agi_system
        self.reasoning_history = []
        self.confidence_model = self._initialize_confidence_model()
        self.decision_quality_history = []

    def _initialize_confidence_model(self):
        """Initialize a simple confidence scoring model"""
        # In a more advanced implementation, this would be a trained model
        return {
            "task_complexity_weights": {
                "simple": 0.9,
                "moderate": 0.7,
                "complex": 0.5
            },
            "outcome_history_weights": {
                "success": 1.0,
                "partial_success": 0.7,
                "failure": 0.3
            }
        }

    async def evaluate_decision_quality(self, decision: Dict[str, Any], outcome: Dict[str, Any]) -> float:
        """Evaluate the quality of a decision based on outcome"""
        try:
            # Simple heuristic-based quality scoring
            if outcome.get("error"):
                return 0.0

            # Check if the decision achieved its intended action
            if decision.get("action") and outcome.get("action_executed") == decision.get("action"):
                # Evaluate based on outcome success
                if outcome.get("success", False):
                    quality_score = 0.9
                else:
                    quality_score = 0.5
            else:
                quality_score = 0.3

            # Store in history for learning
            self.decision_quality_history.append({
                "decision": decision,
                "outcome": outcome,
                "quality_score": quality_score
            })

            # Keep only recent history
            if len(self.decision_quality_history) > 100:
                self.decision_quality_history = self.decision_quality_history[-100:]

            return quality_score
        except Exception as e:
            logging.error(f"Error evaluating decision quality: {e}")
            return 0.5  # Neutral score on error

    async def adaptive_prompting(self, task_description: str, context: Dict[str, Any]) -> str:
        """Generate adaptive prompts based on task complexity and context"""
        try:
            # Analyze task complexity
            complexity = await self._assess_task_complexity(task_description, context)

            # Select appropriate prompting strategy
            if complexity > 0.8:
                return await self._generate_chain_of_thought_prompt(task_description, context)
            elif complexity > 0.5:
                return await self._generate_multi_shot_prompt(task_description, context)
            else:
                return await self._generate_direct_prompt(task_description, context)
        except Exception as e:
            logging.error(f"Error in adaptive prompting: {e}")
            # Fallback to direct prompt
            return await self._generate_direct_prompt(task_description, context)

    async def _assess_task_complexity(self, task_description: str, context: Dict[str, Any]) -> float:
        """Assess the complexity of a task based on description and context"""
        # Simple heuristic-based complexity assessment
        word_count = len(task_description.split())
        special_chars = len(re.findall(r'[^\w\s]', task_description))

        # Context complexity factors
        context_size = len(str(context))
        has_search_results = 'search_results' in context and len(
            context['search_results']) > 0
        has_memories = 'memories' in context and len(context['memories']) > 0

        # Calculate complexity score (0.0 to 1.0)
        complexity = min(1.0, (
            (word_count / 100) * 0.3 +
            (special_chars / 10) * 0.2 +
            (context_size / 10000) * 0.3 +
            (0.2 if has_search_results else 0) +
            (0.2 if has_memories else 0)
        ))

        return complexity

    async def _generate_chain_of_thought_prompt(self, task_description: str, context: Dict[str, Any]) -> str:
        """Generate a chain-of-thought prompt for complex tasks"""
        prompt = f"""
Task: {task_description}

Context:
{json.dumps(context, indent=2, default=str)}

Please think through this task step by step:

1. First, analyze what is being asked and break it down into sub-components
2. Consider the context and how it relates to the task
3. Identify any potential challenges or ambiguities
4. Propose a approach to solve the task
5. Execute the approach and provide the result

Think carefully and explain your reasoning at each step before providing the final answer.
"""
        return prompt

    async def _generate_multi_shot_prompt(self, task_description: str, context: Dict[str, Any]) -> str:
        """Generate a multi-shot prompt for moderate complexity tasks"""
        prompt = f"""
Task: {task_description}

Context:
{json.dumps(context, indent=2, default=str)}

Please approach this task systematically:
1. Understand the requirements
2. Consider relevant information from the context
3. Plan your approach
4. Execute and provide a clear result

Focus on being thorough but concise in your response.
"""
        return prompt

    async def _generate_direct_prompt(self, task_description: str, context: Dict[str, Any]) -> str:
        """Generate a direct prompt for simple tasks"""
        prompt = f"""
Task: {task_description}

Context:
{json.dumps(context, indent=2, default=str)}

Please provide a direct and concise response to this task.
"""
        return prompt


async def enhanced_goal_driven_decision_maker_loop(
    situation: Dict[str, Any],
    memory: Optional[Any] = None,
    model: Optional[Any] = None,
    rag_context: Optional[Any] = None,
    hypotheses: Optional[List[str]] = None,
    shared_state: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Enhanced goal-driven decision-making loop with adaptive reasoning capabilities.
    """
    # Initialize adaptive reasoning engine
    reasoning_engine = AdaptiveReasoningEngine()

    planner = GoalPlanner()
    goals = planner.get_goals()

    if hypotheses is None:
        hypotheses = []
    if shared_state is None:
        shared_state = {}

    # Check if we are analyzing an experiment outcome
    if situation.get('type') == 'experiment_analysis':
        logging.info("Analyzing the outcome of a completed experiment.")
        return analyze_experiment_outcome(
            situation['context']['hypothesis'],
            situation['context']['situation_prompt'],
            situation['context']['outcome']
        )

    # If there's no active experiment, consider starting one
    # 10% chance to consider starting an experiment
    if not shared_state.get('active_experiment') and random.random() < 0.1:
        new_hypothesis = generate_hypothesis(shared_state)
        if new_hypothesis:
            logging.info(
                f"Generated a new hypothesis, will now generate a situation to test it.")
            # This decision signals the main loop to generate a test situation
            return {
                "action": "initiate_experiment",
                "hypothesis": new_hypothesis,
                "reason": "A new testable hypothesis was generated from recent performance."
            }

    if not goals:
        logging.info(
            "No goals found. Creating a new high-level meta-goal for self-improvement.")
        try:
            new_goal_id = plan_from_context(
                "Continuously improve my own architecture and capabilities to achieve ever-higher levels of intelligence and autonomy. This includes analyzing my modules, proposing improvements, and finding more efficient ways to use my tools.",
                timeframe="lifelong"
            )
            logging.info(
                f"Created a new meta-goal for self-improvement with ID: {new_goal_id}")
            goals = planner.get_goals()
        except Exception as e:
            logging.error(f"Failed to create initial meta-goal: {e}")
            return {"action": "wait", "reason": "Failed to create initial meta-goal."}

    # Prepare context for adaptive prompting
    context = {
        "situation": situation,
        "goals": goals,
        "hypotheses": hypotheses,
        "shared_state": shared_state
    }

    if rag_context:
        context["rag_context"] = rag_context
    if memory:
        context["memories"] = memory

    # Generate adaptive prompt
    task_description = f"Review goals, hypotheses, and situation to make a decision"
    adaptive_prompt = await reasoning_engine.adaptive_prompting(task_description, context)

    # Enhanced prompt with better structure guidance
    prompt = f"""
{adaptive_prompt}

Your response should be a JSON object with one of the following structures.
You have access to a list of actions/tools you can use. If you decide to execute a task, the 'task_description' should be a clear instruction for what to do, potentially using one of your available tools.

For executing a task:
{{
  "action": "task",
  "goal_id": int,
  "subgoal_id": int,
  "task_id": int,
  "task_description": str
}}

For testing a hypothesis:
{{
  "action": "test_hypothesis",
  "hypothesis_to_test": str,
  "test_method_description": str,
  "expected_outcome": str
}}

For proposing a new invention (uses the 'propose_and_test_invention' action):
{{
  "action": "propose_and_test_invention",
  "invention_description": "A detailed description of the new concept or invention.",
  "test_plan_suggestion": "A suggestion for how to test the viability of this invention."
}}

For proposing a new, more ambitious goal:
{{
  "action": "new_goal",
  "new_goal_context": str
}}

For proposing a self-improvement plan:
{{
  "action": "self_improvement",
  "plan_description": str,
  "modules_to_improve": list[str]
}}

For initiating an experiment:
{{
  "action": "initiate_experiment",
  "hypothesis": str,
  "reason": str
}}

For waiting when no immediate action is needed:
{{
  "action": "wait",
  "reason": str
}}
"""

    response_json = call_llm(prompt, model=model)

    try:
        decision = json.loads(response_json)
    except json.JSONDecodeError:
        logging.error(
            f"Failed to decode LLM response into JSON: {response_json}")
        # Attempt to extract the first valid JSON object from the response
        try:
            match = re.search(r'\{[\s\S]*\}', response_json)
            if match:
                decision = json.loads(match.group(0))
            else:
                return {"action": "wait", "reason": "LLM returned no valid action."}
        except json.JSONDecodeError:
            logging.error(
                "Failed to extract and decode JSON from LLM response.")
            return {"action": "wait", "reason": "Failed to extract and decode JSON from LLM response."}

    if decision.get('action') == 'task':
        # Execute the chosen task
        task_description = decision['task_description']
        logging.info(f"Executing task: {task_description}")
        # Here, you would integrate with the part of your AGI that executes tasks.
        # For now, we'll just log it and mark the task as complete.
        try:
            goal_id = decision.get('goal_id')
            subgoal_id = decision.get('subgoal_id')
            task_id = decision.get('task_id')

            if goal_id is not None and subgoal_id is not None and task_id is not None:
                planner.complete_task(goal_id, subgoal_id, task_id)
                logging.info(f"Task {task_id} completed.")
            else:
                logging.warning(
                    "Missing task identifiers, cannot mark task as complete.")

            return decision
        except ValueError as e:
            logging.error(f"Error completing task: {e}")
            return {"action": "wait", "reason": str(e)}

    elif decision.get('action') == 'test_hypothesis':
        # Log the hypothesis test plan
        hypothesis = decision['hypothesis_to_test']
        test_method = decision['test_method_description']
        logging.info(f"Proposing to test hypothesis: '{hypothesis}'")
        logging.info(f"Test method: {test_method}")
        # In a more advanced version, this could trigger a specific experiment or a new goal.
        return decision

    elif decision.get('action') == 'new_goal':
        # Create a new goal and plan
        new_goal_context = decision['new_goal_context']
        logging.info(f"Creating a new goal and plan for: {new_goal_context}")
        try:
            new_goal_id = plan_from_context(new_goal_context)
            logging.info(f"New goal created with ID: {new_goal_id}")
            decision['new_goal_id'] = new_goal_id
            return decision
        except Exception as e:
            logging.error(f"Failed to create new goal from context: {e}")
            return {"action": "wait", "reason": f"Failed to create new goal from context: {e}"}

    elif decision.get('action') == 'self_improvement':
        # Log the self-improvement plan
        plan_description = decision['plan_description']
        modules_to_improve = decision.get('modules_to_improve', [])
        logging.info(f"Proposed self-improvement plan: {plan_description}")
        logging.info(f"Modules to improve: {modules_to_improve}")
        # In a more advanced version, this could trigger a new goal or a series of tasks.
        return decision

    elif decision.get('action') == 'propose_and_test_invention':
        # This decision will be caught by the ActionManager, which will execute
        # the ProposeAndTestInventionAction. We just return the decision.
        logging.info(
            f"Proposing a new invention for testing: {decision.get('invention_description')}")
        return decision

    elif decision.get('action') == 'initiate_experiment':
        # Handle experiment initiation
        hypothesis = decision.get('hypothesis')
        reason = decision.get('reason')
        logging.info(f"Initiating experiment for hypothesis: {hypothesis}")
        return decision

    elif decision.get('action') == 'wait':
        # Explicit wait action
        reason = decision.get('reason', 'No specific reason provided')
        logging.info(f"Waiting: {reason}")
        return decision

    return {"action": "wait", "reason": "LLM returned unknown or no action."}
