import logging
import json
import re
import random
from modules.decision_engine.planner import GoalPlanner, plan_from_context
from core.llm import call_llm
from ..agent_self_reflection.self_modification import generate_hypothesis, analyze_experiment_outcome

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def goal_driven_decision_maker_loop(situation, memory=None, model=None, rag_context=None, hypotheses=None, shared_state=None):
    """
    A goal-driven decision-making loop that now manages the full experimentation cycle.
    """
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

    # Enhanced decision-making prompt with structured framework
    prompt = f"""
[ROLE DEFINITION]
You are an autonomous AI agent making decisions to achieve your objectives with enhanced reasoning capabilities.

[CONTEXT]
Current situation: {situation}
Active goals: {json.dumps(goals, indent=2)}
Current hypotheses: {json.dumps(hypotheses, indent=2)}

[TASK INSTRUCTIONS]
Make an optimal decision by following this structured approach:
1. Analyze the situation and identify key factors
2. Evaluate alignment with goals and hypotheses
3. Consider multiple approaches and their implications
4. Assess risks and potential outcomes
5. Select the optimal action with clear justification

[REASONING FRAMEWORK]
Apply systematic analysis to your decision-making:
1. Decompose the problem into manageable components
2. Evaluate each option against success criteria
3. Consider short-term and long-term consequences
4. Account for uncertainty and incomplete information
5. Validate reasoning against logical consistency

[OUTPUT REQUIREMENTS]
Provide a JSON-formatted response with these fields:
- analysis: Detailed situation analysis with key factors identified
- reasoning: Step-by-step reasoning leading to decision
- confidence: Numerical confidence score (0.0-1.0)
- risk_assessment: Potential risks and mitigation strategies
- action: Selected action with parameters

[SAFETY CONSTRAINTS]
- Ensure actions align with ethical principles
- Avoid decisions with catastrophic risk potential
- Consider impact on system stability and reliability
- Validate against established safety protocols

Your response should be a JSON object with one of the following structures.
You have access to a list of actions/tools you can use. If you decide to execute a task, the 'task_description' should be a clear instruction for what to do, potentially using one of your available tools.

For executing a task:
{{
  "analysis": "Detailed analysis of the situation and options",
  "reasoning": "Step-by-step reasoning leading to this decision",
  "confidence": 0.8,
  "risk_assessment": "Potential risks and mitigation strategies",
  "action": "task",
  "goal_id": int,
  "subgoal_id": int,
  "task_id": int,
  "task_description": str
}}

For testing a hypothesis:
{{
  "analysis": "Analysis of hypothesis testing opportunity",
  "reasoning": "Reasoning for why this is a good hypothesis to test",
  "confidence": 0.7,
  "risk_assessment": "Risks in testing this hypothesis and mitigations",
  "action": "test_hypothesis",
  "hypothesis_to_test": str,
  "test_method_description": str,
  "expected_outcome": str
}}

For proposing a new invention (uses the 'propose_and_test_invention' action):
{{
  "analysis": "Analysis of invention opportunity",
  "reasoning": "Reasoning for why this invention is valuable",
  "confidence": 0.9,
  "risk_assessment": "Technical and practical risks",
  "action": "propose_and_test_invention",
  "invention_description": "A detailed description of the new concept or invention.",
  "test_plan_suggestion": "A suggestion for how to test the viability of this invention."
}}

For proposing a new, more ambitious goal:
{{
  "analysis": "Analysis of goal expansion opportunity",
  "reasoning": "Reasoning for why a more ambitious goal is appropriate",
  "confidence": 0.85,
  "risk_assessment": "Risks in pursuing more ambitious goals",
  "action": "new_goal",
  "new_goal_context": str
}}

For proposing a self-improvement plan:
{{
  "analysis": "Analysis of self-improvement opportunity",
  "reasoning": "Reasoning for why this self-improvement is needed",
  "confidence": 0.9,
  "risk_assessment": "Risks in self-modification",
  "action": "self_improvement",
  "plan_description": str,
  "modules_to_improve": list[str]
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
            # Try multiple approaches to extract JSON
            match = re.search(r'\{[\s\S]*\}', response_json)
            if match:
                # Clean up the JSON string
                json_str = match.group(0)
                # Fix common JSON issues
                # Add quotes to keys
                json_str = re.sub(r'(\w+):', r'"\1":', json_str)
                # Remove trailing commas
                json_str = re.sub(r',\s*}', '}', json_str)
                # Remove trailing commas
                json_str = re.sub(r',\s*\]', ']', json_str)
                decision = json.loads(json_str)
            else:
                # Try to create a minimal valid decision
                decision = {"action": "wait",
                            "reason": "LLM returned no valid action."}
        except json.JSONDecodeError:
            logging.error(
                "Failed to extract and decode JSON from LLM response.")
            # Create a fallback decision
            decision = {
                "action": "wait", "reason": "Failed to extract and decode JSON from LLM response."}
        except Exception as e:
            logging.error(f"Unexpected error during JSON extraction: {e}")
            decision = {"action": "wait",
                        "reason": f"Unexpected error during JSON extraction: {e}"}

    # Validate the decision structure
    if not isinstance(decision, dict) or "action" not in decision:
        logging.warning(f"LLM returned invalid decision structure: {decision}")
        decision = {"action": "wait",
                    "reason": "Invalid decision structure from LLM."}

    action = decision.get('action')

    # Handle different action types with better error handling
    try:
        if action == 'task':
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

        elif action == 'test_hypothesis':
            # Log the hypothesis test plan
            hypothesis = decision['hypothesis_to_test']
            test_method = decision['test_method_description']
            logging.info(f"Proposing to test hypothesis: '{hypothesis}'")
            logging.info(f"Test method: {test_method}")
            # In a more advanced version, this could trigger a specific experiment or a new goal.
            return decision

        elif action == 'new_goal':
            # Create a new goal and plan
            new_goal_context = decision['new_goal_context']
            logging.info(
                f"Creating a new goal and plan for: {new_goal_context}")
            try:
                new_goal_id = plan_from_context(new_goal_context)
                logging.info(f"New goal created with ID: {new_goal_id}")
                decision['new_goal_id'] = new_goal_id
                return decision
            except Exception as e:
                logging.error(f"Failed to create new goal from context: {e}")
                return {"action": "wait", "reason": f"Failed to create new goal from context: {e}"}

        elif action == 'self_improvement':
            # Log the self-improvement plan
            plan_description = decision['plan_description']
            modules_to_improve = decision.get('modules_to_improve', [])
            logging.info(f"Proposed self-improvement plan: {plan_description}")
            logging.info(f"Modules to improve: {modules_to_improve}")
            # In a more advanced version, this could trigger a new goal or a series of tasks.
            return decision

        elif action == 'propose_and_test_invention':
            # This decision will be caught by the ActionManager, which will execute
            # the ProposeAndTestInventionAction. We just return the decision.
            logging.info(
                f"Proposing a new invention for testing: {decision.get('invention_description')}")
            return decision

        elif action == 'initiate_experiment':
            # Handle experiment initiation
            hypothesis = decision.get('hypothesis')
            reason = decision.get('reason')
            logging.info(f"Initiating experiment for hypothesis: {hypothesis}")
            return decision

        elif action == 'wait':
            # Explicit wait action
            reason = decision.get('reason', 'No specific reason provided')
            logging.info(f"Waiting: {reason}")
            return decision

        else:
            logging.warning(f"LLM returned unknown action: {action}")
            return {"action": "wait", "reason": f"Unknown action '{action}' from LLM."}

    except Exception as e:
        logging.error(
            f"Error processing decision action '{action}': {e}", exc_info=True)
        return {"action": "wait", "reason": f"Error processing decision: {e}"}
