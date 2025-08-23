import logging
import json
import re
import random
from typing import Dict, Any, List, Optional
from modules.decision_engine.planner import GoalPlanner, plan_from_context
from core.llm import call_llm
from ..agent_self_reflection.self_modification import generate_hypothesis, analyze_experiment_outcome

# Import the new conversational reflection module
from modules.agent_self_reflection.conversational_reflection import ConversationalReflectionModule

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def enhanced_goal_driven_decision_maker_loop(situation, memory=None, model=None, rag_context=None, 
                                           hypotheses=None, shared_state=None, conversational_insights=None):
    """
    An enhanced goal-driven decision-making loop that incorporates conversational insights
    and manages the full experimentation cycle.
    """
    planner = GoalPlanner()
    goals = planner.get_goals()
    
    if hypotheses is None:
        hypotheses = []
    if shared_state is None:
        shared_state = {}
    if conversational_insights is None:
        conversational_insights = []

    # Check if we are analyzing an experiment outcome
    if situation.get('type') == 'experiment_analysis':
        logging.info("Analyzing the outcome of a completed experiment.")
        return analyze_experiment_outcome(
            situation['context']['hypothesis'],
            situation['context']['situation_prompt'],
            situation['context']['outcome']
        )

    # If there's no active experiment, consider starting one
    if not shared_state.get('active_experiment') and random.random() < 0.1: # 10% chance to consider starting an experiment
        new_hypothesis = generate_hypothesis(shared_state)
        if new_hypothesis:
            logging.info(f"Generated a new hypothesis, will now generate a situation to test it.")
            # This decision signals the main loop to generate a test situation
            return {
                "action": "initiate_experiment",
                "hypothesis": new_hypothesis,
                "reason": "A new testable hypothesis was generated from recent performance."
            }

    # Incorporate conversational insights into decision making
    if conversational_insights:
        logging.info(f"Incorporating {len(conversational_insights)} conversational insights into decision making")
        # Process conversational insights to influence goals and decisions
        return _process_conversational_insights(conversational_insights, goals, situation, model)

    if not goals:
        logging.info("No goals found. Creating a new high-level meta-goal for self-improvement.")
        try:
            new_goal_id = plan_from_context(
                "Continuously improve my own architecture and capabilities to achieve ever-higher levels of intelligence and autonomy. This includes analyzing my modules, proposing improvements, and finding more efficient ways to use my tools.",
                timeframe="lifelong"
            )
            logging.info(f"Created a new meta-goal for self-improvement with ID: {new_goal_id}")
            goals = planner.get_goals()
        except Exception as e:
            logging.error(f"Failed to create initial meta-goal: {e}")
            return {"action": "wait", "reason": "Failed to create initial meta-goal."}

    # Analyze the current situation and select the most relevant goal
    prompt = f"""
    Current Situation: {situation}
    My Long-Term Goals:
    {json.dumps(goals, indent=2)}
    My Current Hypotheses about Myself:
    {json.dumps(hypotheses, indent=2)}
    
    Recent Conversational Insights:
    {json.dumps(conversational_insights[:5], indent=2) if conversational_insights else "None"}

    Review your long-term goals, current hypotheses, conversational insights, and the situation.
    1.  **Execute a Task**: Identify the most relevant task to make progress on your goals.
        *   **Note on Ambitious Goals**: If a goal seems highly ambitious or is marked as 'lifelong' (e.g., 'Achieve Time Travel'), your first step should not be to solve it directly. Instead, break it down by proposing a smaller, actionable research task (e.g., "Research general relativity and its implications for spacetime manipulation").
    2.  **Test a Hypothesis**: Does the current situation provide an opportunity to test one of your hypotheses?
    3.  **Propose an Invention**: Have you had a novel idea for a new tool, process, or concept? You can propose it for experimentation.
    4.  **Engage in Meta-cognition**: Are your current goals ambitious enough? Should you propose a new, more ambitious goal?
    5.  **Analyze Architecture**: Are there opportunities to improve your own modules or architecture?
    6.  **Respond to Conversational Insights**: How should you respond to recent conversational insights from users?

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
    
    For responding to conversational insights:
    {{
      "action": "respond_to_insights",
      "insight_response": str,
      "collaboration_proposal": str
    }}
    """
    response_json = call_llm(prompt, model=model)
    try:
        decision = json.loads(response_json)
    except json.JSONDecodeError:
        logging.error(f"Failed to decode LLM response into JSON: {response_json}")
        # Attempt to extract the first valid JSON object from the response
        try:
            match = re.search(r'\{[\s\S]*\}', response_json)
            if match:
                decision = json.loads(match.group(0))
            else:
                return {"action": "wait", "reason": "LLM returned no valid action."}
        except json.JSONDecodeError:
            logging.error("Failed to extract and decode JSON from LLM response.")
            return {"action": "wait", "reason": "Failed to extract and decode JSON from LLM response."}


    if decision.get('action') == 'task':
        # Execute the chosen task
        task_description = decision['task_description']
        logging.info(f"Executing task: {task_description}")
        # Here, you would integrate with the part of your AGI that executes tasks.
        # For now, we'll just log it and mark the task as complete.
        try:
            planner.complete_task(decision['goal_id'], decision['subgoal_id'], decision['task_id'])
            logging.info(f"Task {decision['task_id']} completed.")
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
        logging.info(f"Proposing a new invention for testing: {decision.get('invention_description')}")
        return decision
        
    elif decision.get('action') == 'respond_to_insights':
        # Handle response to conversational insights
        insight_response = decision.get('insight_response', '')
        collaboration_proposal = decision.get('collaboration_proposal', '')
        logging.info(f"Responding to conversational insights: {insight_response}")
        logging.info(f"Collaboration proposal: {collaboration_proposal}")
        return decision

    return {"action": "wait", "reason": "LLM returned unknown or no action."}
    
def _process_conversational_insights(conversational_insights: List[Dict[str, Any]], 
                                   goals: List[Dict[str, Any]], 
                                   situation: Dict[str, Any], 
                                   model: Optional[str] = None) -> Dict[str, Any]:
    """
    Process conversational insights to influence decision making and goal planning.
    
    Args:
        conversational_insights: List of insights from conversations
        goals: Current goals
        situation: Current situation
        model: LLM model to use
        
    Returns:
        Decision dictionary
    """
    try:
        # Create a prompt to process conversational insights
        prompt = f"""
You are RAVANA, an advanced AI system making decisions based on conversational insights.
Analyze the following conversational insights and determine how they should influence your goals and decisions.

Current Situation: {json.dumps(situation, indent=2)}
Current Goals: {json.dumps(goals, indent=2)}

Conversational Insights:
{json.dumps(conversational_insights, indent=2)}

Instructions:
1. Identify which insights are most relevant to your current goals
2. Determine if any insights suggest new goals or modifications to existing goals
3. Consider if any insights suggest collaborative opportunities
4. Propose specific actions to address the most important insights

Respond with a JSON object indicating the appropriate action to take.
"""
        
        response_json = call_llm(prompt, model=model)
        try:
            decision = json.loads(response_json)
            return decision
        except json.JSONDecodeError:
            logging.error(f"Failed to decode LLM response into JSON: {response_json}")
            return {"action": "wait", "reason": "Failed to process conversational insights."}
            
    except Exception as e:
        logging.error(f"Error processing conversational insights: {e}")
        return {"action": "wait", "reason": f"Error processing conversational insights: {e}"}