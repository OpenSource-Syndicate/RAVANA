import logging
import json
import re
from modules.decision_engine.planner import GoalPlanner, plan_from_context
from modules.decision_engine.llm import call_llm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def goal_driven_decision_maker_loop(situation, memory=None, model=None, rag_context=None):
    """
    A goal-driven decision-making loop that uses a GoalPlanner to manage and execute goals.
    """
    planner = GoalPlanner()
    goals = planner.get_goals()

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

    Review your long-term goals and the current situation.
    1.  **Identify the most relevant task** to make progress on your goals. This could be a task from your existing plan or a new task that needs to be created.
    2.  **Engage in meta-cognition:** Are your current goals ambitious enough? Is there a more challenging, "tougher" goal you should be pursuing? Propose a new, more ambitious goal if you believe it's time to raise the bar.
    3.  **Analyze your own architecture:** Based on the current situation and your long-term goals, are there any opportunities to improve your own modules or architecture? Could you be using your tools more effectively? Propose a plan for self-improvement if you identify an opportunity.

    Your response should be a JSON object with one of the following structures:

    For executing a task:
    {{
      "action": "task",
      "goal_id": int,
      "subgoal_id": int,
      "task_id": int,
      "task_description": str
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

    return {"action": "wait", "reason": "LLM returned unknown or no action."}
