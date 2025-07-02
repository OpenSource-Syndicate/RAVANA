import os
import json
from typing import List, Dict, Optional

GOALS_DB_FILE = os.path.join(os.path.dirname(__file__), 'goals_db.json')

class GoalPlanner:
    def __init__(self, db_file: str = GOALS_DB_FILE):
        self.db_file = db_file
        self.goals = self._load_goals()

    def _load_goals(self) -> List[Dict]:
        if not os.path.exists(self.db_file):
            return []
        with open(self.db_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_goals(self):
        with open(self.db_file, 'w', encoding='utf-8') as f:
            json.dump(self.goals, f, indent=2)

    def add_goal(self, title: str, description: str = "", timeframe: str = "month") -> int:
        goal = {
            "id": len(self.goals) + 1,
            "title": title,
            "description": description,
            "timeframe": timeframe,
            "subgoals": [],
            "completed": False
        }
        self.goals.append(goal)
        self._save_goals()
        return goal["id"]

    def add_subgoal(self, goal_id: int, title: str, description: str = "") -> int:
        goal = self._find_goal(goal_id)
        if goal is None:
            raise ValueError("Goal not found")
        subgoal = {
            "id": len(goal["subgoals"]) + 1,
            "title": title,
            "description": description,
            "tasks": [],
            "completed": False
        }
        goal["subgoals"].append(subgoal)
        self._save_goals()
        return subgoal["id"]

    def add_task(self, goal_id: int, subgoal_id: int, description: str) -> int:
        subgoal = self._find_subgoal(goal_id, subgoal_id)
        if subgoal is None:
            raise ValueError("Subgoal not found")
        task = {
            "id": len(subgoal["tasks"]) + 1,
            "description": description,
            "completed": False
        }
        subgoal["tasks"].append(task)
        self._save_goals()
        return task["id"]

    def complete_task(self, goal_id: int, subgoal_id: int, task_id: int):
        task = self._find_task(goal_id, subgoal_id, task_id)
        if task is None:
            raise ValueError("Task not found")
        task["completed"] = True
        self._save_goals()
        self._check_subgoal_completion(goal_id, subgoal_id)

    def _check_subgoal_completion(self, goal_id: int, subgoal_id: int):
        subgoal = self._find_subgoal(goal_id, subgoal_id)
        if subgoal and all(t["completed"] for t in subgoal["tasks"]):
            subgoal["completed"] = True
            self._save_goals()
            self._check_goal_completion(goal_id)

    def _check_goal_completion(self, goal_id: int):
        goal = self._find_goal(goal_id)
        if goal and all(sg["completed"] for sg in goal["subgoals"]):
            goal["completed"] = True
            self._save_goals()
            # Optionally: generate new objectives here

    def get_goals(self) -> List[Dict]:
        return self.goals

    def get_goal(self, goal_id: int) -> Optional[Dict]:
        return self._find_goal(goal_id)

    def _find_goal(self, goal_id: int) -> Optional[Dict]:
        for g in self.goals:
            if g["id"] == goal_id:
                return g
        return None

    def _find_subgoal(self, goal_id: int, subgoal_id: int) -> Optional[Dict]:
        goal = self._find_goal(goal_id)
        if goal:
            for sg in goal["subgoals"]:
                if sg["id"] == subgoal_id:
                    return sg
        return None

    def _find_task(self, goal_id: int, subgoal_id: int, task_id: int) -> Optional[Dict]:
        subgoal = self._find_subgoal(goal_id, subgoal_id)
        if subgoal:
            for t in subgoal["tasks"]:
                if t["id"] == task_id:
                    return t
        return None

def plan_from_context(context: str, timeframe: str = "month", model: str = None) -> int:
    """
    Given a string context (e.g., "Learn reinforcement learning basics"), use the LLM to decompose it into a goal,
    subgoals, and tasks, and populate the planner DB. Returns the new goal's ID.
    """
    from llm import call_llm
    planner = GoalPlanner()
    prompt = f"""
    Given the following high-level objective, break it down into a hierarchical plan with subgoals and tasks.
    Format the output as JSON with this structure:
    ```
    {{
      "goal": {{"title": str, "description": str}},
      "subgoals": [
        {{
          "title": str,
          "description": str,
          "tasks": [str, ...]
        }}, ...
      ]
    }}
    ```
    Objective: {context}
    """
    response = call_llm(prompt, model=model)
    import re
    import json as pyjson
    # Try to extract JSON from triple backticks first
    code_block = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
    if code_block:
        plan_json = code_block.group(1)
    else:
        # fallback: extract first JSON-like block
        match = re.search(r'\{[\s\S]*\}', response)
        if match:
            plan_json = match.group(0)
        else:
            raise ValueError("Could not extract plan JSON from LLM response")
    try:
        plan = pyjson.loads(plan_json)
    except Exception:
        # fallback: try to fix common JSON issues
        plan = pyjson.loads(plan_json.replace("'", '"'))
    goal = plan["goal"]
    goal_id = planner.add_goal(goal["title"], goal.get("description", ""), timeframe)
    for subgoal in plan["subgoals"]:
        subgoal_id = planner.add_subgoal(goal_id, subgoal["title"], subgoal.get("description", ""))
        for task_desc in subgoal["tasks"]:
            planner.add_task(goal_id, subgoal_id, task_desc)
    return goal_id

# Example usage:
# new_goal_id = plan_from_context("Implement an image classifier for cats vs dogs")
# print(GoalPlanner().get_goal(new_goal_id))

# Example usage/demo
if __name__ == "__main__":
    planner = GoalPlanner()
    goal_id = plan_from_context("Learn reinforcement learning basics")
    print(planner.get_goal(goal_id))
    print(planner.get_goals())