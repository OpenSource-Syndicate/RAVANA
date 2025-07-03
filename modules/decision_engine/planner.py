import logging
import time
import uuid
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# In-memory storage for goals, shared across all instances of GoalPlanner
_goals: Dict[str, Dict[str, Any]] = {}

def plan_from_context(context: str, timeframe: str = "short-term", priority: int = 5) -> str:
    """
    Creates a new, simple goal from a given context and stores it in memory.
    This function does not call an LLM, making it fast and reliable for initial planning.
    """
    logger.info("--> [Planner] INPUT: Creating a new plan from context.")
    logger.debug(f"Context: '{context}', Timeframe: {timeframe}, Priority: {priority}")
    
    goal_id = str(uuid.uuid4())
    goal = {
        "id": goal_id,
        "title": context,
        "description": f"A goal to address the context: {context}",
        "timeframe": timeframe,
        "priority": priority,
        "status": "pending",
        "sub_goals": [],
        "context": context,
        "created_at": time.time(),
        "updated_at": time.time()
    }
    _goals[goal_id] = goal
    
    logger.info(f"<-- [Planner] OUTPUT: Created new goal with ID: {goal_id}")
    return goal_id

class GoalPlanner:
    def __init__(self):
        """
        A simple in-memory goal planner that manages goals created by plan_from_context.
        """
        logger.info("[Planner] Initialized GoalPlanner.")
        self._goals = _goals  # Use the shared in-memory dictionary

    def get_goal(self, goal_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a goal by its string UUID.
        """
        logger.info(f"--> [Planner] INPUT: Retrieving goal with ID: {goal_id}")
        goal = self._goals.get(goal_id)
        if goal:
            logger.info(f"<-- [Planner] OUTPUT: Found goal titled: '{goal.get('title')}'")
        else:
            logger.warning(f"<-- [Planner] OUTPUT: Goal with ID {goal_id} not found.")
        return goal

    def get_all_goals(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieves all goals, optionally filtered by status.
        """
        logger.info(f"--> [Planner] INPUT: Retrieving all goals with status filter: {status}")
        if status:
            filtered_goals = [g for g in self._goals.values() if g.get('status') == status]
            logger.info(f"<-- [Planner] OUTPUT: Found {len(filtered_goals)} goals with status '{status}'.")
            return filtered_goals
        logger.info(f"<-- [Planner] OUTPUT: Found {len(self._goals)} total goals.")
        return list(self._goals.values())

    def update_goal_status(self, goal_id: str, status: str) -> bool:
        """
        Updates the status of a goal (e.g., "pending", "in_progress", "completed").
        """
        logger.info(f"--> [Planner] INPUT: Updating goal {goal_id} to status: '{status}'")
        if goal_id in self._goals:
            self._goals[goal_id]['status'] = status
            self._goals[goal_id]['updated_at'] = time.time()
            logger.info(f"<-- [Planner] OUTPUT: Goal {goal_id} status updated successfully.")
            return True
        logger.warning(f"<-- [Planner] OUTPUT: Failed to update status for goal {goal_id} - not found.")
        return False

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

    def get_goals(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieves all goals, optionally filtered by status.
        """
        logger.info(f"--> [Planner] INPUT: Retrieving all goals with status: {status}")
        if status:
            filtered_goals = [g for g in self._goals.values() if g['status'] == status]
            logger.info(f"<-- [Planner] OUTPUT: Found {len(filtered_goals)} goals with status {status}.")
            return filtered_goals
        logger.info(f"<-- [Planner] OUTPUT: Found {len(self._goals)} total goals.")
        return list(self._goals.values())

    def add_sub_goal(self, parent_goal_id: str, sub_goal_description: str) -> Optional[str]:
        """
        Adds a sub-goal to a parent goal.
        """
        logger.info(f"--> [Planner] INPUT: Adding sub-goal to parent {parent_goal_id}.")
        logger.debug(f"Sub-goal description: {sub_goal_description}")
        
        if parent_goal_id in self._goals:
            sub_goal_id = str(uuid.uuid4())
            sub_goal = {
                "id": sub_goal_id,
                "title": sub_goal_description,
                "description": "",
                "tasks": [],
                "completed": False,
                "status": "pending",
                "context": self._goals[parent_goal_id]['context']
            }
            self._goals[parent_goal_id]['sub_goals'].append(sub_goal)
            logger.info(f"<-- [Planner] OUTPUT: Added sub-goal {sub_goal_id} to parent {parent_goal_id}.")
            return sub_goal_id
            
        logger.warning(f"<-- [Planner] OUTPUT: Failed to add sub-goal to parent {parent_goal_id} - not found.")
        return None

# Example usage:
# new_goal_id = plan_from_context("Implement an image classifier for cats vs dogs")
# print(GoalPlanner().get_goal(new_goal_id))

# Example usage/demo
if __name__ == "__main__":
    planner = GoalPlanner()
    goal_id = plan_from_context("Learn reinforcement learning basics")
    print(planner.get_goal(goal_id))
    print(planner.get_goals())