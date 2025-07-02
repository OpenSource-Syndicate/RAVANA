# Hierarchical Goal Planner (planner.py)

This module provides a basic hierarchical planner inspired by Hierarchical Task Networks (HTN) for long-term AI planning (weeks/months). The planner allows the AI to:
- Set high-level goals (e.g., "Learn reinforcement learning basics")
- Break each goal into subgoals and tasks
- Track progress and completion
- Automatically generate new objectives as tasks are completed

Goals, subgoals, and tasks are stored in a JSON file (`goals_db.json`) for persistence.

## Basic Usage Example
```python
from planner import GoalPlanner

planner = GoalPlanner()
gid = planner.add_goal("Learn reinforcement learning basics", "Study RL over the next month", "month")
sgid = planner.add_subgoal(gid, "Understand Q-learning", "Read papers and tutorials on Q-learning")
tid = planner.add_task(gid, sgid, "Read Sutton & Barto RL book chapter on Q-learning")
print(planner.get_goals())
planner.complete_task(gid, sgid, tid)
print(planner.get_goals())
```

This enables the AI to plan, decompose, and track long-term objectives in a structured, extensible way.
