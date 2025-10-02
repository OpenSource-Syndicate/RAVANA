"""
Self-Goal Management Module for RAVANA AGI System

This module enables the AGI to set, track, and pursue its own improvement goals.
"""
import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

class GoalPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class GoalStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"
    OVERDUE = "overdue"

@dataclass
class SelfGoal:
    """Represents a self-set goal for the AGI"""
    id: str
    title: str
    description: str
    category: str  # performance, learning, capability, etc.
    priority: GoalPriority
    status: GoalStatus
    created_at: datetime
    target_date: datetime
    current_progress: float  # 0.0 to 1.0
    progress_details: List[Dict[str, Any]]  # Track progress steps
    dependencies: List[str]  # IDs of other goals this depends on
    metrics: Dict[str, float]  # Success metrics and targets
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['id'] = self.id
        result['priority'] = self.priority.value
        result['status'] = self.status.value
        result['created_at'] = self.created_at.isoformat()
        result['target_date'] = self.target_date.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create SelfGoal from dictionary"""
        return cls(
            id=data['id'],
            title=data['title'],
            description=data['description'],
            category=data['category'],
            priority=GoalPriority(data['priority']),
            status=GoalStatus(data['status']),
            created_at=datetime.fromisoformat(data['created_at']),
            target_date=datetime.fromisoformat(data['target_date']),
            current_progress=data['current_progress'],
            progress_details=data['progress_details'],
            dependencies=data['dependencies'],
            metrics=data['metrics']
        )

class SelfGoalManager:
    """Manages self-set goals for the AGI"""
    
    def __init__(self, storage_path: str = "self_goals.json"):
        self.storage_path = Path(storage_path)
        self.goals: Dict[str, SelfGoal] = {}
        self.goal_history: List[SelfGoal] = []
        
        logger.info("Self Goal Manager initialized")
    
    def create_goal(self, title: str, description: str, category: str, 
                   priority: GoalPriority, target_date: datetime, 
                   metrics: Dict[str, float], dependencies: List[str] = None) -> SelfGoal:
        """Create a new self-goal"""
        goal_id = str(uuid.uuid4())
        
        goal = SelfGoal(
            id=goal_id,
            title=title,
            description=description,
            category=category,
            priority=priority,
            status=GoalStatus.PENDING,
            created_at=datetime.utcnow(),
            target_date=target_date,
            current_progress=0.0,
            progress_details=[],
            dependencies=dependencies or [],
            metrics=metrics
        )
        
        self.goals[goal_id] = goal
        logger.info(f"Created new goal: {title} (ID: {goal_id})")
        
        return goal
    
    def update_goal_progress(self, goal_id: str, progress: float, 
                           details: Dict[str, Any] = None) -> bool:
        """Update progress on a goal"""
        if goal_id not in self.goals:
            return False
        
        goal = self.goals[goal_id]
        goal.current_progress = min(1.0, max(0.0, progress))
        
        # Add progress detail if provided
        if details:
            details['timestamp'] = datetime.utcnow().isoformat()
            goal.progress_details.append(details)
        
        # Update status based on progress
        if goal.current_progress >= 1.0:
            goal.status = GoalStatus.COMPLETED
            self.goal_history.append(self.goals.pop(goal_id))
            logger.info(f"Goal completed: {goal.title}")
        elif goal.status == GoalStatus.PENDING and progress > 0.0:
            goal.status = GoalStatus.IN_PROGRESS
            logger.info(f"Goal in progress: {goal.title}")
        
        return True
    
    def get_goals_by_category(self, category: str) -> List[SelfGoal]:
        """Get all goals in a specific category"""
        return [goal for goal in self.goals.values() if goal.category == category]
    
    def get_goals_by_priority(self, priority: GoalPriority) -> List[SelfGoal]:
        """Get all goals with a specific priority"""
        return [goal for goal in self.goals.values() if goal.priority == priority]
    
    def get_goals_by_status(self, status: GoalStatus) -> List[SelfGoal]:
        """Get all goals with a specific status"""
        return [goal for goal in self.goals.values() if goal.status == status]
    
    def get_overdue_goals(self) -> List[SelfGoal]:
        """Get all goals that are past their target date and update their status to OVERDUE if needed"""
        now = datetime.utcnow()
        overdue_goals = []
        
        for goal in self.goals.values():
            # Check if goal is past its target date and not completed
            if goal.target_date < now and goal.status != GoalStatus.COMPLETED:
                # Update status to OVERDUE if it's not already
                if goal.status != GoalStatus.OVERDUE:
                    goal.status = GoalStatus.OVERDUE
                    logger.info(f"Updated goal '{goal.title}' status to OVERDUE (target date: {goal.target_date})")
                
                overdue_goals.append(goal)
        
        return overdue_goals
    
    def get_high_priority_goals(self) -> List[SelfGoal]:
        """Get all high priority or critical goals"""
        return [goal for goal in self.goals.values() 
                if goal.priority in [GoalPriority.HIGH, GoalPriority.CRITICAL]]
    
    async def save_goals(self):
        """Save goals to persistent storage"""
        try:
            with open(self.storage_path, 'w') as f:
                goals_data = [goal.to_dict() for goal in self.goals.values()]
                history_data = [goal.to_dict() for goal in self.goal_history]
                
                data = {
                    'active_goals': goals_data,
                    'goal_history': history_data,
                    'last_updated': datetime.utcnow().isoformat()
                }
                
                json.dump(data, f, indent=2)
            
            logger.info(f"Goals saved to {self.storage_path}")
        except Exception as e:
            logger.error(f"Error saving goals: {e}")
    
    async def load_goals(self):
        """Load goals from persistent storage"""
        try:
            if not self.storage_path.exists():
                logger.info(f"Goals file {self.storage_path} does not exist, starting fresh")
                return
            
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                
                # Load active goals
                for goal_data in data.get('active_goals', []):
                    goal = SelfGoal.from_dict(goal_data)
                    self.goals[goal.id] = goal
                
                # Load goal history
                for goal_data in data.get('goal_history', []):
                    goal = SelfGoal.from_dict(goal_data)
                    self.goal_history.append(goal)
            
            logger.info(f"Loaded {len(self.goals)} active goals and {len(self.goal_history)} from history")
        except Exception as e:
            logger.error(f"Error loading goals: {e}")
    
    def get_performance_insights(self) -> Dict[str, Any]:
        """Get insights about goal achievement performance"""
        total_goals = len(self.goal_history)
        if total_goals == 0:
            return {
                "completion_rate": 0,
                "average_completion_time": 0,
                "success_categories": {},
                "performance_trend": "insufficient_data"
            }
        
        completed_goals = [goal for goal in self.goal_history if goal.status == GoalStatus.COMPLETED]
        completion_rate = len(completed_goals) / total_goals if total_goals > 0 else 0
        
        # Calculate average time to completion
        completion_times = []
        for goal in completed_goals:
            time_to_completion = (goal.progress_details[-1]['timestamp'] if goal.progress_details 
                                  else goal.created_at) - goal.created_at
            completion_times.append(time_to_completion.total_seconds())
        
        avg_completion_time = sum(completion_times) / len(completion_times) if completion_times else 0
        
        # Breakdown by category
        category_success = {}
        for goal in completed_goals:
            if goal.category not in category_success:
                category_success[goal.category] = {"completed": 0, "attempted": 0}
            category_success[goal.category]["completed"] += 1
        
        for goal in self.goal_history:
            if goal.category not in category_success:
                category_success[goal.category] = {"completed": 0, "attempted": 0}
            category_success[goal.category]["attempted"] += 1
        
        # Calculate success rates by category
        for category, stats in category_success.items():
            stats["success_rate"] = stats["completed"] / stats["attempted"] if stats["attempted"] > 0 else 0
        
        # Determine performance trend
        recent_goals = sorted(self.goal_history, key=lambda g: g.created_at, reverse=True)[:10]
        if len(recent_goals) >= 5:
            recent_completion_rate = len([g for g in recent_goals if g.status == GoalStatus.COMPLETED]) / len(recent_goals)
            overall_rate = completion_rate
            
            if recent_completion_rate > overall_rate * 1.1:
                trend = "improving"
            elif recent_completion_rate < overall_rate * 0.9:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "completion_rate": round(completion_rate, 2),
            "average_completion_time": round(avg_completion_time / 3600, 2),  # In hours
            "success_categories": category_success,
            "performance_trend": trend,
            "total_goals_tracked": total_goals
        }

# Global instance
self_goal_manager = SelfGoalManager()